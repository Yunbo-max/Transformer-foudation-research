"""Predicted-Skip on GDN-2: retrieval-aware JOINT erase/write suppression.

baseline : original GDN-2 recurrence
write    : earlier Predicted-Skip, only scale w
joint    : NEW main candidate, scale BOTH b and w
shuffled : shuffle value between batch rows for negative control
keyread  : current Key replaces previous Query, temporal ablation

No labels are given to gate. Optional 'official' backend invokes NVIDIA's
unchanged chunk_gdn2, but uses this small matched-projection LM wrapper.
"""
import math,sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

ARMS=("baseline","write","joint","shuffled","keyread")

class GDN2PredSkipMixer(nn.Module):
    def __init__(self,hidden,heads=2,key_dim=16,value_dim=None,chunk=16,
                 mode="joint",gamma=2.,backend="reference",official_root=None,
                 beta_guard=False,gate_memory_detach=True):
        super().__init__()
        assert mode in ARMS and backend in ("reference","official")
        self.heads=heads;self.kdim=key_dim;self.vdim=value_dim or key_dim
        self.chunk=chunk;self.mode=mode;self.backend=backend
        self.official_root=official_root;self.beta_guard=beta_guard
        self.gate_memory_detach=gate_memory_detach
        self.q_proj=nn.Linear(hidden,heads*key_dim,bias=False)
        self.k_proj=nn.Linear(hidden,heads*key_dim,bias=False)
        self.v_proj=nn.Linear(hidden,heads*self.vdim,bias=False)
        self.erase_proj=nn.Linear(hidden,heads*key_dim)
        self.write_proj=nn.Linear(hidden,heads*self.vdim)
        self.decay_proj=nn.Linear(hidden,heads*key_dim)
        self.output=nn.Linear(heads*self.vdim,hidden,bias=False)
        self.log_gamma=nn.Parameter(torch.tensor(math.log(gamma)))
        self._official_kernel=None
        self.last_rho=None

    def official_kernel(self):
        if self._official_kernel is None:
            if not self.official_root:
                raise RuntimeError("official backend requires --official-root")
            root=Path(self.official_root).expanduser().resolve()
            if not (root/"lit_gpt/gdn2_ops/chunk_gdn2.py").is_file():
                raise FileNotFoundError("expected NVIDIA GatedDeltaNet-2 checkout")
            sys.path.insert(0,str(root))
            from lit_gpt.gdn2_ops.chunk_gdn2 import chunk_gdn2
            self._official_kernel=chunk_gdn2
        return self._official_kernel

    def _reference_chunk(self,q,k,v,g,b,w,state):
        ys=[]
        for t in range(q.shape[2]):
            kt,vt,qt=k[:,:,t],v[:,:,t],q[:,:,t]
            prior=state*torch.exp(g[:,:,t]).unsqueeze(-1)
            erase=torch.einsum("bhk,bhkv->bhv",b[:,:,t]*kt,prior)
            edit=w[:,:,t]*vt-erase
            state=prior+kt.unsqueeze(-1)*edit.unsqueeze(-2)
            ys.append(torch.einsum("bhkv,bhk->bhv",state,qt)*(self.kdim**-0.5))
        return torch.stack(ys,2),state

    def _official_chunk(self,q,k,v,g,b,w,state):
        kernel=self.official_kernel()
        out,final=kernel(q.permute(0,2,1,3).contiguous(),
            k.permute(0,2,1,3).contiguous(),
            v.permute(0,2,1,3).contiguous(),
            g.permute(0,2,1,3).contiguous(),
            b.permute(0,2,1,3).contiguous(),
            w.permute(0,2,1,3).contiguous(),
            initial_state=state.float(),output_final_state=True,
            use_qk_l2norm_in_kernel=True,beta_guard=self.beta_guard)
        return out.permute(0,2,1,3),final

    def forward(self,x):
        B,T,_=x.shape;H=self.heads;K=self.kdim;V=self.vdim
        q=F.normalize(self.q_proj(x).view(B,T,H,K),dim=-1,eps=1e-6).permute(0,2,1,3)
        k=F.normalize(self.k_proj(x).view(B,T,H,K),dim=-1,eps=1e-6).permute(0,2,1,3)
        v=F.silu(self.v_proj(x).view(B,T,H,V)).permute(0,2,1,3)
        b=torch.sigmoid(self.erase_proj(x).view(B,T,H,K)).permute(0,2,1,3)
        w=torch.sigmoid(self.write_proj(x).view(B,T,H,V)).permute(0,2,1,3)
        g=(-.02*F.softplus(self.decay_proj(x).view(B,T,H,K))).permute(0,2,1,3)
        prev_q=torch.cat((torch.zeros_like(q[:,:,:1]),q[:,:,:-1]),dim=2)
        state=x.new_zeros((B,H,K,V),dtype=torch.float32 if self.backend=="official" else x.dtype)
        outs=[];gatevals=[]
        gamma=self.log_gamma.exp().clamp(max=20)
        for start in range(0,T,self.chunk):
            stop=min(T,start+self.chunk)
            qq=q[:,:,start:stop];kk=k[:,:,start:stop];vv=v[:,:,start:stop]
            gg=g[:,:,start:stop];bb=b[:,:,start:stop];ww=w[:,:,start:stop]
            if self.mode!="baseline":
                S0=state.detach() if self.gate_memory_detach else state
                probe=kk if self.mode=="keyread" else prev_q[:,:,start:stop]
                read=torch.einsum("bhkv,bhtk->bhtv",S0.to(probe.dtype),probe)
                compare=vv.roll(1,dims=0) if self.mode=="shuffled" and B>1 else vv
                score=F.cosine_similarity(read,compare,dim=-1,eps=1e-6)
                rho=torch.exp(-gamma*F.relu(score))
                gatevals.append(rho.detach())
                if self.mode=="write":ww=ww*rho.unsqueeze(-1)
                else:
                    bb=bb*rho.unsqueeze(-1)
                    ww=ww*rho.unsqueeze(-1)
            if self.backend=="official":
                y,state=self._official_chunk(qq,kk,vv,gg,bb,ww,state)
            else:y,state=self._reference_chunk(qq,kk,vv,gg,bb,ww,state)
            outs.append(y)
        y=torch.cat(outs,dim=2).permute(0,2,1,3).reshape(B,T,H*V)
        self.last_rho=(torch.cat(gatevals,dim=2).detach() if gatevals and not self.training else None)
        stats=dict(skip_mean=float(torch.cat(gatevals,dim=2).mean().detach()) if gatevals else 1.,
                   gamma=float(gamma.detach()))
        return self.output(y),stats

def edit_step(state,k,v,b,w,g=None):
    """Exact one-token GDN-2 edit, S[K,V], for mathematical unit tests."""
    decay=torch.exp(g) if g is not None else torch.ones_like(k)
    prior=decay[:,None]*state
    return prior+torch.outer(k,w*v-(b*k)@prior)
