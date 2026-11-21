"""Latest SG-Delta: a SINGLE recurrent state with joint Query Write+Decay.
No fork, routing or stale chunk-start read. Readable PyTorch reference, not
an optimized NVIDIA kernel. Key-major S[B,H,K,V].
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

ARMS=("isotropic","query-write","query-decay","query-joint",
      "key-joint","channel-decay","reverse-joint")

class SGDeltaMixer(nn.Module):
    def __init__(self,hidden,heads=2,key_dim=16,value_dim=None,chunk=16,
                 mode="query-joint",mu=8.,lam=1.,tau=1.,ema=.98,
                 detach_statistics=True):
        super().__init__()
        assert mode in ARMS and chunk>0 and heads>0 and hidden>0
        assert mu>=0 and lam>0 and tau>0 and 0<=ema<1
        self.heads=heads;self.kdim=key_dim;self.vdim=value_dim or key_dim
        self.chunk=chunk;self.mode=mode;self.mu=mu;self.lam=lam
        self.tau=tau;self.ema=ema;self.detach_statistics=detach_statistics
        self.q_proj=nn.Linear(hidden,heads*key_dim,bias=False)
        self.k_proj=nn.Linear(hidden,heads*key_dim,bias=False)
        self.v_proj=nn.Linear(hidden,heads*self.vdim,bias=False)
        self.output=nn.Linear(heads*self.vdim,hidden,bias=False)
        self.fast_gate=nn.Linear(hidden,heads)
        self.slow_gate=nn.Linear(hidden,heads)
        self.channel_gate=nn.Linear(hidden,heads*key_dim)
        nn.init.constant_(self.fast_gate.bias,1.)
        nn.init.constant_(self.slow_gate.bias,2.)

    def forward(self,x):
        B,T,_=x.shape;H=self.heads;K=self.kdim;V=self.vdim
        q=F.normalize(self.q_proj(x).view(B,T,H,K),dim=-1,eps=1e-6).permute(0,2,1,3)
        k=F.normalize(self.k_proj(x).view(B,T,H,K),dim=-1,eps=1e-6).permute(0,2,1,3)
        v=self.v_proj(x).view(B,T,H,V).permute(0,2,1,3)
        fast=(.86+.11*torch.sigmoid(self.fast_gate(x))).permute(0,2,1)
        slow=fast+(1-fast)*torch.sigmoid(self.slow_gate(x)).permute(0,2,1)
        channel=(.86+.13*torch.sigmoid(self.channel_gate(x).view(B,T,H,K))).permute(0,2,1,3)
        state=x.new_zeros((B,H,K,V))
        C=x.new_zeros((B,H,K,K))
        eye=torch.eye(K,device=x.device,dtype=x.dtype).view(1,1,K,K)
        outputs=[];cosines=[];Pmeans=[]
        for start in range(0,T,self.chunk):
            stop=min(T,start+self.chunk);L=stop-start
            use_C=self.mode in ("query-write","query-decay","query-joint",
                                 "key-joint","reverse-joint")
            Cprior=C.detach() if self.detach_statistics else C
            Hmat=(self.lam*eye+self.mu*Cprior) if use_C else self.lam*eye.expand(B,H,K,K)
            factor=torch.linalg.cholesky(Hmat)
            Hinv=torch.cholesky_solve(eye.expand(B,H,K,K),factor)
            P=eye-self.lam*Hinv
            Pmeans.append(P.diagonal(dim1=-2,dim2=-1).mean())
            ku=k[:,:,start:stop].transpose(-1,-2)
            if self.mode in ("query-write","query-joint","key-joint","reverse-joint"):
                direction=torch.cholesky_solve(ku,factor).transpose(-1,-2)
            else:direction=ku.transpose(-1,-2)/self.lam
            norm=(direction*k[:,:,start:stop]).sum(-1,keepdim=True)
            direction=direction/(self.tau+norm)
            cosines.append(F.cosine_similarity(direction,k[:,:,start:stop],dim=-1).mean())
            for j,t in enumerate(range(start,stop)):
                ft=fast[:,:,t];st=slow[:,:,t]
                if self.mode in ("query-decay","query-joint","key-joint","reverse-joint"):
                    protect=P if self.mode!="reverse-joint" else eye-P
                    D=ft[...,None,None]*eye+(st-ft)[...,None,None]*protect
                elif self.mode=="channel-decay":
                    D=torch.diag_embed(channel[:,:,t])
                else:D=ft[...,None,None]*eye
                before=D@state
                kt=k[:,:,t];vt=v[:,:,t];qt=q[:,:,t]
                error=vt-torch.einsum("bhkv,bhk->bhv",before,kt)
                state=before+direction[:,:,j].unsqueeze(-1)*error.unsqueeze(-2)
                outputs.append(torch.einsum("bhkv,bhk->bhv",state,qt))
            if use_C:
                obs=q[:,:,start:stop] if self.mode!="key-joint" else k[:,:,start:stop]
                if self.detach_statistics:obs=obs.detach()
                powers=self.ema**torch.arange(L-1,-1,-1,device=x.device,dtype=x.dtype)
                obs2=torch.einsum("bhlk,bhlj,l->bhkj",obs,obs,(1-self.ema)*powers)
                C=self.ema**L*Cprior+obs2
        out=torch.stack(outputs,dim=2).permute(0,2,1,3).reshape(B,T,H*V)
        stats=dict(write_cosine=float(torch.stack(cosines).mean().detach()),
                   importance_fraction=float(torch.stack(Pmeans).mean().detach()),
                   fast_decay=float(fast.mean().detach()),
                   slow_decay=float(slow.mean().detach()))
        return self.output(out),stats
