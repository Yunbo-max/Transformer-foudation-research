"""Common synthetic MQAR benchmark and small decoder, PyTorch only.
No official NVIDIA code is redistributed. No real LM benchmark claim.
"""
import json, time, random
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

BOS, QUERY, FACT, PAD = 0, 1, 2, 3

def generate_batch(batch=4,pairs=8,queries=8,distractors=4,alias=True,*,
                   seed=0,key_space=128,device="cpu"):
    if 2*pairs+distractors+2>key_space:
        raise ValueError("insufficient key_space for distinct facts")
    if pairs<2 or queries<2: raise ValueError("pairs and queries must >=2")
    rng=random.Random(seed)
    keys_base=4; values_base=4+key_space
    tokens=[]
    for _ in range(batch):
        keys=rng.sample(range(keys_base,keys_base+key_space),pairs+distractors+1)
        vals=rng.sample(range(values_base,values_base+key_space),pairs+distractors)
        text=[BOS]
        facts=list(zip(keys[:pairs],vals[:pairs]))
        for k,v in facts:text.extend([FACT,k,v])
        targets=[];groups=[]
        # Same number of INITIAL fact writes, intentionally skewed query usage.
        # Hot fact 0 is queried often; cold fact 1 is queried at the end.
        # Important: query tokens ALSO produce K/V writes in a recurrent LM,
        # so historical Key statistics are not held fixed end-to-end.
        # A separate controlled mathematical check must isolate that contrast.
        query_ids=[0]+[0 if rng.random()<.80 else rng.randrange(pairs)
                       for _ in range(1,queries-1)]+[1]
        if queries>=3: query_ids[-2]=0  # guarantee hot post-interference probe
        for qi,j in enumerate(query_ids):
            if qi==queries//2:
                for d in range(distractors):
                    text.extend([FACT,keys[pairs+d],vals[pairs+d]])
                # Crucial plasticity control: immediately query genuinely new facts.
                # This separates remembering old facts from rejecting all new writes.
                for d in range(min(2,distractors)):
                    text.extend([QUERY,keys[pairs+d],vals[pairs+d]])
                    targets.append(len(text)-1)
                    groups.append("new_post")
            text.extend([QUERY,facts[j][0],facts[j][1]])
            targets.append(len(text)-1)
            groups.append(("old_post_hot" if j==0 else "old_post_cold")
                          if qi>=queries//2 else "old_pre")
            if alias and qi==0:
                ak=keys[-1];av=facts[j][1]
                text.extend([FACT,ak,av]) # new Key, same old Value
                text.extend([QUERY,ak,av])
                targets.append(len(text)-1)
                groups.append("alias")
        tokens.append((text,targets,groups))
    T=len(tokens[0][0]);assert all(len(z[0])==T for z in tokens)
    x=torch.tensor([z[0] for z in tokens],dtype=torch.long,device=device)
    mask=torch.zeros((batch,T-1),dtype=torch.bool,device=device)
    group={g:torch.zeros_like(mask) for g in ("old_pre","old_post","old_post_hot","old_post_cold","alias","new_post")}
    answer_token_mask=torch.zeros((batch,T),dtype=torch.bool,device=device)
    fact_token_mask=torch.zeros_like(answer_token_mask)
    for b,(seq,targets,labels) in enumerate(tokens):
        for pos,name in zip(targets,labels):
            mask[b,pos-1]=True;group[name][b,pos-1]=True
            if name in ("old_post_hot","old_post_cold"):
                group["old_post"][b,pos-1]=True
            answer_token_mask[b,pos]=True
        for pos in range(1,len(seq)-2):
            if seq[pos]==FACT:fact_token_mask[b,pos+2]=True
    return {"tokens":x,"loss_mask":mask,"groups":group,
      "answer_token_mask":answer_token_mask,"fact_token_mask":fact_token_mask,
      "vocab_size":4+2*key_space,"length":T}

class MemoryBlock(nn.Module):
    def __init__(self,dim,mixer):
        super().__init__()
        self.norm1=nn.LayerNorm(dim);self.mixer=mixer
        self.norm2=nn.LayerNorm(dim)
        self.ff=nn.Sequential(nn.Linear(dim,4*dim),nn.SiLU(),nn.Linear(4*dim,dim))
    def forward(self,x):
        y,stats=self.mixer(self.norm1(x))
        x=x+y;x=x+self.ff(self.norm2(x))
        return x,stats

class MemoryLM(nn.Module):
    def __init__(self,vocab,dim,layers,mixer_factory):
        super().__init__()
        self.embed=nn.Embedding(vocab,dim)
        self.blocks=nn.ModuleList([MemoryBlock(dim,mixer_factory(dim,i)) for i in range(layers)])
        self.norm=nn.LayerNorm(dim);self.lm_head=nn.Linear(dim,vocab,bias=False)
    def forward(self,tokens,return_stats=False):
        x=self.embed(tokens);info=[]
        for b in self.blocks:
            x,z=b(x)
            if return_stats:info.append(z)
        logits=self.lm_head(self.norm(x))
        return (logits,info) if return_stats else logits

def answer_stats(logits,batch):
    logits=logits[:,:-1].float();target=batch["tokens"][:,1:]
    loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),
                         target.reshape(-1),reduction="none").reshape_as(target)
    pred=logits.argmax(-1);scores={}
    for name,mask in [("answer",batch["loss_mask"]),*batch["groups"].items()]:
        n=int(mask.sum())
        if n:
            scores[name+"_acc"]=float((((pred==target)&mask).sum()/n).item())
            scores[name+"_nll"]=float(((loss.detach()*mask).sum()/n).item())
            scores[name+"_n"]=n
    train_loss=(loss*batch["loss_mask"]).sum()/batch["loss_mask"].sum().clamp_min(1)
    return scores,train_loss

PRESETS={
 "smoke":dict(dim=32,layers=1,heads=1,head_dim=8,pairs=4,queries=4,distractors=1,batch=2,steps=2,chunk=8,key_space=32),
 "pilot":dict(dim=128,layers=2,heads=2,head_dim=16,pairs=8,queries=8,distractors=4,batch=8,steps=150,chunk=16,key_space=128),
 "stress":dict(dim=192,layers=3,heads=3,head_dim=32,pairs=16,queries=16,distractors=16,batch=4,steps=300,chunk=32,key_space=256),
 "100m-sizing":dict(dim=768,layers=12,heads=12,head_dim=64,pairs=16,queries=16,distractors=16,batch=1,steps=100,chunk=64,key_space=512),
 "340m-sizing":dict(dim=896,layers=24,heads=14,head_dim=64,pairs=16,queries=16,distractors=16,batch=1,steps=100,chunk=64,key_space=512,vocab_size=32000),
}

def train_case(model,cfg,*,device,seed,steps,budget_seconds,out_path):
    torch.manual_seed(seed)
    if device=="cuda":torch.cuda.reset_peak_memory_stats()
    model=model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.01)
    start=time.monotonic();curve=[];status="ok";batch=None
    for i in range(steps):
        if time.monotonic()-start>budget_seconds:
            status="timeout";break
        batch=generate_batch(batch=cfg["batch"],pairs=cfg["pairs"],queries=cfg["queries"],
            distractors=cfg["distractors"],seed=seed*1000003+i,
            key_space=cfg["key_space"],device=device)
        model.train();optimizer.zero_grad(set_to_none=True)
        logits=model(batch["tokens"])
        _,loss=answer_stats(logits,batch)
        if not torch.isfinite(loss):status="nonfinite";break
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if i%max(1,steps//10)==0 or i==steps-1:
            curve.append(dict(step=i+1,loss=float(loss.detach()),
                              seconds=time.monotonic()-start))
    model.eval();eval_rows=[]
    with torch.no_grad():
        for pairs,ds in [(cfg["pairs"],0),(cfg["pairs"],cfg["distractors"]),
                         (min(cfg["pairs"]*2,cfg["key_space"]//4),cfg["distractors"])]:
            z=generate_batch(batch=16,pairs=pairs,queries=cfg["queries"],
              distractors=ds,seed=seed+9000000+pairs*100+ds,
              key_space=cfg["key_space"],device=device)
            lg,extra=model(z["tokens"],return_stats=True)
            score,_=answer_stats(lg,z)
            score.update(pairs=pairs,distractors=ds,seq_len=z["length"])
            agg={}
            for entry in extra:
                if isinstance(entry,dict):
                    for key,val in entry.items():
                        if isinstance(val,(float,int)):
                            agg.setdefault(key,[]).append(float(val))
            score.update({key:sum(v)/len(v) for key,v in agg.items()})
            eval_rows.append(score)
    if device=="cuda":torch.cuda.synchronize()
    peak=torch.cuda.max_memory_allocated()/1e9 if device=="cuda" else None
    completed=(i+1 if status=="ok" else i)
    result=dict(status=status,seed=seed,params=sum(p.numel() for p in model.parameters()),
        config=cfg,steps_requested=steps,steps_completed=completed,
        peak_allocated_vram_gb=peak,
        train_tokens=completed*cfg["batch"]*batch["length"] if batch else 0,
        elapsed_seconds=time.monotonic()-start,curve=curve,eval=eval_rows,
        data="generated MQAR, NOT FineWeb or RULER")
    path=Path(out_path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2))
    return result
