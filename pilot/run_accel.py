#!/usr/bin/env python3
"""Separate and crossed sparse Pointer + online Jump-state caching studies.

S0/C0 dense no cache; S1/C0 causal block-sparse no cache;
S0/C1 dense+utility cache; S1/C1 block-sparse+utility cache.
This is a frozen synthetic edge-reader proxy; NOT trained coauthor LM speed.
A third S2=identity-hash routing arm is included because block-mean routing
may miss singleton identity keys (prior work in hashing, not a novelty claim).
"""
from __future__ import annotations
import argparse,json,math,random,time
from pathlib import Path
import torch
import torch.nn.functional as F
from pointer_accel import pointer_lookup
from jump_cache import JumpCache,make_trace,replay_trace


def sync(device):
    if device.type=="cuda":torch.cuda.synchronize(device)


@torch.no_grad()
def sparse_microbench(device,seed,lengths,block_size=16,blocks=2,dim=32):
    rows=[]
    for n in lengths:
        torch.manual_seed(seed)
        base=F.normalize(torch.randn(1,n,dim,device=device),dim=-1)
        answers=torch.tensor([random.Random(seed+110*i).randrange(i+1)
                              for i in range(n)],device=device)
        q=base[:,answers]*math.sqrt(dim)*16.
        k=base
        v=F.normalize(torch.randn(1,n,dim,device=device),dim=-1)
        dense,_=pointer_lookup(q,k,v,mode="dense")
        for mode in ("dense","dense_topk","block","hash"):
            sync(device)
            if device.type=="cuda":
                torch.cuda.reset_peak_memory_stats(device)
                before=torch.cuda.memory_allocated(device)
            else:before=0
            start=time.perf_counter()
            output,info=pointer_lookup(q,k,v,mode=mode,
                block_size=block_size,blocks=blocks,
                topk=max(1,blocks*block_size),chunk=32)
            sync(device)
            elapsed=1000*(time.perf_counter()-start)
            extra=(torch.cuda.max_memory_allocated(device)-before
                   if device.type=="cuda" else None)
            # Exclude validation decode from timed region.
            guess=(output@v.transpose(1,2)).argmax(-1)
            accuracy=float((guess==answers[None]).float().mean())
            rows.append(dict(length=n,needle_accuracy=accuracy,
                output_mae_vs_dense=float((output-dense).abs().mean()),
                runtime_ms=elapsed,peak_extra_cuda_bytes=extra,**info))
    return rows


class NeuralEdgeReader:
    """Frozen random-embedding identity lookup; evaluates sparse one-hop errors."""
    def __init__(self,successor,device,mode,*,seed,dim,block_size,blocks):
        self.device=device;self.mode=mode;self.block_size=block_size;self.blocks=blocks
        self.next=list(successor)
        torch.manual_seed(seed)
        self.emb=F.normalize(torch.randn(len(self.next),dim,device=device),dim=-1)
        self.order=torch.randperm(len(self.next),device=device)
        self.keys=self.emb[self.order][None]
        self.refresh()
        self.scoring_pairs=self.router_pairs=self.lookups=0
    def refresh(self):
        targets=torch.tensor(self.next,device=self.device,dtype=torch.long)
        self.values=self.emb[targets[self.order]][None]
    @torch.no_grad()
    def __call__(self,node):
        q=self.emb[node][None,None]*math.sqrt(self.emb.shape[-1])*16
        out,meta=pointer_lookup(q,self.keys,self.values,mode=self.mode,
            block_size=self.block_size,blocks=self.blocks,
            query_positions=torch.tensor([len(self.next)],device=self.device))
        self.scoring_pairs+=meta["scoring_pairs"]
        self.router_pairs+=meta["router_pairs"]
        self.lookups+=1
        return int((out[0,0]@self.emb.T).argmax().item())
    def update_edge(self,node,dest):
        self.next[node]=dest
        self.refresh()


@torch.no_grad()
def crossed_pilot(device,seed,num_nodes,num_queries,block_size,blocks,
                  budget_fraction,update_every=0):
    succ,events=make_trace(num_nodes,num_queries,max_hops=7,seed=seed,
                           skew=1.5,update_every=update_every)
    truth=JumpCache(succ,0,"none")
    answers=[]
    for typ,x,y in events:
        if typ=="update":truth.update_edge(x,y)
        else:answers.append(truth.query(x,y))
    rows=[]
    for sparse in ("dense","block","hash"):
        for policy in ("none","utility"):
            reader=NeuralEdgeReader(succ,device,sparse,seed=seed+42,
                 dim=32,block_size=block_size,blocks=blocks)
            slots=0 if policy=="none" else max(1,int(num_nodes*budget_fraction))
            cache=JumpCache(succ,slots,policy,seed=seed,
                state_bytes=64,edge_lookup=reader)
            correct=0;pos=0
            begin=time.perf_counter()
            for typ,x,y in events:
                if typ=="update":cache.update_edge(x,y)
                else:
                    result=cache.query(x,y)
                    correct+=int(result==answers[pos])
                    pos+=1
            sync(device)
            ms=1000*(time.perf_counter()-begin)
            rows.append(dict(sparse=sparse,cache=policy,
                exact_answer_accuracy=correct/max(1,pos),
                wall_ms=ms,neural_edge_lookups=reader.lookups,
                neural_score_pairs=reader.scoring_pairs,
                neural_router_pairs=reader.router_pairs,**cache.report()))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite",choices=["all","sparse","cache","combined"],default="all")
    p.add_argument("--device",choices=["cpu","cuda","mps"],default="cpu")
    p.add_argument("--seed",type=int,default=11)
    p.add_argument("--lengths",type=int,nargs="+",default=[128,256,512])
    p.add_argument("--block-size",type=int,default=16)
    p.add_argument("--blocks",type=int,default=2)
    p.add_argument("--cache-nodes",type=int,default=1024)
    p.add_argument("--cache-queries",type=int,default=512)
    p.add_argument("--combined-nodes",type=int,default=96)
    p.add_argument("--combined-queries",type=int,default=64)
    p.add_argument("--update-every",type=int,default=0)
    p.add_argument("--out",default=".pilot_workspace_pointer_accel/reports/accel.json")
    a=p.parse_args()
    if (not a.lengths or min(a.lengths)<1 or a.block_size<1 or a.blocks<1 or
        a.cache_nodes<4 or a.combined_nodes<4 or a.cache_queries<1 or
        a.combined_queries<1 or a.update_every<0):
        p.error("invalid dimensions")
    if a.device=="cuda" and not torch.cuda.is_available():
        p.error("CUDA unavailable")
    device=torch.device(a.device)
    report=dict(status="PROXY_ONLY",seed=a.seed,device=a.device,
        warning="Not coauthor trained LM. Sparse operator CUDA peak only; cache bytes are symbolic estimates.",
        sparse=[],cache=[],combined=[])
    if a.suite in ("all","sparse"):
        report["sparse"]=sparse_microbench(device,a.seed,a.lengths,
                                          a.block_size,a.blocks)
    if a.suite in ("all","cache"):
        succ,trace=make_trace(a.cache_nodes,a.cache_queries,seed=a.seed,
                              update_every=a.update_every)
        for ratio in (0.,.125,.25,.5):
            for policy in ("none","lru","lfu","random","level_lru","utility"):
                if ratio==0 and policy!="none":continue
                if ratio>0 and policy=="none":continue
                info=replay_trace(succ,trace,int(a.cache_nodes*ratio),
                     policy,seed=a.seed,state_bytes=64)
                report["cache"].append(dict(cache_fraction_of_nodes=ratio,**info))
    if a.suite in ("all","combined"):
        report["combined"]=crossed_pilot(device,a.seed,a.combined_nodes,
             a.combined_queries,a.block_size,a.blocks,.25,
             update_every=a.update_every)
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
