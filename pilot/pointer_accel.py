"""Causal candidate-sparse lookup for Identity/Pointer heads.

Reference PyTorch implementation, not a fused kernel. Block routing only uses
COMPLETED historical blocks; the active block is searched exactly. This
prevents future-key leakage into routing centroids. Dense-topk is a strong
baseline but still computes all pairs; do not count it as sparse FLOPs.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def _validate(q,k,v,query_positions):
    if (q.ndim!=3 or k.ndim!=3 or q.shape[0]!=k.shape[0] or
        q.shape[-1]!=k.shape[-1] or v.shape[:2]!=k.shape[:2]):
        raise ValueError("Expected q[B,Q,d], k[B,N,d], v[B,N,dv]")
    if query_positions is None and q.shape[1]!=k.shape[1]:
        raise ValueError("explicit query_positions required for cross-attention")


def pointer_lookup(q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,
                   *,mode:str="dense",block_size:int=32,blocks:int=2,
                   topk:int=64,chunk:int=32,
                   history:torch.Tensor|None=None,
                   historical_keys:torch.Tensor|None=None,
                   penalty:float=0.,
                   query_positions:torch.Tensor|None=None):
    """Return causal pointer output and actual routing/matching counts.

    For mode=block, each query scores all COMPLETED block centroids, selects
    at most 'blocks' of them, and refines those keys plus its local active
    block.  No [Q,N] logits exist in block mode. A block's centroid is
    NEVER visible to a query inside that block before the block completes.

    No-backtrack penalty uses history/history_keys and is computed only over
    selected candidates. Gradients pass through selected q/k/v but not
    discrete routing indices. Block-mean routing may miss rare matching keys.
    """
    _validate(q,k,v,query_positions)
    if mode not in ("dense","dense_topk","block"):raise ValueError("invalid mode")
    if min(block_size,blocks,topk,chunk)<1:raise ValueError("invalid budget")
    if (history is None)!=(historical_keys is None):
        raise ValueError("supply both no-backtrack tensors")
    if history is not None and (history.shape[:2]!=q.shape[:2] or
        historical_keys.shape[:2]!=k.shape[:2] or
        history.shape[-1]!=historical_keys.shape[-1]):
        raise ValueError("history shape mismatch")
    b,nq,d=q.shape
    nk=k.shape[1]
    if query_positions is None:
        positions=torch.arange(nq,device=q.device)[None].expand(b,-1)
    else:
        positions=query_positions.to(device=q.device,dtype=torch.long)
        if positions.shape==(nq,):positions=positions[None].expand(b,-1)
        if positions.shape!=(b,nq):raise ValueError("query_positions must be [B,Q] or [Q]")
    scale=d**-.5
    if mode in ("dense","dense_topk"):
        scores=q@k.transpose(1,2)*scale
        if history is not None and penalty:
            scores=scores-penalty*(F.normalize(history,dim=-1,eps=1e-8)@
               F.normalize(historical_keys,dim=-1,eps=1e-8).transpose(1,2))
        valid=torch.arange(nk,device=q.device)[None,None,:]<=positions[:,:,None]
        scores=scores.masked_fill(~valid,float("-inf"))
        if not valid.any(-1).all():raise ValueError("query has no causal key")
        if mode=="dense":
            return scores.softmax(-1)@v,dict(mode=mode,scoring_pairs=b*nq*nk,
              retained_pairs=int(valid.sum()),router_pairs=0,
              max_logit_cells=b*nq*nk)
        values,ids=scores.topk(k=min(nk,topk),dim=-1)
        chosen=v[torch.arange(b,device=q.device)[:,None,None],ids]
        out=(values.softmax(-1).unsqueeze(-1)*chosen).sum(-2)
        return out,dict(mode=mode,scoring_pairs=b*nq*nk,
            retained_pairs=int(torch.isfinite(values).sum()),router_pairs=0,
            max_logit_cells=b*nq*nk)

    bs=block_size; groups=(nk+bs-1)//bs
    pad=groups*bs-nk
    blocks_k=F.pad(k,(0,0,0,pad)).reshape(b,groups,bs,d)
    group_count=torch.full((groups,),bs,device=q.device,dtype=k.dtype)
    if pad:group_count[-1]-=pad
    centers=blocks_k.sum(2)/group_count[None,:,None]
    ends=torch.minimum((torch.arange(groups,device=q.device)+1)*bs-1,
                        torch.tensor(nk-1,device=q.device))
    outputs=[]
    routed=peak=0
    # Hold counter tensors on GPU until after the final chunk.
    actual_counts=[]
    for first in range(0,nq,chunk):
        stop=min(nq,first+chunk);width=stop-first
        pos=positions[:,first:stop]
        qc=q[:,first:stop]
        local_base=(pos//bs)*bs
        routing=qc@centers.transpose(1,2)*scale
        fully_old=ends[None,None,:]<local_base[:,:,None]
        routing=routing.masked_fill(~fully_old,float("-inf"))
        chosen=routing.topk(min(blocks,groups),dim=-1).indices
        prev=(chosen[...,None]*bs+
              torch.arange(bs,device=q.device)).reshape(b,width,-1)
        local=local_base[:,:,None]+torch.arange(bs,device=q.device)[None,None,:]
        candidates=torch.cat([prev,local],dim=-1)
        complete=ends[chosen].repeat_interleave(bs,dim=-1)<local_base[:,:,None]
        valid=torch.cat([(prev<nk)&complete,
                         (local<nk)&(local<=pos[:,:,None])],dim=-1)
        if not valid.any(-1).all():raise ValueError("query has no causal sparse key")
        candidates=candidates.clamp(max=nk-1)
        ib=torch.arange(b,device=q.device)[:,None,None]
        ks=k[ib,candidates];vs=v[ib,candidates]
        scores=(qc.unsqueeze(-2)*ks).sum(-1)*scale
        if history is not None and penalty:
            h=F.normalize(history[:,first:stop],dim=-1,eps=1e-8)
            hk=F.normalize(historical_keys[ib,candidates],dim=-1,eps=1e-8)
            scores=scores-penalty*(h.unsqueeze(-2)*hk).sum(-1)
        scores=scores.masked_fill(~valid,float("-inf"))
        outputs.append((scores.softmax(-1).unsqueeze(-1)*vs).sum(-2))
        actual_counts.append(valid.sum())
        routed+=b*width*groups
        peak=max(peak,scores.numel()+routing.numel())
    scored=int(torch.stack(actual_counts).sum().item())
    return torch.cat(outputs,dim=1),dict(
        mode=mode,scoring_pairs=scored,retained_pairs=scored,
        router_pairs=routed,max_logit_cells=peak,
        group_count=groups,selected_blocks=blocks,
        note="Only Pointer operator is sparse; no fused CUDA kernel or dense-LM speedup claimed")
