"""Causally indexed sign-LSH candidate routing for Identity-style Pointer heads.

Unlike a block MEAN, identical normalized keys/queries share the same hash.
This is a REFERENCE Python/Torch implementation: bucket sorting O(NlogN),
gather and searchsorted are not optimized or fused. Hash itself is prior art.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


@torch.no_grad()
def _build_hash_index(q,k,bits,seed):
    if bits<1 or bits>16:raise ValueError("hash_bits must be 1..16")
    generator=torch.Generator().manual_seed(seed)
    plane=torch.randn(k.shape[-1],bits,generator=generator).to(k.device,k.dtype)
    weights=(1<<torch.arange(bits,device=k.device,dtype=torch.long))
    kh=((k@plane>=0).long()*weights).sum(-1)
    projection=q@plane
    qh=((projection>=0).long()*weights).sum(-1)
    return kh,qh,projection


def hash_pointer_lookup(q,k,v,positions,scale,history,historical_keys,
                        penalty,bits=8,probes=2,bucket=32,window=8,
                        seed=721,chunk=32):
    """Approximate causal sparse pointer lookup using a prefix-safe hash index.

    Hash keys and each query with fixed random hyperplanes, try exact hash
    plus nearest Hamming-one probes, read at most bucket candidate keys
    plus recent window, then apply unchanged original QK and optional
    no-backtrack scores. No full QN score matrix is materialized.
    A future key can never change a past query's result: hash planes are
    fixed; all bucket lookups are bounded by each query position.
    """
    if min(probes,bucket,window)<1 or probes>bits+1:
        raise ValueError("invalid hash parameters")
    b,nq,d=q.shape;nk=k.shape[1]
    kh,qh,projection=_build_hash_index(q,k,bits,seed)
    ids=torch.arange(nk,device=q.device)
    packed=kh*(nk+1)+ids[None,:]
    sorted_packed,order=packed.sort(dim=-1)
    sorted_pos=ids[None].expand(b,-1).gather(1,order)
    if probes>1:
        uncertainty=projection.abs().topk(probes-1,dim=-1,largest=False).indices
        toggled=qh[...,None]^(1<<uncertainty)
        codes=torch.cat([qh[...,None],toggled],dim=-1)
    else:codes=qh[...,None]
    quota=max(1,bucket//probes)
    outputs=[];counts=[];router_pairs=0;peak=0
    for first in range(0,nq,chunk):
        stop=min(nq,first+chunk);nc=stop-first
        qc=q[:,first:stop];pos=positions[:,first:stop]
        own_codes=codes[:,first:stop]
        low=own_codes*(nk+1)
        high=low+(pos[:,:,None]+1).clamp(max=nk)
        lo=torch.searchsorted(sorted_packed.contiguous(),
             low.flatten(1).contiguous(),right=False).reshape_as(low)
        hi=torch.searchsorted(sorted_packed.contiguous(),
             high.flatten(1).contiguous(),right=False).reshape_as(high)
        source=hi[...,None]-1-torch.arange(quota,device=q.device)[None,None,None,:]
        present=source>=lo[...,None]
        batch=torch.arange(b,device=q.device)[:,None,None,None]
        candidate_hash=sorted_pos[batch,source.clamp(0,nk-1)].reshape(b,nc,-1)
        present=present.reshape(b,nc,-1)
        local=pos[:,:,None]-torch.arange(window,device=q.device)[None,None,:]
        candidates=torch.cat([candidate_hash,local.clamp(0,nk-1)],dim=-1)
        valid=torch.cat([present,(local>=0)&(local<nk)],dim=-1)
        valid=valid&(candidates<=pos[:,:,None])
        # Dedup without materializing O(b*nq*bucket^2) comparisons.
        ranked,perm=candidates.masked_fill(~valid,nk).sort(dim=-1)
        distinct=torch.ones_like(ranked,dtype=torch.bool)
        distinct[...,1:]=ranked[...,1:]!=ranked[...,:-1]
        valid=torch.zeros_like(valid).scatter(-1,perm,distinct&(ranked<nk))
        if not valid.any(-1).all():
            raise ValueError("hash routing found no causal candidates")
        batch2=torch.arange(b,device=q.device)[:,None,None]
        ks=k[batch2,candidates];vs=v[batch2,candidates]
        scores=(qc.unsqueeze(-2)*ks).sum(-1)*scale
        if history is not None and penalty:
            h=F.normalize(history[:,first:stop],dim=-1,eps=1e-8)
            old=F.normalize(historical_keys[batch2,candidates],dim=-1,eps=1e-8)
            scores=scores-penalty*(h.unsqueeze(-2)*old).sum(-1)
        scores=scores.masked_fill(~valid,float("-inf"))
        outputs.append((scores.softmax(-1).unsqueeze(-1)*vs).sum(-2))
        counts.append(valid.sum())
        router_pairs+=b*nc*probes*bits
        peak=max(peak,scores.numel()+candidates.numel())
    scored=int(torch.stack(counts).sum().item())
    return torch.cat(outputs,dim=1),dict(
        mode="hash",scoring_pairs=scored,retained_pairs=scored,
        router_pairs=router_pairs,max_logit_cells=peak,
        index_build_items=b*nk,hash_bits=bits,hash_probes=probes,
        hash_bucket=bucket,local_window=window,
        note="Hash-based routing is existing ANN theory; reference Torch sort is not GPU-optimized")
