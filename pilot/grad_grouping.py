"""Historical positive-target gradient grouping for a shared LM output head.

Preserve the coauthor's parameterisation:
    effective_weight[i] = own[i] + shared[assignment[i]]
This module ONLY selects assignment from past training statistics. It does not
change logits directly, perform optimizer steps, or inspect validation data.

Novel hypothesis to test, NOT a proven result: target-conditioned historical
gradient direction (optionally with log magnitude) may yield more transferable
output groups than k-means on current effective output weights.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def accumulate_positive_target_gradients(
    gradient_sum: torch.Tensor,
    target_count: torch.Tensor,
    targets: torch.Tensor,
    hidden: torch.Tensor,
    logits: torch.Tensor,
) -> None:
    """Accumulate exact per-target CE output-row gradients, BEFORE loss averaging.

    For each training example n with target y_n, this row's positive-target
    gradient is (softmax(logits_n)[y_n] - 1) * hidden_n.

    We deliberately do NOT include ubiquitous non-target softmax gradients.
    All contributions use ONLY already-seen TRAIN tokens. The historical
    statistic is a per-occurrence mean, not a per-step EMA.
    """
    y=targets.detach().reshape(-1).long()
    h=hidden.detach().reshape(-1,hidden.shape[-1]).float()
    z=logits.detach().reshape(-1,logits.shape[-1]).float()
    if h.shape[0]!=y.numel() or z.shape[0]!=y.numel():
        raise ValueError("targets, hidden and logits must align")
    if gradient_sum.shape!=(z.shape[-1],h.shape[-1]):
        raise ValueError("gradient_sum must be [vocab, hidden_width]")
    if target_count.shape!=(z.shape[-1],):
        raise ValueError("target_count must be [vocab]")
    logp_y=z.gather(1,y.unsqueeze(1)).squeeze(1)-torch.logsumexp(z,dim=1)
    g=(logp_y.exp()-1.).unsqueeze(1)*h
    gradient_sum.index_add_(0,y,g)
    target_count.index_add_(0,y,torch.ones_like(y,dtype=target_count.dtype))



@torch.no_grad()
def _spherical_kmeans(unit: torch.Tensor, clusters: int, seed: int=1, iters: int=10) -> torch.Tensor:
    """Deterministic cosine k-means; always NORMALIZES cluster centroids."""
    K=int(clusters)
    indices=torch.randperm(len(unit),generator=torch.Generator().manual_seed(seed))[:K].to(unit.device)
    centers=unit[indices].clone()
    for _ in range(iters):
        assignment=(unit@centers.T).argmax(dim=1)
        count=torch.bincount(assignment,minlength=K)
        totals=unit.new_zeros((K,unit.shape[-1])).index_add_(0,assignment,unit)
        candidates=F.normalize(totals,dim=1,eps=1e-9)
        centers=torch.where(count[:,None]>0,candidates,centers)
    return (unit@centers.T).argmax(dim=1)


@torch.no_grad()
def assign_by_positive_gradient(
    effective_weights: torch.Tensor,
    gradient_sum: torch.Tensor,
    target_count: torch.Tensor,
    clusters: int,
    *,
    kmeans_fn,
    min_count: int=3,
    magnitude_weight: float=0.,
) -> tuple[torch.Tensor,dict]:
    """Cluster reliable target-gradient directions, optional log magnitudes.

    1. The average gradient per correct-target OCCURRENCE avoids frequency
       scaling and step-by-step EMA decay of extremely rare tokens.
    2. Pure direction uses spherical k-means with NORMALIZED centroids.
       Direction + magnitude uses Euclidean k-means on augmented features.
    3. Optionally append standardized log mean-gradient magnitude. The
       magnitude weight is a prespecified hyperparameter (0 for direction).
    4. Tokens with fewer than min_count target hits fall back to output-row
       similarity using group-specific weight centroids from eligible tokens.
       If too few eligible rows exist, fall back to original weight k-means.
    5. The caller must use function-preserving reassignment to change the
       output-head parameter decomposition without changing effective rows.
    """
    R=effective_weights.detach().float()
    K=int(clusters)
    if K<1 or K>R.shape[0]:raise ValueError("invalid number of clusters")
    if min_count<1 or magnitude_weight<0:raise ValueError("invalid hyperparameters")
    if gradient_sum.shape!=R.shape or target_count.shape!=(R.shape[0],):
        raise ValueError("gradient history shape mismatch")

    counts=target_count.detach().float()
    means=gradient_sum.detach().float()/counts.clamp_min(1.).unsqueeze(-1)
    strengths=means.norm(dim=1)
    eligible=(counts>=min_count)&(strengths>1e-9)
    num=int(eligible.sum().item())
    if num<K:
        labels=kmeans_fn(R,K).long()
        sizes=torch.bincount(labels,minlength=K)
        return labels,dict(mode="weight_fallback",reason="fewer_reliable_words_than_clusters",
            reliable_words=num,reliable_fraction=num/len(R),
            min_group_size=int(sizes.min()),max_group_size=int(sizes.max()),
            magnitude_weight=float(magnitude_weight))

    eligible_rows=R[eligible]
    eligible_grad=means[eligible]
    unit=F.normalize(eligible_grad,dim=1,eps=1e-9)
    if magnitude_weight>0:
        logmag=strengths[eligible].clamp_min(1e-9).log()
        # Same global scale for ALL reliable words; avoid rare outlier magnitude.
        logmag=(logmag-logmag.mean())/(logmag.std(unbiased=False)+1e-6)
        logmag=logmag.clamp(-3.,3.)
        features=torch.cat([unit,magnitude_weight*logmag[:,None]],dim=1)
    else:
        features=unit

    assigned_reliable=(
        kmeans_fn(features,K) if magnitude_weight>0
        else _spherical_kmeans(unit,K)
    ).long()
    group_counts=torch.bincount(assigned_reliable,minlength=K)
    weight_centroids=R.new_zeros(K,R.shape[1]).index_add_(0,assigned_reliable,eligible_rows)
    weight_centroids/=group_counts.clamp_min(1).unsqueeze(1)
    # An empty reliable group has no weight reference. Use an actual row.
    empty=group_counts==0
    if empty.any():
        weight_centroids[empty]=eligible_rows[:K][empty]
    labels=torch.cdist(R,weight_centroids).argmin(dim=1)
    labels[eligible]=assigned_reliable

    # Diagnostic: sign of inner product between each reliable word's
    # unit gradient and its group's mean direction (not a success label).
    centroids=unit.new_zeros(K,unit.shape[1]).index_add_(0,assigned_reliable,unit)
    centroids=F.normalize(centroids,dim=-1,eps=1e-9)
    cosine=(unit*centroids[assigned_reliable]).sum(dim=1)
    sizes=torch.bincount(labels,minlength=K)
    result=dict(mode="positive_direction_magnitude" if magnitude_weight>0 else "positive_direction",
        reliable_words=num,reliable_fraction=num/len(R),
        min_group_size=int(sizes.min()),max_group_size=int(sizes.max()),
        mean_group_size=float(sizes.float().mean()),
        mean_reliable_centroid_cosine=float(cosine.mean()),
        negative_reliable_centroid_fraction=float((cosine<0).float().mean()),
        magnitude_weight=float(magnitude_weight),
        min_target_occurrences=int(min_count))
    return labels.long(),result
