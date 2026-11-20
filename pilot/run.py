#!/usr/bin/env python3
"""<=3-hour causal mechanism screen for historical-query-protected writes.

No learned LM is claimed. First stage is a transparent GDN-2-equation reference
to test identifiability of scalar gates versus matrix-direction protection.
Optional --official-root performs a kernel-parity check on NVIDIA's unchanged
chunk_gdn2. No custom CUDA or Triton is written.
"""
import argparse
import json
import math
from pathlib import Path
import time
import torch
import torch.nn.functional as F

def unit(x):
    return x / x.norm().clamp_min(1e-12)

def gdn2_step(S,k,v,b,w):
    # Key-major S[K,V]; g=0, D=I; official rank-one erase/write semantics.
    return S - torch.outer(k,(b*k) @ S) + torch.outer(k,w*v)

def full_geometry_step(S,k,v,C,mu=8.,lam=1.,tau=1.):
    H=lam*torch.eye(k.numel(),dtype=S.dtype) + mu*C
    u=torch.linalg.solve(H,k)
    return S+torch.outer(u,v-k@S)/(tau+k@u)

def setup(seed,dim=8,aligned=False):
    g=torch.Generator().manual_seed(seed)
    Q,_=torch.linalg.qr(torch.randn(dim,dim,generator=g,dtype=torch.float64))
    e=Q.T
    S=torch.zeros(dim,dim,dtype=torch.float64)
    S+=torch.outer(e[0],e[0])+torch.outer(e[1],e[1])
    # Equal past write frequency, unequal future read usage.
    Ck=.5*torch.outer(e[0],e[0])+.5*torch.outer(e[1],e[1])
    # Seeds vary actual task difficulty, not only rotate coordinates.
    p_hi=0.80+0.15*torch.rand((),generator=g).item()
    mA=0.30+0.45*torch.rand((),generator=g).item()
    mB=0.30+0.45*torch.rand((),generator=g).item()
    Cq=p_hi*torch.outer(e[0],e[0])+(1-p_hi)*torch.outer(e[1],e[1])
    if aligned: Cq=Ck.clone()
    kA=unit(e[0]+mA*e[2])
    kB=unit(e[1]+mB*e[3])
    vA=e[4];vB=e[5]
    return S,e,Cq,Ck,(kA,kB),(vA,vB)

def run_new_writes(seed,method,mu=8.,beta=.35,nsteps=32,aligned=False):
    S,e,Cq,Ck,keys,values=setup(seed,aligned=aligned)
    S0=S.clone()
    # identical interleaved interfering writes for all arms
    for t in range(nsteps):
        j=t%2
        k,v=keys[j],values[j]
        if method=="full_query":
            S=full_geometry_step(S,k,v,Cq,mu=mu)
        elif method=="full_key":
            S=full_geometry_step(S,k,v,Ck,mu=mu)
        else:
            C=Cq if method=="query_gate" else Ck
            r=1. if method=="gdn2" else 1/(1+mu*(k@C@k))
            gate=torch.full_like(k,beta*r)
            wg=torch.full_like(v,beta*r)
            S=gdn2_step(S,k,v,gate,wg)
    # Do NOT select a method by old-recall only; report learning of NEW facts.
    p_hi=float(e[0]@Cq@e[0])
    old=p_hi*((e[0]@S)-(e[0]@S0)).square().sum().item()+(1-p_hi)*((e[1]@S)-(e[1]@S0)).square().sum().item()
    new=sum(((keys[j]@S)-values[j]).square().sum().item() for j in range(2))/2
    return {"old_readout_mse":old,"new_fact_mse":new}

def repeat_answer_oracle(seed,steps=32):
    S,e,_,_,_,_=setup(seed)
    old=S.clone()
    k=e[1]                # value-token key need not be queried key e[0]
    q=e[0]
    recorded=[]
    for typ in ("plain","oracle","predskip"):
        state=S.clone()
        for _ in range(steps):
            retrieved=q@state
            value=retrieved.clone()  # answer reappears, NOT a novel mapping for key k
            similarity=float(F.cosine_similarity(retrieved[None],value[None]))
            rho=0. if typ=="oracle" else math.exp(-2*max(0.,similarity)) if typ=="predskip" else 1.
            b=torch.full_like(k,.35*rho);w=torch.full_like(value,.35*rho)
            state=gdn2_step(state,k,value,b,w)
        old_loss=((state-old).square().sum()).item()
        important=((e[1]@state)-(e[1]@old)).square().sum().item()
        recorded.append({"arm":typ,"state_drift":old_loss,"other_fact_mse":important})
    return recorded

def contrast(seed,aligned=False):
    strengths=[0.,.25,.5,1.,2.,4.,8.,16.,32.,64.]
    result={}
    for arm in ("gdn2","key_gate","query_gate","full_key","full_query"):
        rows=[]
        for mu in strengths:
            rows.append(dict(mu=mu,**run_new_writes(seed,arm,mu=mu,aligned=aligned)))
        result[arm]=rows
    # Matched-plasticity retention: for a given new fact error threshold, choose
    # each method's best old error among feasible strength settings.
    thresholds=(.10,.20,.40,.80)
    matched={}
    for th in thresholds:
        matched[str(th)]={}
        for arm,rows in result.items():
            eligible=[r for r in rows if r["new_fact_mse"]<=th]
            matched[str(th)][arm]=min((r["old_readout_mse"] for r in eligible),default=None)
    return result,matched

def official_parity(repo_path):
    """Unmodified NVIDIA kernel: one-step parity plus 32 controlled writes.

    No learned model or nontrivial decay is used in this diagnostic. beta_guard
    is set False here to isolate the published recurrence and match the exact
    reference; normal model training should keep its default safeguard.
    """
    import sys
    root=Path(repo_path).expanduser().resolve()
    if not (root/"lit_gpt/gdn2_ops/chunk_gdn2.py").exists():
        raise FileNotFoundError("expected official NVlabs/GatedDeltaNet-2 checkout")
    if not torch.cuda.is_available():
        raise RuntimeError("official kernel requires CUDA")
    sys.path.insert(0,str(root))
    from lit_gpt.gdn2_ops.chunk_gdn2 import chunk_gdn2
    dtype=torch.bfloat16
    dev="cuda"
    K=32;V=32;T=64
    torch.manual_seed(101)
    k=F.normalize(torch.randn(K,device=dev,dtype=dtype),dim=0)
    v=torch.randn(V,device=dev,dtype=dtype)
    q=torch.zeros(1,T,1,K,device=dev,dtype=dtype)
    q[:,0,0]=k
    ks=torch.zeros_like(q);ks[:,0,0]=k
    vs=torch.zeros(1,T,1,V,device=dev,dtype=dtype)
    vs[:,0,0]=v
    b=torch.zeros_like(ks);b[:,0,0]=.35
    w=torch.zeros_like(vs);w[:,0,0]=.35
    g=torch.zeros_like(ks)
    init=torch.zeros(1,1,K,V,device=dev,dtype=torch.float32)
    torch.cuda.synchronize();t0=time.perf_counter()
    with torch.no_grad():
        _,fin=chunk_gdn2(q,ks,vs,g,b,w,initial_state=init,
            output_final_state=True,use_qk_l2norm_in_kernel=True,
            beta_guard=False)
    torch.cuda.synchronize()
    expected=.35*torch.outer(k.float(),v.float())
    err=(fin[0,0].float()-expected).abs().max().item()
    if not math.isfinite(err) or err>0.04:
        raise AssertionError(f"official one-step parity error {err}")
    controlled=[]
    for seed in (17,23,42):
        S,e,Cq,Ck,keys,values=setup(seed)
        for arm in ("gdn2","query_gate","key_gate"):
            q=torch.zeros(1,T,1,K,device=dev,dtype=dtype)
            kk=torch.zeros_like(q)
            vv=torch.zeros(1,T,1,V,device=dev,dtype=dtype)
            bb=torch.zeros_like(q)
            ww=torch.zeros_like(vv)
            for t in range(32):
                j=t%2
                k8=keys[j];v8=values[j]
                key=torch.cat([k8,torch.zeros(K-8,dtype=S.dtype)])
                value=torch.cat([v8,torch.zeros(V-8,dtype=S.dtype)])
                C=Cq if arm=="query_gate" else Ck
                factor=1. if arm=="gdn2" else 1/(1+8*(k8@C@k8))
                kk[0,t,0]=key.to(device=dev,dtype=dtype)
                q[0,t,0]=kk[0,t,0]
                vv[0,t,0]=value.to(device=dev,dtype=dtype)
                bb[0,t,0]=.35*float(factor)
                ww[0,t,0]=.35*float(factor)
            initial=torch.zeros(1,1,K,V,device=dev,dtype=torch.float32)
            initial[0,0,:8,:8]=S.float().to(dev)
            with torch.no_grad():
                _,h=chunk_gdn2(q,kk,vv,torch.zeros_like(q),bb,ww,
                    initial_state=initial,output_final_state=True,
                    use_qk_l2norm_in_kernel=True,beta_guard=False)
            reference=run_new_writes(seed,arm,mu=8.,beta=.35,nsteps=32)
            Sm=h[0,0,:8,:8].double().cpu()
            p_hi=float(e[0]@Cq@e[0])
            old=p_hi*((e[0]@Sm)-(e[0]@S)).square().sum().item()
            old+=(1-p_hi)*((e[1]@Sm)-(e[1]@S)).square().sum().item()
            new=sum(((keys[j]@Sm)-values[j]).square().sum().item() for j in range(2))/2
            diff=max(abs(reference["old_readout_mse"]-old),abs(reference["new_fact_mse"]-new))
            controlled.append(dict(seed=seed,arm=arm,old_readout_mse=old,
                new_fact_mse=new,reference_max_metric_error=diff))
            if diff>0.15:
                raise AssertionError(f"official/reference mismatch {seed} {arm}: {diff}")
    return {"status":"PASS","max_abs_state_error":err,
            "controlled_trials":controlled,
            "elapsed_seconds":time.perf_counter()-t0,"official_repo":str(root)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--budget-seconds",type=int,default=10500)
    p.add_argument("--official-root",default=None)
    p.add_argument("--seeds",nargs="+",type=int,default=[17,23,42])
    p.add_argument("--output",default=".pilot_workspace_branch/branch_reports")
    a=p.parse_args()
    assert 0<a.budget_seconds<=10600
    root=Path(__file__).resolve().parent.parent
    out=root/a.output;out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    data={"type":"reference-equations","disclaimer":"NOT trained GDN-2 LM",
      "seeds":[],"official":{"status":"NOT_REQUESTED"},"decision":"INCONCLUSIVE"}
    for seed in a.seeds:
        if time.monotonic()-start>a.budget_seconds-30:break
        grid,matched=contrast(seed)
        aligned_grid,aligned_matched=contrast(seed,aligned=True)
        null_gap=max(abs(a["old_readout_mse"]-b["old_readout_mse"])
            for a,b in zip(aligned_grid["key_gate"],aligned_grid["query_gate"]))
        if null_gap>1e-8:
            raise AssertionError("query/key aligned negative control must tie")
        data["seeds"].append(dict(seed=seed,grid=grid,matched=matched,
                        aligned_negative_control_gap=null_gap,
                        redundancy=repeat_answer_oracle(seed)))
        (out/"results.json").write_text(json.dumps(data,indent=2))
    if a.official_root:
        try:data["official"]=official_parity(a.official_root)
        except Exception as e:
            data["official"]={"status":"BLOCKED","error":repr(e)}
    # Gate-only novel claim must win at MATCHED plasticity, not by disabling writes.
    diffs=[]
    full_diffs=[]
    for s in data["seeds"]:
        m=s["matched"]
        for th in (".2",".4"):
            a1=m[th]["query_gate"];b1=m[th]["key_gate"]
            fq=m[th]["full_query"];fk=m[th]["full_key"]
            if a1 is not None and b1 is not None:diffs.append(b1-a1)
            if fq is not None and fk is not None:full_diffs.append(fk-fq)
    data["statistics"]={"gate_query_minus_key_retention_advantages":diffs,
       "full_query_minus_key_retention_advantages":full_diffs}
    if len(data["seeds"])!=len(a.seeds) or not diffs:
        data["decision"]="INCONCLUSIVE"
    elif min(diffs)>0.02:
        data["decision"]="POTENTIAL_QUERY_GATE_ADVANTAGE_VALIDATE_TRAINED_GDN2"
    elif full_diffs and max(full_diffs)>0.02:
        data["decision"]="DIRECTION_MATTERS_GATE_ONLY_NOT_SUFFICIENT"
    else:
        data["decision"]="NO_GO_FOR_THIS_CONTROLLED_SETTING"
    data["elapsed_seconds"]=time.monotonic()-start
    (out/"results.json").write_text(json.dumps(data,indent=2))
    print(json.dumps({"decision":data["decision"],"official":data["official"],
                      "statistics":data["statistics"],"elapsed_s":data["elapsed_seconds"]},indent=2))
if __name__=="__main__":
    main()
