#!/usr/bin/env python3
"""Optional REAL coauthor pdg.py model-training comparison: dense vs block Pointer.

The PyTorch block operator sparsifies only internal pointer recurrences P/Q/R.
The ordinary output heads and no-backtrack residual attention remain DENSE.
This is accuracy feasibility, not full-model O(Nr) throughput.
"""
from __future__ import annotations
import argparse,json,shutil
from pathlib import Path
from core import ensure_upstream,add_seed_flag,run_cases
from integrate_pointer import patch_pdg

CONFIGS=[
    ("reverse7",7,1,0,0.0),
    ("reverse5",5,1,0,0.0),
    ("repeated7",7,1,2,0.3),
    ("random7",7,2,0,0.0),
]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device",choices=["cpu","cuda","mps"],default="cpu")
    p.add_argument("--seeds",nargs="+",type=int,default=[0,1])
    p.add_argument("--steps",type=int,default=50)
    p.add_argument("--budget-seconds",type=int,default=9600)
    p.add_argument("--work-dir",default=".pilot_workspace_pointer_sparse_model")
    p.add_argument("--block-size",type=int,default=16)
    p.add_argument("--blocks",nargs="+",type=int,default=[2,4])
    a=p.parse_args()
    if not(0<a.budget_seconds<=10600 and 0<a.steps<=200 and
           a.block_size>0 and a.blocks and min(a.blocks)>0):
        p.error("invalid budget")
    if a.device=="cuda":
        import torch
        if not torch.cuda.is_available():p.error("CUDA unavailable")
    work=Path(__file__).resolve().parent.parent/a.work_dir
    upstream=ensure_upstream(work)
    path=upstream/"05-attention-pointer-doubling"/"pdg.py"
    add_seed_flag(path)
    shutil.copyfile(Path(__file__).with_name("pointer_accel.py"),
                    path.with_name("pointer_accel.py"))
    shutil.copyfile(Path(__file__).with_name("pointer_hash.py"),
                    path.with_name("pointer_hash.py"))
    patch_pdg(path)
    cases=[]
    for cname,hops,order,fill,dup in CONFIGS:
        for seed in a.seeds:
            base=["--task","hopx","--hops",hops,"--pairs",8,
                "--layers",3,"--width",64,"--heads",4,"--L",0,
                "--batch",16,"--steps",a.steps,"--ee",25,"--stop",0,
                "--nev",256,"--init",0.5,"--rev",order,
                "--fill",fill,"--dup",dup,"--seed",seed,
                "--dev",a.device,"--threads",1,
                "--sink",0,"--alibi",0,"--eqh",2,"--eqt",3,
                "--ptr",11,"--pnb",3,"--lr",0.02]
            arms=[("dense",["--wuke_sparse_mode","none"])]
            arms += [(f"block{bs}",["--wuke_sparse_mode","block",
                         "--wuke_bs",a.block_size,"--wuke_blocks",bs])
                     for bs in a.blocks]
            arms.append(("hash",["--wuke_sparse_mode","hash",
                                  "--wuke_bs",a.block_size,"--wuke_blocks",2]))
            for name,opts in arms:
                cases.append(dict(name=f"ptrsp_{cname}_s{seed}_{name}",
                    path="05-attention-pointer-doubling/pdg.py",
                    args=base+opts,results_file="results.jsonl",
                    timeout=900,data_mode="generated-final-answer-only"))
    report=run_cases(upstream,cases,a.budget_seconds,
                     work/"reports",default_timeout=900)
    comparisons=[]
    for cname,*_ in CONFIGS:
        for seed in a.seeds:
            def get(which):
                z=report["runs"].get(f"ptrsp_{cname}_s{seed}_{which}",{})
                return z.get("record") if z.get("status")=="ok" else None
            dense=get("dense")
            sparse={name:get(f"block{name}") for name in a.blocks}
            sparse["hash"]=get("hash")
            row=dict(config=cname,seed=seed,
               dense_acc=None if dense is None else float(dense["acc"]),
               sparse_acc={str(k):(None if r is None else float(r["acc"]))
                          for k,r in sparse.items()},
               pointer_pairs={str(k):(None if r is None else
                         r.get("wuke_pointer_stats")) for k,r in sparse.items()},
               train_s={"dense":None if dense is None else dense["train_s"],
                        **{str(k):None if r is None else r["train_s"]
                           for k,r in sparse.items()}})
            comparisons.append(row)
    complete=all(r["dense_acc"] is not None and
                 all(v is not None for v in r["sparse_acc"].values())
                 for r in comparisons)
    screen="INCOMPLETE"
    if complete:
        rev=[r for r in comparisons if r["config"]=="reverse7"]
        supported=all(r["dense_acc"]>=.9 and
            any(v>=r["dense_acc"]-.03 for v in r["sparse_acc"].values()) for r in rev)
        screen="SPARSE_POINTER_ACCURACY_FEASIBLE" if supported else "NO_CLEAR_FEASIBILITY"
    summary=dict(screen=screen,comparisons=comparisons,
       warning="Ordinary Transformer attention remains dense. No GPU speedup claim.")
    (work/"reports"/"sparse_model_decision.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    if not complete:raise SystemExit("Incomplete paired coauthor model screen")

if __name__=="__main__":
    main()
