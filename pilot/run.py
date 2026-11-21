"""Run bounded, paired-seed SG-Delta study. Primary: query-joint.
See README for what the program does NOT prove.
"""
import argparse,json,time
from pathlib import Path
import torch
from common import MemoryLM,PRESETS,train_case
from sg_delta import SGDeltaMixer,ARMS

DEFAULT=["isotropic","query-write","query-decay","query-joint",
         "key-joint","channel-decay"]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--profile",choices=PRESETS,default="pilot")
    p.add_argument("--device",choices=["cpu","cuda","mps"],default="cpu")
    p.add_argument("--methods",nargs="+",default=DEFAULT,choices=ARMS)
    p.add_argument("--seeds",nargs="+",type=int,default=[0,1])
    p.add_argument("--steps",type=int,default=None)
    p.add_argument("--budget-seconds",type=int,default=10200)
    p.add_argument("--output",default="runs/sgdelta_joint")
    p.add_argument("--dry-run",action="store_true")
    args=p.parse_args()
    if args.device=="cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")
    if not 1<=args.budget_seconds<=10800:raise SystemExit("invalid time budget")
    cfg=PRESETS[args.profile].copy()
    cfg["steps"]=args.steps or cfg["steps"]
    if args.profile in ("100m-sizing","340m-sizing") and not args.dry_run:
        raise SystemExit("large profiles are sizing-only: full matrix PyTorch recurrence is not a large-model fused implementation")
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();cases=[(seed,arm) for seed in args.seeds for arm in args.methods]
    results={};nparams=None
    for index,(seed,arm) in enumerate(cases):
        torch.manual_seed(seed)
        with torch.device("meta" if args.dry_run else "cpu"):
            model=MemoryLM(cfg.get("vocab_size",4+2*cfg["key_space"]),
                cfg["dim"],cfg["layers"],
                lambda d,i:SGDeltaMixer(d,heads=cfg["heads"],key_dim=cfg["head_dim"],
                     chunk=cfg["chunk"],mode=arm,detach_statistics=True))
        nparams=sum(z.numel() for z in model.parameters())
        if args.dry_run:
            print(f"{arm} seed={seed}: {nparams:,} parameters");continue
        remaining=args.budget_seconds-(time.monotonic()-start)
        if remaining<35:break
        case_seconds=max(25.,.93*remaining/(len(cases)-index))
        name=f"{arm}_seed{seed}"
        print("RUN",name,"params",nparams,"budget_s",round(case_seconds,1),flush=True)
        try:
            row=train_case(model,cfg,device=args.device,seed=seed,steps=cfg["steps"],
                budget_seconds=case_seconds,out_path=out/f"{name}.json")
            results[name]={"status":row["status"],"eval":row["eval"],
                           "elapsed_seconds":row["elapsed_seconds"],
                           "peak_vram_gb":row["peak_allocated_vram_gb"]}
        except Exception as exc:
            results[name]={"status":"error","error":repr(exc)}
            (out/f"{name}.error.txt").write_text(repr(exc))
            print("ERROR",name,repr(exc),flush=True)
        (out/"manifest.json").write_text(json.dumps(dict(
            profile=args.profile,actual_params=nparams,config=cfg,
            data="generated causal MQAR, not real text",
            requested_methods=args.methods,requested_seeds=args.seeds,runs=results),indent=2))
    if args.dry_run:return
    matched=[]
    for seed in args.seeds:
        arms={k:results.get(f"{k}_seed{seed}") for k in DEFAULT[:5]}
        if not all(x is not None and x["status"]=="ok" for x in arms.values()):
            continue
        def read(arm,key):return arms[arm]["eval"][1].get(key)
        matched.append(dict(seed=seed,
            old_post_query_joint=read("query-joint","old_post_acc"),
            old_post_key_joint=read("key-joint","old_post_acc"),
            old_post_query_write=read("query-write","old_post_acc"),
            old_post_query_decay=read("query-decay","old_post_acc"),
            old_post_isotropic=read("isotropic","old_post_acc"),
            alias_query_joint=read("query-joint","alias_acc"),
            alias_key_joint=read("key-joint","alias_acc")))
    decision="INCONCLUSIVE"
    if len(matched)==len(args.seeds) and matched:
        if all(r["old_post_query_joint"]>=r["old_post_key_joint"]+.03 and
               r["old_post_query_joint"]>=max(r["old_post_query_write"],
                                               r["old_post_query_decay"])+.02 and
               r["alias_query_joint"]>=r["alias_key_joint"]-.03
               for r in matched):
            decision="GO_TO_REPLICATION"
        else:decision="NO_CLEAR_ADVANTAGE"
    report=dict(decision=decision,matched_seeds=matched,
        actual_params=nparams,elapsed_seconds=time.monotonic()-start,
        limitation="small synthetic mechanism study; not paper-grade GDN2 or text LM result")
    (out/"decision.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
