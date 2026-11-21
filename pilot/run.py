#!/usr/bin/env python3
"""<=3h pilot: function-preserving output regrouping vs strong Adam-eps baseline.
Run: python3 pilot/run.py --device cuda --budget-seconds 10500
Proxy data is synthetic, never FineWeb. Real data requires --real-data-dir.
"""
import argparse
import json
from pathlib import Path
from core import ensure_upstream, add_seed_flag, prepare_token_data, run_cases

def patch_gradient_regroup(path):
    s=path.read_text()
    if 'p.add_argument("--group_on"' in s:return
    k='a = p.parse_args()'
    assert s.count(k)==1
    s=s.replace(k,'p.add_argument("--group_on", choices=["weights", "grads"], default="weights")\n'+k,1)
    anchor='s.head = nn.Parameter(torch.randn(V, d) * d ** -0.5 * a.hinit)'
    assert s.count(anchor)==1
    s=s.replace(anchor,anchor+'\n        s.register_buffer("grad_ema", torch.zeros(V, d))',1)
    old='asg = kmeans(R, K)'
    assert s.count(old)==1
    s=s.replace(old,'asg = kmeans(s.grad_ema if a.group_on == "grads" else R, K)',1)
    old='for o in opts: o.step()'
    assert s.count(old)==1
    s=s.replace(old,'if a.clus and a.group_on == "grads":\n            with torch.no_grad():\n                m.grad_ema.mul_(0.95).add_(m.head.grad.detach(), alpha=0.05)\n        '+old,1)
    old='s.Clast = cm.clone()'
    assert s.count(old)==1
    s=s.replace(old,old+'\n        assert torch.allclose(s.H(), R, atol=1e-4, rtol=1e-5), "regroup must preserve effective head"',1)
    path.write_text(s)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--device",choices=["cpu","cuda","mps"],default="cuda")
    p.add_argument("--budget-seconds",type=int,default=10500)
    p.add_argument("--work-dir",default=".pilot_workspace_rare_v2")
    p.add_argument("--real-data-dir",default=None)
    p.add_argument("--steps",type=int,default=400)
    p.add_argument("--seeds",type=int,nargs="+",default=[0,1])
    a=p.parse_args()
    assert 0<a.budget_seconds<=10600
    root=Path(__file__).resolve().parent.parent
    work=root/a.work_dir
    upstream=ensure_upstream(work)
    script=upstream/"01-rare-words-shared-output-vectors"/"grid.py"
    add_seed_flag(script)
    patch_gradient_regroup(script)
    if a.device=="cuda":
        import torch
        assert torch.cuda.is_available(),"cuda requested but unavailable; use --device cpu"
    manifest=prepare_token_data(upstream,a.steps*32*64,vocab=8000,real_dir=a.real_data_dir)
    out="feasibility_pilot.jsonl"
    variants=[
        ("adam_eps",["--clus",0,"--heps","1e-5"]),
        ("fixed_random",["--clus",64,"--cinit","rand","--heps","1e-5"]),
        ("fixed_frequency",["--clus",64,"--cinit","freq","--heps","1e-5"]),
        ("learned_weights",["--clus",64,"--cinit","kmeans","--group_on","weights","--heps","1e-5"]),
        ("learned_gradients",["--clus",64,"--cinit","kmeans","--group_on","grads","--heps","1e-5"]),
    ]
    cases=[]
    for seed in a.seeds:
        for variant,flags in variants:
            name=f"rare_s{seed}_{variant}"
            cases.append(dict(name=name,path="01-rare-words-shared-output-vectors/grid.py",
                args=["--V",6000,"--width",32,"--layers",2,"--heads",2,
                      "--L",64,"--batch",32,"--steps",a.steps,"--dev",a.device,
                      "--seed",seed,"--recl",50,"--out",out,"--nval",600]+flags,
                results_file=out,timeout=1100,data_mode=manifest["mode"]))
    report=run_cases(upstream,cases,a.budget_seconds,work/"reports",default_timeout=1100)
    def nll_without_unk(rec):
        """Weighted held-out NLL without the mapped unknown-token artefact."""
        overall=float(rec["scores"]["val"])
        unk=rec.get("bins",{}).get("unk")
        if unk is None:
            raise ValueError("upstream did not emit the required UNK bin")
        unk_loss,unk_share=map(float,unk)
        if not 0<=unk_share<1:
            raise ValueError("invalid UNK target mass")
        return (overall-unk_share*unk_loss)/(1-unk_share)

    summary={"data":manifest,"budget_s":a.budget_seconds,
             "screen":"INCONCLUSIVE","pairs":[],
             "method_hypothesis":"Gradient-history regrouping must beat row-weight regrouping"}
    for seed in a.seeds:
        def fetch(v):
            r=report["runs"].get(f"rare_s{seed}_{v}",{})
            return r.get("record") if r.get("status")=="ok" else None
        E,W,G,R,F=(fetch(v) for v in ("adam_eps","learned_weights",
                 "learned_gradients","fixed_random","fixed_frequency"))
        if all(x is not None for x in (E,W,G,R,F)):
            n={name:nll_without_unk(record)
               for name,record in (("adam_eps",E),("learned_weights",W),
                   ("learned_gradients",G),("fixed_random",R),("fixed_frequency",F))}
            def rare_10_31(rec):
                z=rec.get("bins",{}).get("10-31")
                return None if z is None else {"nll":float(z[0]),"mass":float(z[1])}
            summary["pairs"].append(dict(
              seed=seed,
              nll_all={name:float(rec["scores"]["val"]) for name,rec in (
                ("adam_eps",E),("learned_weights",W),("learned_gradients",G),
                ("fixed_random",R),("fixed_frequency",F))},
              nll_no_unk=n,
              gain_weight_vs_eps=n["adam_eps"]-n["learned_weights"],
              gain_grad_vs_weight=n["learned_weights"]-n["learned_gradients"],
              gain_grad_vs_fixed=max(n["fixed_random"],n["fixed_frequency"])-n["learned_gradients"],
              rare_10_31={name:rare_10_31(rec) for name,rec in (
                ("adam_eps",E),("learned_weights",W),("learned_gradients",G))},
              rare_target_mass=W["rare_mass_lt100"],
              unknown_target_mass=float(W["bins"]["unk"][1])))
    if len(summary["pairs"])==len(a.seeds):
        support=all(x["gain_weight_vs_eps"]>=.01 and
                    x["rare_target_mass"]>=.05 for x in summary["pairs"])
        new_support=all(x["gain_grad_vs_weight"]>=.005 and
                        x["nll_no_unk"]["learned_gradients"]<
                            min(x["nll_no_unk"]["fixed_random"],
                                x["nll_no_unk"]["fixed_frequency"])
                        for x in summary["pairs"])
        summary["weight_regroup_support"]=support
        summary["gradient_extension_support"]=new_support
        if manifest["mode"]=="synthetic-topic-zipf":
            summary["screen"]="INCONCLUSIVE_PROXY_ONLY"
        elif support and new_support:
            summary["screen"]="GO_REAL_REPLICATION_NEW_GRADIENT_EXTENSION"
        elif support:
            summary["screen"]="BASELINE_REPLICATES_GRADIENT_EXTENSION_NOT_SUPPORTED"
        else:
            summary["screen"]="NO_CLEAR_REAL_DATA_ADVANTAGE"
    (work/"reports"/"decision.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    if not summary["pairs"]:
        raise SystemExit("No completed paired comparison; inspect per-case logs")
if __name__=="__main__":
    main()
