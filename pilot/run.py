#!/usr/bin/env python3
"""<=3h counterexample-first Muon pilot: adaptive stop vs 30%-hold and noise guard.
Clones and patches the MIT reference only inside an ignored local workspace.
Run python3 pilot/run.py --device cpu or --device cuda.
"""
import argparse
import json
import math
from pathlib import Path
from core import ensure_upstream, add_seed_flag, prepare_token_data, run_cases

def patch_confidence_hold(path):
    s=path.read_text()
    if 'p.add_argument("--dh_patience"' in s:return
    old='a = p.parse_args()'
    assert s.count(old)==1
    s=s.replace(old,'p.add_argument("--dh_patience", type=int, default=0)\n'+old,1)
    old='curve, slog, loss_ema, diverged = [], [], None, False'
    assert s.count(old)==1
    s=s.replace(old,old+'\n    dh_counter = 0',1)
    old='mu.adapt = a.adapt_start <= step < aend and (a.freeze_at < 0 or step < a.freeze_at)'
    assert s.count(old)==1
    new='''# Causal confidence-based hold: only use cosine diagnostics from prior steps.
        # This is an experimental extension; a fixed 30% hold remains the control.
        if a.dh_patience and step >= max(60, int(a.steps * 0.15)) and step < aend:
            signal = abs(float(getattr(mu, "cm", 0.0)))
            noise = math.sqrt(max(float(getattr(mu, "cv", 0.0)), 1e-10))
            dh_counter = dh_counter + 1 if signal < 0.6 * noise else 0
            if dh_counter >= a.dh_patience:
                aend = step
        '''+old
    s=s.replace(old,new,1)
    path.write_text(s)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    p.add_argument("--budget-seconds",type=int,default=10500)
    p.add_argument("--work-dir",default=".pilot_workspace")
    p.add_argument("--real-data-dir",default=None)
    p.add_argument("--seeds",type=int,nargs="+",default=[0,1])
    a=p.parse_args()
    assert 0<a.budget_seconds<=10600
    if a.device=="cuda":
        import torch
        assert torch.cuda.is_available()
    root=Path(__file__).resolve().parent.parent
    work=root/a.work_dir
    upstream=ensure_upstream(work)
    src=upstream/"02-muon-hypergradient-learning-rate"/"ss.py"
    add_seed_flag(src)
    patch_confidence_hold(src)
    configs=[
      dict(name="short225",steps=225,width=64,batch=16,layers=4),
      dict(name="high900",steps=900,width=128,batch=4,layers=4),
    ]
    need=max(x["steps"]*x["batch"]*64 for x in configs)
    manifest=prepare_token_data(upstream,need,vocab=3000,real_dir=a.real_data_dir)
    outfile="feasibility_pilot.jsonl"
    cases=[]
    for cfg in configs:
        gain=math.sqrt(cfg["width"]/32)
        freeze=min(int(cfg["steps"]*.3),450)
        common=["--width",cfg["width"],"--layers",cfg["layers"],"--heads",4,
          "--V",1000,"--batch",cfg["batch"],"--L",64,"--steps",cfg["steps"],
          "--dev",a.device,"--elr",0.04,"--wd",1,"--warm",0,"--mwarm",0,
          "--cool",0.3,"--out",outfile,"--threads",1]
        for seed in a.seeds:
            arms=[
                ("fixed_lr_low",["--rule","none","--lr",0.00125]),
                ("fixed_lr_mid",["--rule","none","--lr",0.005]),
                ("fixed_lr_high",["--rule","none","--lr",0.02]),
                ("hold",["--rule","mult","--sig","hyper","--gran","glob",
                          "--eta",gain,"--lr",0.05,"--freeze_at",freeze]),
                ("post_noise",["--rule","mult","--sig","hyper","--gran","glob",
                          "--eta",gain,"--lr",0.05,"--freeze_at",freeze,
                          "--post","down","--psd",2]),
                ("confidence",["--rule","mult","--sig","hyper","--gran","glob",
                          "--eta",gain,"--lr",0.05,"--freeze_at",-1,
                          "--dh_patience",35]),
            ]
            for arm,params in arms:
                name=f"{cfg['name']}_s{seed}_{arm}"
                cases.append(dict(name=name,path="02-muon-hypergradient-learning-rate/ss.py",
                    args=common+["--seed",seed]+params,results_file=outfile,
                    timeout=1600,data_mode=manifest["mode"]))
    report=run_cases(upstream,cases,a.budget_seconds,work/"reports",default_timeout=1600)
    summary={"data":manifest,"comparisons":[],"screen":"INCONCLUSIVE"}
    for cfg in configs:
        for seed in a.seeds:
            def find(arm):
                r=report["runs"].get(f"{cfg['name']}_s{seed}_{arm}",{}).get("record")
                return None if not r else float(r["scores"]["val"])
            hold,post,conf=find("hold"),find("post_noise"),find("confidence")
            const=[find(k) for k in ("fixed_lr_low","fixed_lr_mid","fixed_lr_high")]
            if None not in (hold,post,conf,*const):
                summary["comparisons"].append(dict(config=cfg["name"],seed=seed,
                    best_of_three_fixed=min(const),hold=hold,post_noise=post,
                    confidence=conf,conf_minus_hold=conf-hold,
                    conf_minus_best_fixed=conf-min(const)))
    hard=[x for x in summary["comparisons"] if x["config"]=="high900"]
    if len(hard)==len(a.seeds):
        summary["screen"]="GO_REAL_DATA" if all(x["conf_minus_hold"]<=-0.005 and x["conf_minus_best_fixed"]<=0.006 for x in hard) else "NO_GO_DYNAMIC_HOLD"
    (work/"reports"/"decision.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    if not summary["comparisons"]:raise SystemExit("No complete paired result")
if __name__=="__main__":main()
