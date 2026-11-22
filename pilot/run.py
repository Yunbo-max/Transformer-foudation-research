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
    """Apply only to the pinned upstream grid.py, in an ignored workspace."""
    s=path.read_text()
    marker="# WUKE_POSITIVE_TARGET_GRADIENT_GROUPING_V2"
    if marker in s:
        return
    if 'p.add_argument("--group_on"' in s:
        raise RuntimeError("Old gradient pilot patch in checkout. Use a fresh --work-dir.")

    def replace_once(old,new):
        nonlocal s
        if s.count(old)!=1:
            raise RuntimeError(f"pinned grid.py changed: expected unique anchor {old[:75]!r}")
        s=s.replace(old,new,1)

    replace_once("import numpy as np, torch, torch.nn as nn, torch.nn.functional as F",
        "import numpy as np, torch, torch.nn as nn, torch.nn.functional as F\n"
        "from grad_grouping import accumulate_positive_target_gradients, assign_by_positive_gradient\n"
        +marker)
    replace_once("a = p.parse_args()",
        'p.add_argument("--group_on", choices=["weights","grads","positive_direction","positive_dir_mag"], default="weights")\n'
        'p.add_argument("--positive_min_count",type=int,default=3)\n'
        'p.add_argument("--gradient_magnitude_weight",type=float,default=0.35)\n'
        'a = p.parse_args()\n'
        'if a.group_on.startswith("positive_") and (a.asm or a.cls):\n'
        '    raise ValueError("positive-target gradient tracking requires ordinary cross-entropy")')
    replace_once("s.head = nn.Parameter(torch.randn(V, d) * d ** -0.5 * a.hinit)",
        's.head = nn.Parameter(torch.randn(V, d) * d ** -0.5 * a.hinit)\n'
        '        s.register_buffer("grad_ema",torch.zeros(V,d))\n'
        '        s.register_buffer("positive_grad_sum",torch.zeros(V,d))\n'
        '        s.register_buffer("positive_grad_count",torch.zeros(V))\n'
        '        s.grouping_stats={}')
    replace_once("        asg = kmeans(R, K)\n",
        '        if a.group_on in ("positive_direction","positive_dir_mag"):\n'
        '            asg,s.grouping_stats=assign_by_positive_gradient(\n'
        '                R,s.positive_grad_sum,s.positive_grad_count,K,kmeans_fn=kmeans,\n'
        '                min_count=a.positive_min_count,\n'
        '                magnitude_weight=a.gradient_magnitude_weight if a.group_on=="positive_dir_mag" else 0.)\n'
        '        else:\n'
        '            asg=kmeans(s.grad_ema if a.group_on=="grads" else R,K)\n'
        '            s.grouping_stats={"mode":a.group_on}\n')
    replace_once("        return F.cross_entropy(lg, tg, reduction=\"none\")",
        '        if s.training and a.clus and a.group_on in ("positive_direction","positive_dir_mag"):\n'
        '            accumulate_positive_target_gradients(\n'
        '                s.positive_grad_sum,s.positive_grad_count,tg,h,lg)\n'
        '        return F.cross_entropy(lg, tg, reduction="none")')
    replace_once("        for o in opts: o.step()",
        '        if a.clus and a.group_on == "grads":\n'
        '            with torch.no_grad():\n'
        '                m.grad_ema.mul_(0.95).add_(m.head.grad.detach(),alpha=0.05)\n'
        '        for o in opts: o.step()')
    replace_once("s.Clast = cm.clone()",
        's.Clast = cm.clone()\n'
        '        assert torch.allclose(s.H(),R,atol=1e-4,rtol=1e-5), "regroup changed effective head"')
    replace_once("    if a.save:\n",
        '    if a.clus and hasattr(m,"grouping_stats"):\n'
        '        res["group_diagnostics"]=m.grouping_stats\n'
        '        res["positive_target_observations"]=int(m.positive_grad_count.sum().item())\n'
        '    if a.save:\n')
    path.write_text(s)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--device",choices=["cpu","cuda","mps"],default="cuda")
    p.add_argument("--budget-seconds",type=int,default=10500)
    p.add_argument("--work-dir",default=".pilot_workspace_rare_v3")
    p.add_argument("--real-data-dir",default=None)
    p.add_argument("--steps",type=int,default=400)
    p.add_argument("--seeds",type=int,nargs="+",default=[0,1])
    p.add_argument("--positive-min-count",type=int,default=3)
    p.add_argument("--magnitude-weight",type=float,default=0.35)
    a=p.parse_args()
    assert 0<a.budget_seconds<=10600
    if a.positive_min_count<1 or not 0<=a.magnitude_weight<=3.:
        raise ValueError("invalid positive gradient clustering settings")
    root=Path(__file__).resolve().parent.parent
    work=root/a.work_dir
    upstream=ensure_upstream(work)
    script=upstream/"01-rare-words-shared-output-vectors"/"grid.py"
    add_seed_flag(script)
    # Vendored helper is OUR code; upstream remains pinned and unchanged in Git.
    import shutil
    shutil.copyfile(Path(__file__).with_name("grad_grouping.py"),script.with_name("grad_grouping.py"))
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
        ("positive_direction",["--clus",64,"--cinit","kmeans","--group_on","positive_direction",
                               "--positive_min_count",a.positive_min_count,"--heps","1e-5"]),
        ("positive_dir_mag",["--clus",64,"--cinit","kmeans","--group_on","positive_dir_mag",
                             "--positive_min_count",a.positive_min_count,
                             "--gradient_magnitude_weight",a.magnitude_weight,
                             "--heps","1e-5"]),
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
            return overall  # no UNK targets: excluding UNK equals ordinary NLL
        unk_loss,unk_share=map(float,unk)
        if not 0<=unk_share<1:
            raise ValueError("invalid UNK target mass")
        return (overall-unk_share*unk_loss)/(1-unk_share)

    variants_dict=dict(variants)
    summary={"data":manifest,"budget_s":a.budget_seconds,
             "screen":"INCONCLUSIVE","pairs":[],
             "method_hypothesis":"Positive-target gradient direction/magnitude vs existing weight k-means",
             "variants":list(variants_dict),
             "positive_min_count":a.positive_min_count,
             "magnitude_weight":a.magnitude_weight,
             "warning":"Pilot screen only. Complete 3+ real-data seeds and tune matched baselines."}
    for seed in a.seeds:
        def fetch(v):
            run=report["runs"].get(f"rare_s{seed}_{v}",{})
            return run.get("record") if run.get("status")=="ok" else None
        records={v:fetch(v) for v in variants_dict}
        if not all(rec is not None for rec in records.values()):
            summary.setdefault("incomplete_seeds",[]).append(
                dict(seed=seed,missing=[v for v,r in records.items() if r is None]))
            continue
        n={name:nll_without_unk(record) for name,record in records.items()}
        def bin_score(rec,name):
            z=rec.get("bins",{}).get(name)
            return None if z is None else {"nll":float(z[0]),"mass":float(z[1])}
        summary["pairs"].append(dict(
            seed=seed,
            nll_all={name:float(rec["scores"]["val"]) for name,rec in records.items()},
            nll_no_unk=n,
            gain_weight_vs_eps=n["adam_eps"]-n["learned_weights"],
            gain_full_ema_vs_weight=n["learned_weights"]-n["learned_gradients"],
            gain_direction_vs_weight=n["learned_weights"]-n["positive_direction"],
            gain_dir_mag_vs_weight=n["learned_weights"]-n["positive_dir_mag"],
            gain_magnitude_vs_direction=n["positive_direction"]-n["positive_dir_mag"],
            bins={band:{name:bin_score(rec,band) for name,rec in records.items()}
                  for band in ("1-9","10-31","32-99","100-315","unk")},
            group_diagnostics={
                name:rec.get("group_diagnostics",{})
                for name,rec in records.items() if name!="adam_eps"},
            rare_target_mass=float(records["learned_weights"]["rare_mass_lt100"]),
            unknown_target_mass=float(records["learned_weights"]["bins"].get("unk",[0.,0.])[1])))
    if len(summary["pairs"])==len(a.seeds):
        summary["weight_regroup_support"]=all(
            x["gain_weight_vs_eps"]>=.01 and x["rare_target_mass"]>=.05
            for x in summary["pairs"])
        summary["direction_support"]=all(
            x["gain_direction_vs_weight"]>=.005 and
            x["nll_no_unk"]["positive_direction"]<
                min(x["nll_no_unk"]["fixed_random"],
                    x["nll_no_unk"]["fixed_frequency"])
            for x in summary["pairs"])
        summary["direction_plus_magnitude_support"]=all(
            x["gain_dir_mag_vs_weight"]>=.005 and
            x["nll_no_unk"]["positive_dir_mag"]<
                min(x["nll_no_unk"]["fixed_random"],
                    x["nll_no_unk"]["fixed_frequency"])
            for x in summary["pairs"])
        summary["magnitude_incremental_support"]=all(
            x["gain_magnitude_vs_direction"]>=.003 for x in summary["pairs"])
        if manifest["mode"]=="synthetic-topic-zipf":
            summary["screen"]="INCONCLUSIVE_PROXY_ONLY"
        elif summary["weight_regroup_support"] and (
                 summary["direction_support"] or summary["direction_plus_magnitude_support"]):
            summary["screen"]="GO_TO_3PLUS_SEED_REAL_DATA_REPLICATION"
        elif summary["weight_regroup_support"]:
            summary["screen"]="OLD_BASELINE_SUPPORTED_NO_NEW_GRADIENT_GAIN"
        else:
            summary["screen"]="NO_CLEAR_REAL_DATA_ADVANTAGE"
    (work/"reports"/"decision.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    if not summary["pairs"]:
        raise SystemExit("No completed paired comparison; inspect per-case logs")
if __name__=="__main__":
    main()
