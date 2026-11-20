#!/usr/bin/env python3
"""<=3h paired pointer paper pilot: one-hop vs doubling+lifting, by fact order.
Uses the pinned MIT reference. Synthetic final-answer supervised recall ONLY;
no ordinary LM or natural language gain is claimed.
"""
import argparse
import json
from pathlib import Path
from core import ensure_upstream, add_seed_flag, run_cases

ARMS={
 "F":["--sink",1,"--alibi",1,"--lr",0.04],
 "PNB":["--sink",1,"--alibi",1,"--eqh",2,"--eqt",3,"--eqna",1,"--ptr",3,"--pnb",3,"--lr",0.02],
 "BL":["--sink",0,"--alibi",0,"--eqh",2,"--eqt",3,"--ptr",11,"--lr",0.02],
 "BLN":["--sink",0,"--alibi",0,"--eqh",2,"--eqt",3,"--ptr",11,"--pnb",3,"--lr",0.02],
 "PRD":["--sink",0,"--alibi",0,"--eqh",2,"--eqt",3,"--ptr",6,"--lr",0.02],
}
CONFIGS=[
 ("reverse_7",7,3,1,0,0.0,["F","PNB","BL","BLN"]),
 ("forward_4",4,3,0,0,0.0,["F","BLN","PRD"]),
 ("random_7",7,3,2,0,0.0,["F","PNB","BLN"]),
 ("reverse_5",5,3,1,0,0.0,["BL","BLN"]),
 ("repeat_7",7,3,1,2,0.30,["PNB","BLN"]),
]

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--device",choices=["cpu","cuda","mps"],default="cpu")
 p.add_argument("--budget-seconds",type=int,default=10500)
 p.add_argument("--work-dir",default=".pilot_workspace_pointer")
 p.add_argument("--seeds",type=int,nargs="+",default=[0,1])
 p.add_argument("--steps",type=int,default=75)
 a=p.parse_args()
 assert 0<a.budget_seconds<=10600 and 0<a.steps<=200
 if a.device=="cuda":
    import torch
    assert torch.cuda.is_available()
 root=Path(__file__).resolve().parent.parent
 work=root/a.work_dir
 upstream=ensure_upstream(work)
 code=upstream/"05-attention-pointer-doubling"/"pdg.py"
 add_seed_flag(code)
 cases=[]
 for cname,hops,layers,order,fill,dup,arms in CONFIGS:
  for seed in a.seeds:
   for arm in arms:
    name=f"pointer_{cname}_s{seed}_{arm}"
    args=["--task","hopx","--hops",hops,"--pairs",8,"--layers",layers,
      "--width",64,"--heads",4,"--L",0,"--batch",16,
      "--steps",a.steps,"--ee",25,"--stop",0,"--nev",256,"--init",0.5,
      "--rev",order,"--fill",fill,"--dup",dup,"--seed",seed,"--dev",a.device,
      "--threads",1]+ARMS[arm]
    cases.append(dict(name=name,path="05-attention-pointer-doubling/pdg.py",
       args=args,timeout=850,data_mode="generated-final-answer-only"))
 report=run_cases(upstream,cases,a.budget_seconds,work/"reports",default_timeout=850)
 summary={"screen":"INCONCLUSIVE","pair_results":[],"budget_s":a.budget_seconds}
 for seed in a.seeds:
  def get(config,arm):
   r=report["runs"].get(f"pointer_{config}_s{seed}_{arm}",{}).get("record")
   return None if r is None else float(r["acc"])
  row=dict(seed=seed,reverse7_full=get("reverse_7","BLN"),
           reverse7_plain=get("reverse_7","F"),
           reverse7_onehop=get("reverse_7","PNB"),
           reverse7_no_backtrack=get("reverse_7","BL"),
           forward4_predecessor=get("forward_4","PRD"),
           forward4_full=get("forward_4","BLN"),
           random7_full=get("random_7","BLN"),
           reverse5_full=get("reverse_5","BLN"),
           reverse5_no_backtrack=get("reverse_5","BL"),
           repeated7_full=get("repeat_7","BLN"))
  summary["pair_results"].append(row)
 tested=[r for r in summary["pair_results"] if all(r[k] is not None for k in
    ("reverse7_full","reverse7_plain","reverse7_onehop","forward4_predecessor","random7_full","reverse5_full"))]
 if len(tested)==len(a.seeds):
  mechanism=all(r["reverse7_full"]>=.90 and r["reverse7_full"]-max(r["reverse7_onehop"],r["reverse7_plain"])>=.40 for r in tested)
  general=all(r["random7_full"]>=.80 and r["forward4_predecessor"]>=.80 for r in tested)
  if mechanism and general:summary["screen"]="ADVANCE_ORDER_GENERALIZATION"
  elif mechanism:summary["screen"]="MECHANISM_ONLY_NO_GENERALIZATION"
  else:summary["screen"]="NO_GO_OR_REDESIGN"
 (work/"reports"/"decision.json").write_text(json.dumps(summary,indent=2))
 print(json.dumps(summary,indent=2))
 if not tested:raise SystemExit("Missing paired results")
if __name__=="__main__":main()
