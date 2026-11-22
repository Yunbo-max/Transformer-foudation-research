"""Fail-closed adapter for coauthor's ptr=11 internal double-pointer reads.

Does NOT sparsify the ordinary Transformer attention, which remains dense.
This adapter is an optional accuracy feasibility experiment, not a fused
end-to-end acceleration claim. It preserves original mode unmodified.
"""
from pathlib import Path
import textwrap

MARKER="# WUKE_POINTER_SPARSE_ADAPTER_V1"


def patch_pdg(path):
    path=Path(path)
    source=path.read_text()
    if MARKER in source:return
    imp="import numpy as np, torch, torch.nn as nn, torch.nn.functional as F"
    if source.count("a = p.parse_args()")!=1 or source.count(imp)!=1:
        raise RuntimeError("pinned pdg.py imports or parser changed")
    source=source.replace(imp,imp+"\nfrom pointer_accel import pointer_lookup\n"+MARKER,1)
    source=source.replace("a = p.parse_args()",
        'p.add_argument("--wuke_sparse_mode", choices=["none","block"],default="none")\n'
        'p.add_argument("--wuke_bs",type=int,default=16)\n'
        'p.add_argument("--wuke_blocks",type=int,default=2)\n'
        'p.add_argument("--wuke_chunk",type=int,default=32)\n'
        'a = p.parse_args()',1)
    start="                if a.ptr in (1, 3, 4, 5, 9, 10, 11):"
    stop="                if a.ptr in (6, 7, 8):"
    if source.count(start)!=1 or source.count(stop)!=1:
        raise RuntimeError("pinned pdg.py pointer branch changed")
    i=source.index(start);j=source.index(stop,i)
    original=source[i:j]
    if ("A0 = torch.softmax(lg0, -1)" not in original or
        "A1_ = torch.softmax(lg1, -1)" not in original):
        raise RuntimeError("original pointer equations changed")
    sparse='''                if a.ptr == 11 and a.wuke_sparse_mode == "block" and not s.pfreeze:
                    if getattr(s,"rec",False):
                        raise RuntimeError("sparse pprobe needs separate diagnostics; disable --pprobe")
                    old_e = F.pad(e0, (0,0,2,0))[:, :-2] if a.pnb else None
                    combo = torch.cat([ptr,rr],dim=-1) if rr is not None else ptr
                    jumped,diag0 = pointer_lookup(qe[:,0],ke[:,0],combo,
                        mode="block",block_size=a.wuke_bs,blocks=a.wuke_blocks,
                        chunk=a.wuke_chunk,
                        history=rr if (rr is not None and a.pnb) else None,
                        historical_keys=old_e if (rr is not None and a.pnb) else None,
                        penalty=a.pnb)
                    if rr is None:
                        next_ptr,next_rr=jumped,ptr
                    else:
                        next_ptr,next_rr=jumped.split(d,dim=-1)
                    lifted,diag1=pointer_lookup(qe[:,1],ke[:,1],combo,
                        mode="block",block_size=a.wuke_bs,blocks=a.wuke_blocks,
                        chunk=a.wuke_chunk,
                        history=rq if a.pnb else None,
                        historical_keys=old_e if a.pnb else None,
                        penalty=a.pnb)
                    gate=torch.sigmoid(s.S["pg"][i])
                    rq=gate*(lifted[...,d:] if rr is not None else qq)+(1-gate)*rq
                    qq=gate*lifted[...,:d]+(1-gate)*qq
                    pprev=ptr;rr=next_rr;ptr=next_ptr
                    if not hasattr(s,"wuke_stats"):
                        s.wuke_stats={"pointer_scoring_pairs":0,"pointer_router_pairs":0}
                    for z in (diag0,diag1):
                        s.wuke_stats["pointer_scoring_pairs"]+=z["scoring_pairs"]
                        s.wuke_stats["pointer_router_pairs"]+=z["router_pairs"]
                elif a.ptr in (1, 3, 4, 5, 9, 10, 11):
'''
    # Original branch remains available with a.wuke_sparse_mode==none.
    old_body=original[len(start):].lstrip("\n")
    source=source[:i]+sparse+textwrap.indent(old_body,"    ")+source[j:]
    if 'elif a.ptr in (1, 3, 4, 5, 9, 10, 11):' not in source:
        raise AssertionError("original fallback missing")
    record='    res["steps_run"] = step + 1'
    if source.count(record)!=1:
        raise RuntimeError("pinned pdg.py result writer changed")
    source=source.replace(record,record+'\n    if a.wuke_sparse_mode == "block":\n'
        '        res["wuke_pointer_stats"]=getattr(m,"wuke_stats",{})',1)
    path.write_text(source)
