"""Shared pinned-upstream, bounded-run helpers for the 2026-11 research pilots.
Reference source remains in an ignored work directory. No source results are claimed
unless a subprocess actually completed and emitted a matching JSON record.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

UPSTREAM_URL = "https://github.com/vukrosic/llm-research-paper-drafts.git"
UPSTREAM_SHA = "d0dc052749a97061f6daa6907371bd0a2ec33405"


def ensure_upstream(work):
    root = Path(work) / "llm-research-paper-drafts"
    root.parent.mkdir(parents=True, exist_ok=True)
    if not (root / ".git").exists():
        subprocess.run(["git", "clone", "--quiet", UPSTREAM_URL, str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "checkout", "--quiet", "--detach", UPSTREAM_SHA], check=True)
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if actual != UPSTREAM_SHA:
        raise RuntimeError("Upstream SHA does not match pin")
    return root


def add_seed_flag(path):
    path = Path(path)
    s = path.read_text()
    if 'p.add_argument("--seed", type=int' in s:
        return
    marker = "a = p.parse_args()"
    assert s.count(marker) == 1, "upstream argparse changed: stop, do not guess"
    s = s.replace(marker, 'p.add_argument("--seed", type=int, default=0)\n' + marker, 1)
    assert "torch.manual_seed(0)" in s, "upstream seeding changed: stop"
    s = s.replace("torch.manual_seed(0)", "torch.manual_seed(a.seed)", 1)
    s = s.replace("np.random.default_rng(0)", "np.random.default_rng(a.seed)")
    path.write_text(s)


def prepare_token_data(root, n_tokens, vocab=8000, real_dir=None):
    """Provide source-expected uint16 binary format with 1024-byte header.

    'proxy' is deliberately not FineWeb. Never describe synthetic results as real LM.
    Real data directory must contain train and held-out files; no training/val mixing.
    """
    import numpy as np
    root = Path(root)
    prefix = root / "llm-per-word-weights" / "data"
    prefix.mkdir(parents=True, exist_ok=True)
    mu = root / "02-muon-hypergradient-learning-rate" / "data"
    mu.mkdir(parents=True, exist_ok=True)
    if real_dir:
        source = Path(real_dir).expanduser().resolve()
        def src(name):
            for suffix in (".bin", ".slice"):
                p = source / (name + suffix)
                if p.exists():
                    return p
            raise FileNotFoundError(f"missing {name}(.bin/.slice) in {source}")
        tr = src("fineweb_train_000001")
        va = src("fineweb_val_000000")
        if (tr.stat().st_size - 1024)//2 < n_tokens:
            raise ValueError("real train file has too few tokens for requested cases")
        def link(dst, src_path):
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src_path)
        for ext in (".bin", ".slice"):
            link(prefix / ("fineweb_train_000001" + ext), tr)
            link(prefix / ("fineweb_val_000000" + ext), va)
            link(mu / ("fineweb_train_000001" + ext), tr)
            link(mu / ("fineweb_val_000000" + ext), va)
        counts_path = source / "counts.npy"
        if counts_path.exists():
            link(prefix / "counts.npy", counts_path)
        else:
            arr = np.memmap(tr, dtype=np.uint16, mode="r", offset=1024)
            np.save(prefix / "counts.npy", np.bincount(arr.astype(np.int64), minlength=65536))
        mode="real-pretokenized"
    else:
        n_tokens = max(n_tokens + 128, 180000)
        def generate(n, seed):
            rng = np.random.default_rng(seed)
            # Zipf-like long tail + document-local latent topic = a small,
            # learnable *synthetic* next-token task, not a FineWeb substitute.
            topic = rng.integers(0, 24, size=(n+63)//64, dtype=np.int32)
            local = np.minimum(rng.zipf(1.35, size=n)-1, max(1,vocab//24-1))
            ids = ((local * 24 + np.repeat(topic, 64)[:n]) % vocab).astype(np.uint16)
            ids[::997] = 50256
            return ids
        train = generate(n_tokens, 20261120)
        val = generate(80000, 20261121)
        def write(dst, a):
            dst.parent.mkdir(parents=True,exist_ok=True)
            with dst.open("wb") as f:
                f.write(bytes(1024))
                a.tofile(f)
        for name, arr in [("fineweb_train_000001", train),("fineweb_val_000000",val)]:
            for ext in (".bin",".slice"):
                write(prefix/(name+ext),arr)
                write(mu/(name+ext),arr)
        np.save(prefix/"counts.npy", np.bincount(train.astype(np.int64),minlength=65536))
        mode="synthetic-topic-zipf"
    manifest={"mode":mode, "train_tokens_requested":n_tokens,
              "upstream_sha": UPSTREAM_SHA, "real_dir":str(real_dir) if real_dir else None}
    (Path(root).parent/"data_manifest.json").write_text(json.dumps(manifest,indent=2))
    return manifest


def run_cases(source, cases, budget_seconds, report_dir, default_timeout=1200):
    """Hard run budget, per-case timeouts, resumable by tag, atomic report."""
    source = Path(source)
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
    else:
        report = {"source_sha": UPSTREAM_SHA, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"runs":{}}
    start=time.monotonic()
    for spec in cases:
        name=spec["name"]
        if name in report["runs"] and report["runs"][name].get("status")=="ok":
            print("SKIP completed:",name,flush=True)
            continue
        left=budget_seconds-(time.monotonic()-start)-30
        if left <= 40:
            print("BUDGET EXHAUSTED before:",name,flush=True)
            break
        script=source/spec["path"]
        run_dir=script.parent
        case_timeout=max(40,int(min(left,spec.get("timeout",default_timeout))))
        cmd=[sys.executable,str(script),"--tag",name] + [str(x) for x in spec["args"]]
        print("RUN",name,"timeout_s",case_timeout,flush=True)
        status="ok"
        try:
            p=subprocess.run(cmd,cwd=run_dir,capture_output=True,text=True,timeout=case_timeout)
            if p.returncode:status="failed"
            stdout,stderr=p.stdout,p.stderr
        except subprocess.TimeoutExpired as e:
            status="timeout"
            stdout=(e.stdout or b"").decode(errors="replace") if isinstance(e.stdout,bytes) else (e.stdout or "")
            stderr=(e.stderr or b"").decode(errors="replace") if isinstance(e.stderr,bytes) else (e.stderr or "")
        (report_dir/(name+".stdout.log")).write_text(stdout)
        (report_dir/(name+".stderr.log")).write_text(stderr)
        result_file=run_dir / spec.get("results_file","results.jsonl")
        record=None
        if status=="ok" and result_file.exists():
            with result_file.open() as f:
                for line in f:
                    try:
                        z=json.loads(line)
                        if z.get("tag")==name: record=z
                    except json.JSONDecodeError: continue
        if status=="ok" and record is None:
            status="missing-result"
        report["runs"][name]={"status":status, "cmd":cmd,
                              "record":record,"stderr_tail":stderr[-500:],
                              "data_mode":spec.get("data_mode","synthetic")}
        tmp=report_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(report,indent=2,allow_nan=False,default=str))
        tmp.replace(report_path)
        print(name,status,"score",record.get("scores",{}).get("val",record.get("acc")) if record else None,flush=True)
    return report


def decide_pair(report, first, second, key="val", margin=0.0, lower=True):
    a=report["runs"].get(first,{}).get("record")
    b=report["runs"].get(second,{}).get("record")
    if not (a and b):return {"status":"INCONCLUSIVE","reason":"missing comparison"}
    def get(r):
        return r["scores"][key] if key in r.get("scores",{}) else r[key]
    ga,gb=float(get(a)),float(get(b))
    advantage=(ga-gb) if lower else (gb-ga)
    return {"status":"PASS" if advantage>=margin else "FAIL","first":ga,"second":gb,
            "improvement":advantage,"threshold":margin}
