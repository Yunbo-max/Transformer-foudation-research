"""Source-reviewed developer runner. Scientific/native validation remains pending."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import time

import torch
from common import MemoryLM, PRESETS, train_case
from screening import require_complete_matrix, screen_results
from sg_delta import SGDeltaMixer, ARMS

DEFAULT = ["isotropic", "query-write", "query-decay", "query-joint", "key-joint", "channel-decay"]

def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def source_hashes():
    here = Path(__file__).resolve().parent
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(here.glob("*.py"))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", choices=PRESETS, default="pilot")
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    p.add_argument("--precision", choices=["float32", "bfloat16"], default="float32")
    p.add_argument("--methods", nargs="+", choices=ARMS, default=DEFAULT)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--steps", type=positive_int)
    p.add_argument("--chunk", type=positive_int)
    p.add_argument("--conv-size", type=int, default=4,
                   help="causal Q/K/V short convolution; 0 retains the old no-conv ablation")
    p.add_argument("--eval-batches", type=positive_int, default=16)
    p.add_argument("--eval-batch-size", type=positive_int, default=16)
    p.add_argument("--budget-seconds", type=positive_int, default=10200)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--output", default="runs/sgdelta_joint")
    p.add_argument("--mu", type=float, default=8.)
    p.add_argument("--lam", type=float, default=1.)
    p.add_argument("--tau", type=float, default=1.)
    p.add_argument("--ema", type=float, default=.98)
    p.add_argument("--geometry-scale", choices=["raw", "dimension"], default="raw")
    p.add_argument("--differentiate-statistics", action="store_true")
    args = p.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        p.error("training seeds must be distinct nonnegative integers")
    if len(set(args.methods)) != len(args.methods):
        p.error("methods must be distinct")
    if args.conv_size < 0 or not (0 < args.learning_rate < 1):
        p.error("conv-size must be nonnegative and learning-rate must be in (0,1)")
    if args.budget_seconds > 10800:
        p.error("budget must not exceed 10800 seconds")
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA requested but unavailable")
    if (args.precision == "bfloat16" and args.device == "cuda"
            and not torch.cuda.is_bf16_supported()):
        p.error("this CUDA device does not support bfloat16; use float32")
    if args.precision == "bfloat16" and args.device == "mps":
        p.error("this runner does not qualify bfloat16 on MPS")
    if args.profile in ("100m-sizing", "340m-sizing") and not args.dry_run:
        p.error("large profiles remain sizing-only pending efficient implementation and native evaluation")
    cfg = PRESETS[args.profile].copy()
    cfg.update(steps=args.steps if args.steps is not None else cfg["steps"],
               chunk=args.chunk if args.chunk is not None else cfg["chunk"],
               conv_size=args.conv_size, eval_batches=args.eval_batches,
               eval_batch_size=args.eval_batch_size, precision=args.precision,
               learning_rate=args.learning_rate)
    if not (args.mu >= 0 and args.lam > 0 and args.tau > 0 and 0 <= args.ema < 1):
        p.error("require mu>=0, lam>0, tau>0 and 0<=ema<1")
    cfg.update(mu=args.mu, lam=args.lam, tau=args.tau, ema=args.ema,
               geometry_scale=args.geometry_scale,
               detach_statistics=not args.differentiate_statistics)
    def mixer_factory(dim, layer):
        return SGDeltaMixer(
            dim, heads=cfg["heads"], key_dim=cfg["head_dim"], chunk=cfg["chunk"],
            mode=current_arm, mu=args.mu, lam=args.lam, tau=args.tau, ema=args.ema,
            detach_statistics=not args.differentiate_statistics,
            geometry_scale=args.geometry_scale, conv_size=args.conv_size)
    hashes = source_hashes()
    nparams = None
    if args.dry_run:
        for current_arm in args.methods:
            with torch.device("meta"):
                model = MemoryLM(cfg.get("vocab_size", 4 + 2 * cfg["key_space"]),
                                 cfg["dim"], cfg["layers"], mixer_factory)
            print(json.dumps(dict(method=current_arm, config=cfg,
                                  params=sum(z.numel() for z in model.parameters()),
                                  status="sizing_only")))
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = Path(args.output) / stamp
    out.mkdir(parents=True, exist_ok=False)
    print("OUTPUT", out.resolve(), flush=True)
    start = time.monotonic()
    cases = [(seed, arm) for seed in args.seeds for arm in args.methods]
    results = {}
    manifest = dict(
        profile=args.profile, config=cfg, source_sha256=hashes,
        requested_methods=args.methods, requested_seeds=args.seeds,
        data="existing developer diagnostic, not official MQAR",
        execution_status="running", result_files={})
    write_json(out / "manifest.json", manifest)
    for index, (seed, current_arm) in enumerate(cases):
        remaining = args.budget_seconds - (time.monotonic() - start)
        if remaining < 35:
            break
        case_seconds = .93 * remaining / (len(cases) - index)
        if case_seconds < 5:
            break
        torch.manual_seed(seed)
        model = MemoryLM(cfg.get("vocab_size", 4 + 2 * cfg["key_space"]),
                         cfg["dim"], cfg["layers"], mixer_factory)
        nparams = sum(z.numel() for z in model.parameters())
        name = f"{current_arm}_seed{seed}"
        print("RUN", name, "params", nparams, "budget_s", round(case_seconds, 1), flush=True)
        try:
            row = train_case(
                model, cfg, device=args.device, seed=seed, steps=cfg["steps"],
                budget_seconds=case_seconds, out_path=out / f"{name}.json")
            results[name] = row
            manifest["result_files"][name] = f"{name}.json"
        except Exception as exc:
            results[name] = dict(status="error", error=repr(exc))
            (out / f"{name}.error.txt").write_text(repr(exc))
            manifest["result_files"][name] = f"{name}.error.txt"
            print("ERROR", name, repr(exc), flush=True)
        manifest["statuses"] = {key: value["status"] for key, value in results.items()}
        write_json(out / "manifest.json", manifest)
    report = screen_results(
        results, args.seeds, candidate="query-joint",
        comparators={"isotropic": .02, "channel-decay": .02, "key-joint": .03,
                     "query-write": .02, "query-decay": .02},
        learning_reference="isotropic", plasticity_references=["isotropic", "key-joint"])
    require_complete_matrix(report, results, cases)
    report.update(actual_params=nparams, elapsed_seconds=time.monotonic() - start,
                  source_sha256=hashes, output=str(out.resolve()))
    write_json(out / "decision.json", report)
    manifest["execution_status"] = report["execution_status"]
    manifest["decision_file"] = "decision.json"
    write_json(out / "manifest.json", manifest)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
