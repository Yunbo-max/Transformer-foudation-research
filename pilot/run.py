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
from predicted_skip import GDN2PredSkipMixer, ARMS
from parity_contract import runtime_fingerprint, verification_sha256

DEFAULT = ["baseline", "write", "joint", "shuffled", "keyread"]

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
    p.add_argument("--output", default="runs/predskip_joint")
    p.add_argument("--backend", choices=["reference", "official"], default="reference")
    p.add_argument("--official-root")
    p.add_argument("--parity-report",
                   help="successful matching validate_official.py JSON is required for official training")
    p.add_argument("--guard", choices=["raw", "native"], default="native")
    p.add_argument("--gamma", type=float, default=.25)
    p.add_argument("--differentiate-gate-memory", action="store_true")
    p.add_argument("--diagnostics", action="store_true",
                   help="reference-only per-token active-edit diagnostics during evaluation")
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
    if "shuffled" in args.methods and min(cfg["batch"], cfg["eval_batch_size"]) < 2:
        p.error("shuffled control requires train and evaluation batch sizes >= 2")
    if not (0 < args.gamma <= 20):
        p.error("gamma must be in (0,20]")
    if args.backend == "official" and (args.device != "cuda" or not args.official_root):
        p.error("official backend requires CUDA and --official-root")
    if args.backend == "official" and args.diagnostics:
        p.error("per-token active-edit diagnostics require --backend reference")
    cfg.update(backend=args.backend, guard_mode=args.guard, gamma=args.gamma,
               gate_memory_detach=not args.differentiate_gate_memory,
               diagnostics=args.diagnostics)
    if args.backend == "official" and not args.dry_run:
        if not args.parity_report:
            p.error("run validate_official.py and provide --parity-report")
        receipt = json.loads(Path(args.parity_report).read_text())
        current_hash = hashlib.sha256(
            Path(__file__).with_name("predicted_skip.py").read_bytes()).hexdigest()
        expected_shape = dict(hidden=cfg["dim"], heads=cfg["heads"],
                              key_dim=cfg["head_dim"], value_dim=cfg["head_dim"],
                              chunk=cfg["chunk"], conv_size=cfg["conv_size"],
                              gamma=args.gamma,
                              gate_memory_detach=not args.differentiate_gate_memory)
        upstream = subprocess.run(
            ["git", "-C", args.official_root, "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True).stdout.strip()
        clean = subprocess.run(
            ["git", "-C", args.official_root, "diff", "--quiet", "HEAD"],
            check=False).returncode == 0
        try:
            current_runtime = runtime_fingerprint()
            current_verification_hash = verification_sha256()
        except Exception as exc:
            p.error(f"cannot verify the official runtime or verifier: {exc}")
        valid = (
            receipt.get("status") == "PASS"
            and receipt.get("source_sha256") == current_hash
            and receipt.get("verification_sha256") == current_verification_hash
            and receipt.get("runtime") == current_runtime
            and receipt.get("official_commit") == upstream
            and upstream == "de8476de0fecae08fc16f0bf3154900244c61c73"
            and clean
            and receipt.get("precision") == args.precision
            and args.guard in receipt.get("guard_modes", [])
            and set(args.methods) <= set(receipt.get("methods", []))
            and receipt.get("model_config") == expected_shape
        )
        if not valid:
            p.error("parity report is failed, stale or mismatched; regenerate for these exact sources, verifier, runtime and settings")
    def mixer_factory(dim, layer):
        return GDN2PredSkipMixer(
            dim, heads=cfg["heads"], key_dim=cfg["head_dim"], chunk=cfg["chunk"],
            mode=current_arm, gamma=args.gamma, backend=args.backend,
            official_root=args.official_root, guard_mode=args.guard,
            gate_memory_detach=not args.differentiate_gate_memory,
            conv_size=args.conv_size, diagnostics=args.diagnostics)
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
        results, args.seeds, candidate="joint",
        comparators={"baseline": .03, "write": .03},
        learning_reference="baseline", plasticity_references=["baseline"])
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
