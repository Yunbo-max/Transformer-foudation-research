"""Local CUDA implementation parity; never a method-performance verdict.

Checks the actual training precision, every primitive input gradient, final
states, model input gradients and every used model parameter. Produces a
source- and configuration-bound receipt for run.py. Generated but unexecuted
in the Web host. No dependencies are installed by this script.
"""
import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F

from predicted_skip import ARMS, GUARD_MODES, GDN2PredSkipMixer
from parity_contract import runtime_fingerprint, verification_sha256

PINNED_OFFICIAL_COMMIT = "de8476de0fecae08fc16f0bf3154900244c61c73"
TOLERANCES = {
    "float32": {"atol": 2e-4, "rtol": .01, "relative_l2": .02},
    "bfloat16": {"atol": .004, "rtol": .05, "relative_l2": .08},
}


def compare_tensor(reference, official, tolerances, *, allow_unused=False):
    if reference is None or official is None:
        passed = allow_unused and reference is None and official is None
        return {"status": "PASS" if passed else "FAIL", "reference_missing": reference is None,
                "official_missing": official is None}
    if reference.shape != official.shape:
        return {"status": "FAIL", "reference_shape": list(reference.shape),
                "official_shape": list(official.shape)}
    finite = bool(torch.isfinite(reference).all() and torch.isfinite(official).all())
    if not finite:
        return {"status": "FAIL", "finite": False}
    a, b = reference.detach().double(), official.detach().double()
    difference = a - b
    relative = float(difference.norm() / torch.maximum(a.norm(), b.norm()).clamp_min(1e-8))
    allclose = bool(torch.allclose(a, b, atol=tolerances["atol"], rtol=tolerances["rtol"]))
    passed = allclose and relative <= tolerances["relative_l2"]
    return {"status": "PASS" if passed else "FAIL", "finite": True, "allclose": allclose,
            "max_abs": float(difference.abs().max()), "relative_l2": relative,
            "reference_norm": float(a.norm()), "official_norm": float(b.norm())}


def check_group(checks):
    return "PASS" if checks and all(row["status"] == "PASS" for row in checks.values()) else "FAIL"


def loss_with_state(output, state, output_probe, state_probe):
    return ((output.float() * output_probe).sum() / math.sqrt(output.numel()) +
            (state.float() * state_probe).sum() / math.sqrt(state.numel()))


def primitive_case(config, guard, length, precision, official_root, tolerances):
    torch.manual_seed(417 + length)
    dtype = torch.float32 if precision == "float32" else torch.bfloat16
    batch, heads, keys, values = 2, config["heads"], config["key_dim"], config["value_dim"]
    key_shape, value_shape = (batch, heads, length, keys), (batch, heads, length, values)
    source = {
        "q": torch.randn(key_shape, device="cuda", dtype=dtype),
        "k": torch.randn(key_shape, device="cuda", dtype=dtype),
        "v": torch.randn(value_shape, device="cuda", dtype=dtype),
        "g": -.02 * F.softplus(torch.randn(key_shape, device="cuda")),
        "b": torch.linspace(.02, .98, keys, device="cuda", dtype=dtype).expand(key_shape).clone(),
        "w": torch.sigmoid(torch.randn(value_shape, device="cuda", dtype=dtype)),
        "state": .1 * torch.randn(batch, heads, keys, values, device="cuda"),
    }
    inputs = [{name: value.detach().clone().requires_grad_() for name, value in source.items()}
              for _ in range(2)]
    reference = GDN2PredSkipMixer(**config, guard_mode=guard).cuda()
    official = GDN2PredSkipMixer(**config, guard_mode=guard, backend="official",
                                 official_root=official_root).cuda()
    arguments = lambda values: [values[name] for name in ("q", "k", "v", "g", "b", "w", "state")]
    yr, sr = reference._reference_chunk(*arguments(inputs[0]))
    yo, so = official._official_chunk(*arguments(inputs[1]))
    output_probe, state_probe = torch.randn_like(yr.float()), torch.randn_like(sr)
    loss_with_state(yr, sr, output_probe, state_probe).backward()
    loss_with_state(yo, so, output_probe, state_probe).backward()
    checks = {"output": compare_tensor(yr, yo, tolerances),
              "final_state": compare_tensor(sr, so, tolerances)}
    checks.update({"gradient_" + name: compare_tensor(inputs[0][name].grad, inputs[1][name].grad, tolerances)
                   for name in source})
    return {"guard": guard, "length": length, "status": check_group(checks), "checks": checks}


def model_case(config, guard, method, precision, official_root, tolerances):
    torch.manual_seed(941)
    reference = GDN2PredSkipMixer(**config, mode=method, guard_mode=guard).cuda()
    official = GDN2PredSkipMixer(**config, mode=method, guard_mode=guard,
                                 backend="official", official_root=official_root).cuda()
    official.load_state_dict(reference.state_dict())
    length = max(2 * config["chunk"] + 1, 5)
    x = torch.randn(2, length, config["hidden"], device="cuda")
    xr, xo = x.detach().clone().requires_grad_(), x.detach().clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bfloat16"):
        yr, _, sr = reference(xr, return_final_state=True)
        yo, _, so = official(xo, return_final_state=True)
    output_probe, state_probe = torch.randn_like(yr.float()), torch.randn_like(sr)
    loss_with_state(yr, sr, output_probe, state_probe).backward()
    loss_with_state(yo, so, output_probe, state_probe).backward()
    checks = {"output": compare_tensor(yr, yo, tolerances),
              "final_state": compare_tensor(sr, so, tolerances),
              "input_gradient": compare_tensor(xr.grad, xo.grad, tolerances)}
    official_parameters = dict(official.named_parameters())
    for name, parameter in reference.named_parameters():
        checks["parameter_gradient:" + name] = compare_tensor(
            parameter.grad, official_parameters[name].grad, tolerances,
            allow_unused=method == "baseline" and name == "log_gamma")
    return {"guard": guard, "method": method, "length": length,
            "status": check_group(checks), "checks": checks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--precision", choices=TOLERANCES, default="float32")
    parser.add_argument("--guards", nargs="+", choices=GUARD_MODES, default=list(GUARD_MODES))
    parser.add_argument("--methods", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--key-dim", type=int, default=16)
    parser.add_argument("--value-dim", type=int, default=None)
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--conv-size", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=.25)
    parser.add_argument("--gate-memory-detach", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", default="runs/predskip_joint/parity.json")
    args = parser.parse_args()
    output = Path(args.output)
    config = {"hidden": args.hidden, "heads": args.heads, "key_dim": args.key_dim,
              "value_dim": args.value_dim if args.value_dim is not None else args.key_dim,
              "chunk": args.chunk, "conv_size": args.conv_size, "gamma": args.gamma,
              "gate_memory_detach": args.gate_memory_detach}
    source_file = Path(__file__).with_name("predicted_skip.py")
    report = {
        "status": "FAIL", "verification_scope": "implementation_parity_only",
        "source_sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
        "verification_sha256": verification_sha256(),
        "official_commit": None, "precision": args.precision,
        "guard_modes": list(dict.fromkeys(args.guards)), "methods": list(dict.fromkeys(args.methods)),
        "model_config": config, "tolerances": TOLERANCES[args.precision],
        "primitive_cases": [], "model_cases": [],
        "limitations": ["No benchmark-performance claim", "Float32 reference state recurrence",
                        "Native guard reference has the same STE policy; CUDA cumsum rounding is tested numerically"],
    }
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; no parity receipt can be issued on CPU")
        official_root = Path(args.official_root).expanduser().resolve()
        commit = subprocess.run(["git", "-C", str(official_root), "rev-parse", "HEAD"],
                                check=True, capture_output=True, text=True).stdout.strip()
        report["official_commit"] = commit
        if commit != PINNED_OFFICIAL_COMMIT:
            raise RuntimeError(f"official checkout must be pinned to {PINNED_OFFICIAL_COMMIT}")
        dirty = subprocess.run(["git", "-C", str(official_root), "status", "--porcelain", "--untracked-files=no"],
                               check=True, capture_output=True, text=True).stdout.strip()
        if dirty:
            raise RuntimeError("official tracked files have local modifications")
        # Training compares this full fingerprint, rather than just displaying
        # hardware metadata. FLA/Triton imports are lazy and official-only.
        report["runtime"] = runtime_fingerprint()
        # Cover a one-token chunk, a short tail, and NVIDIA's internal 64-token boundary.
        for guard in report["guard_modes"]:
            for length in sorted({1, 63, 65, config["chunk"] + 1}):
                try:
                    row = primitive_case(config, guard, length, args.precision,
                                         str(official_root), report["tolerances"])
                except Exception as exc:
                    row = {"guard": guard, "length": length, "status": "FAIL", "error": repr(exc)}
                report["primitive_cases"].append(row)
            for method in report["methods"]:
                try:
                    row = model_case(config, guard, method, args.precision,
                                     str(official_root), report["tolerances"])
                except Exception as exc:
                    row = {"guard": guard, "method": method, "status": "FAIL", "error": repr(exc)}
                report["model_cases"].append(row)
        cases = report["primitive_cases"] + report["model_cases"]
        if report["runtime"] != runtime_fingerprint():
            raise RuntimeError("runtime or precision flags changed during validation")
        report["status"] = "PASS" if cases and all(row["status"] == "PASS" for row in cases) else "FAIL"
    except Exception as exc:
        report["error"] = repr(exc)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "output": str(output),
                      "precision": args.precision, "guard_modes": report["guard_modes"],
                      "methods": report["methods"]}, indent=2))
    if report["status"] != "PASS":
        raise SystemExit("Parity did not pass; inspect the saved receipt before training")


if __name__ == "__main__":
    main()
