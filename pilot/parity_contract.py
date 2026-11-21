"""Shared, lazy runtime and verifier identity for official CUDA receipts.

Importing this module needs only the Python standard library. In particular,
the reference runner does not need Triton or Flash Linear Attention installed.
Dependency imports and CUDA inspection happen only inside runtime_fingerprint.
"""
import hashlib
import importlib
from importlib import metadata
import inspect
from pathlib import Path
import platform


def verification_sha256():
    """Bind both the validator and this contract, including file identities."""
    directory = Path(__file__).resolve().parent
    digest = hashlib.sha256(b"predskip-verification-contract-v1\0")
    for name in ("validate_official.py", "parity_contract.py"):
        encoded_name = name.encode("utf-8")
        contents = (directory / name).read_bytes()
        for part in (encoded_name, contents):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
    return digest.hexdigest()


def _distribution_versions(package_name, known_names, package_distributions):
    names = sorted(set(known_names) | set(package_distributions.get(package_name, ())))
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            # Editable/source-only installs may lack a distribution record.
            # The module version and source digest remain separately recorded.
            versions[name] = None
    return versions


def runtime_fingerprint():
    """Return JSON-stable fields that must match the successful CUDA receipt.

The l2norm digest is taken from the file defining the function actually
imported through fla.modules.l2norm, including compatibility aliases. It is
not a guessed path under an unrelated checkout. Missing dependencies or an
uninspectable normalizer fail closed at the official validation boundary.
"""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("the official runtime fingerprint requires CUDA")
    triton = importlib.import_module("triton")
    fla = importlib.import_module("fla")
    normalization_module = importlib.import_module("fla.modules.l2norm")
    normalizer = inspect.unwrap(getattr(normalization_module, "l2norm_fwd"))
    normalization_source = inspect.getsourcefile(normalizer)
    if normalization_source is None:
        raise RuntimeError("cannot identify the imported FLA l2norm_fwd source")
    normalization_file = Path(normalization_source).resolve()
    if not normalization_file.is_file():
        raise RuntimeError("the imported FLA l2norm_fwd source is unavailable")
    distributions = metadata.packages_distributions()
    matmul = torch.backends.cuda.matmul
    cudnn = torch.backends.cudnn
    version = lambda module: (str(module.__version__) if getattr(module, "__version__", None) is not None else None)
    return {
        "schema_version": 1,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cudnn": cudnn.version(),
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "triton": {
            "module_version": version(triton),
            "distributions": _distribution_versions("triton", ("triton",), distributions),
        },
        "flash_linear_attention": {
            "module_version": version(fla),
            "distributions": _distribution_versions(
                "fla", ("flash-linear-attention", "fla-core"), distributions),
        },
        "fla_l2norm_module": normalizer.__module__,
        "fla_l2norm_sha256": hashlib.sha256(normalization_file.read_bytes()).hexdigest(),
        "precision_flags": {
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": bool(matmul.allow_tf32),
            "cuda_matmul_allow_fp16_reduced_precision_reduction": bool(
                matmul.allow_fp16_reduced_precision_reduction),
            "cuda_matmul_allow_bf16_reduced_precision_reduction": bool(
                matmul.allow_bf16_reduced_precision_reduction),
            "cudnn_allow_tf32": bool(cudnn.allow_tf32),
            "cudnn_benchmark": bool(cudnn.benchmark),
            "cudnn_deterministic": bool(cudnn.deterministic),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
    }
