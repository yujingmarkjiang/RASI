"""User-facing diagnostics and smoke checks for a RASI installation.

These checks validate execution paths and hardware availability. They are not
manuscript experiments and do not inspect reserved audit outcomes.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
import ctypes
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import time

import numpy as np

from . import __version__
from .config import ModelConfig, RASIConfig
from .data import Contrast, Query, Trajectory
from .models.base import FeatureSchema, FitRequest, load_model, make_factory


def _command_version(command: str, args: tuple[str, ...] = ("--version",)) -> dict[str, Any]:
    path = shutil.which(command)
    if path is None:
        return {"available": False, "path": None, "version": None}
    try:
        completed = subprocess.run([path, *args], capture_output=True, text=True, timeout=5, check=False)
        output = (completed.stdout or completed.stderr).strip().splitlines()
        version = output[0] if output else None
        return {"available": completed.returncode == 0, "path": path, "version": version}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "path": path, "version": None, "error": str(exc)}


def _native_library() -> dict[str, Any]:
    default = Path(__file__).resolve().parents[2] / "cpp" / "build" / "librasi.so"
    path = Path(os.environ.get("RASI_NATIVE_LIB", str(default)))
    result: dict[str, Any] = {"path": str(path), "exists": path.exists(), "loadable": False}
    if path.exists():
        try:
            ctypes.CDLL(str(path))
            result["loadable"] = True
        except OSError as exc:
            result["error"] = str(exc)
    return result


def torch_report() -> dict[str, Any]:
    if importlib.util.find_spec("torch") is None:
        return {"installed": False, "cuda_available": False, "devices": []}
    import torch

    result: dict[str, Any] = {
        "installed": True,
        "version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cudnn_available": bool(torch.backends.cudnn.is_available()),
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "devices": [],
    }
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            result["devices"].append({
                "index": index,
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_bytes": int(props.total_memory),
                "multi_processor_count": int(props.multi_processor_count),
            })
    return result


def system_report(config: RASIConfig | None = None) -> dict[str, Any]:
    torch_info = torch_report()
    native = _native_library()
    report: dict[str, Any] = {
        "rasi_version": __version__,
        "python": {"version": sys.version.split()[0], "executable": sys.executable, "implementation": platform.python_implementation()},
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "numpy": {"version": np.__version__},
        "torch": torch_info,
        "native_backend": native,
        "toolchain": {
            "cxx": _command_version(os.environ.get("CXX", "c++")),
            "go": _command_version("go", ("version",)),
        },
    }
    if config is not None:
        issues: list[str] = []
        if config.model.kind == "gru" and not torch_info["installed"]:
            issues.append("GRU selected but PyTorch is not installed; install the neural extra.")
        if config.model.device.startswith("cuda"):
            if not torch_info["cuda_available"]:
                issues.append(f"Config requests {config.model.device}, but CUDA is not available to PyTorch.")
            else:
                try:
                    import torch
                    requested = torch.device(config.model.device)
                    index = 0 if requested.index is None else requested.index
                    if index < 0 or index >= len(torch_info["devices"]):
                        issues.append(f"Config requests {config.model.device}, but only {len(torch_info['devices'])} CUDA device(s) are visible.")
                except (RuntimeError, ValueError) as exc:
                    issues.append(f"Invalid CUDA device {config.model.device!r}: {exc}")
        if config.model.kind == "linear" and config.model.backend == "native" and not native["loadable"]:
            issues.append("Native linear backend selected, but librasi.so is not loadable; run `make native` or set RASI_NATIVE_LIB.")
        report["config"] = config.to_dict()
        report["config_ready"] = not issues
        report["config_issues"] = issues
    return report


def gpu_test(device: str = "cuda:0", matrix_size: int = 512, repeats: int = 3) -> dict[str, Any]:
    if matrix_size < 16:
        raise ValueError("matrix_size must be at least 16")
    if repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if importlib.util.find_spec("torch") is None:
        return {"passed": False, "status": "unavailable", "device": device, "reason": "PyTorch is not installed", "action": "Install with `python -m pip install -e '.[neural]'`."}
    import torch

    requested = torch.device(device)
    if requested.type != "cuda":
        raise ValueError("GPU testing requires a CUDA device such as cuda:0")
    if not torch.cuda.is_available():
        return {"passed": False, "status": "unavailable", "device": device, "reason": "torch.cuda.is_available() is false",
                "torch_version": torch.__version__, "cuda_build": torch.version.cuda,
                "action": "Install a CUDA-enabled PyTorch build and verify the NVIDIA driver/runtime."}
    if requested.index is not None and requested.index >= torch.cuda.device_count():
        return {"passed": False, "status": "failed", "device": device, "reason": f"device index is outside the {torch.cuda.device_count()} visible CUDA devices"}

    torch.cuda.set_device(requested)
    torch.cuda.empty_cache()
    props = torch.cuda.get_device_properties(requested)
    free_before, total = torch.cuda.mem_get_info(requested)
    timings: list[float] = []
    checksum = None
    try:
        for _ in range(repeats):
            a = torch.randn((matrix_size, matrix_size), device=requested, dtype=torch.float32, requires_grad=True)
            b = torch.randn((matrix_size, matrix_size), device=requested, dtype=torch.float32)
            torch.cuda.synchronize(requested)
            started = time.perf_counter()
            loss = (a @ b).square().mean()
            loss.backward()
            torch.cuda.synchronize(requested)
            timings.append(time.perf_counter() - started)
            checksum = float(loss.detach().cpu())
            if not bool(torch.isfinite(loss)) or a.grad is None or not bool(torch.isfinite(a.grad).all()):
                raise FloatingPointError("CUDA forward/backward produced non-finite values")
            del a, b, loss
        free_after, _ = torch.cuda.mem_get_info(requested)
        return {
            "passed": True,
            "status": "passed",
            "device": str(requested),
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "torch_version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "matrix_size": matrix_size,
            "repeats": repeats,
            "elapsed_seconds": timings,
            "mean_elapsed_seconds": float(np.mean(timings)),
            "checksum": checksum,
            "memory_total_bytes": int(total),
            "memory_free_before_bytes": int(free_before),
            "memory_free_after_bytes": int(free_after),
        }
    except RuntimeError as exc:
        return {"passed": False, "status": "failed", "device": str(requested), "name": props.name, "reason": str(exc),
                "action": "Check CUDA/PyTorch compatibility and available device memory."}
    finally:
        torch.cuda.empty_cache()


def _toy_data() -> tuple[tuple[Trajectory, ...], tuple[Contrast, ...], tuple[Trajectory, ...], tuple[Query, ...]]:
    times = (0.0, 1.0, 2.0)
    real: list[Trajectory] = []
    contrasts: list[Contrast] = []
    queries: list[Query] = []
    for i, subject in enumerate(("s1", "s2", "s3", "s4")):
        qa = Query(f"{subject}-a", subject, "A", times, ((float(i),),) * 3)
        qb = Query(f"{subject}-b", subject, "B", times, ((float(i),),) * 3)
        ya = tuple(1.0 + 0.1 * i + 0.2 * t for t in times)
        yb = tuple(v + 0.5 for v in ya)
        oa, ob = Trajectory(qa, ya), Trajectory(qb, yb)
        real.extend((oa, ob)); queries.extend((qa, qb))
        contrasts.append(Contrast(f"c-{subject}", (qb,), (qa,), tuple(b - a for a, b in zip(ya, yb)), subject, "randomised_crossover"))
    validation: list[Trajectory] = []
    for treatment, offset in (("A", 0.0), ("B", 0.5)):
        q = Query(f"val-{treatment}", "val", treatment, times, ((4.0,),) * 3)
        y = tuple(1.4 + offset + 0.2 * t for t in times)
        validation.append(Trajectory(q, y)); queries.append(q)
    return tuple(real), tuple(contrasts), tuple(validation), tuple(queries)


def model_smoke(config: ModelConfig, epochs: int = 3) -> dict[str, Any]:
    real, contrasts, validation, queries = _toy_data()
    effective = config
    if config.kind == "gru":
        effective = replace(config, epochs=max(1, min(config.epochs, epochs)), early_stopping_patience=max(1, min(config.early_stopping_patience, 2)))
    schema = FeatureSchema.build(queries, real)
    factory = make_factory(effective, queries, real)
    model = factory(1234)
    started = time.perf_counter()
    model.fit(FitRequest(real, contrasts, 1.0, validation))
    pred = model.predict(queries[0])
    state = model.state_dict()
    restored = load_model(state, device=effective.device if effective.kind == "gru" else None)
    restored_pred = restored.predict(queries[0])
    elapsed = time.perf_counter() - started
    if pred.shape != (3,) or not np.all(np.isfinite(pred)):
        raise RuntimeError("Smoke prediction was malformed or non-finite")
    if not np.allclose(pred, restored_pred, rtol=1e-5, atol=1e-5):
        raise RuntimeError("Checkpoint round-trip changed model predictions")
    return {
        "passed": True,
        "model": effective.kind,
        "backend": effective.backend if effective.kind == "linear" else None,
        "device": effective.device,
        "elapsed_seconds": elapsed,
        "prediction": [float(x) for x in pred],
        "checkpoint_roundtrip": True,
        "fit_metadata": getattr(model, "fit_metadata", {}),
    }
