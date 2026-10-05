"""Atomic local checkpoints. No hidden labels, pickle payloads or network writes."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import hashlib
import json
import os
import tempfile
from . import __version__
from .models import WorldModel, load_model


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(canonical(value)); f.write(b"\n"); f.flush(); os.fsync(f.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def checkpoint(path: str | Path, model: WorldModel, metadata: dict[str, Any]) -> None:
    state = model.state_dict()
    body = {"schema_version": 1, "implementation_version": __version__, "model": state,
            "model_sha256": digest(state), "metadata": metadata}
    atomic_json(path, body)


def read_checkpoint(path: str | Path, device: str | None = None) -> tuple[WorldModel, dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        body = json.load(f)
    if body.get("schema_version") != 1 or digest(body.get("model")) != body.get("model_sha256"):
        raise ValueError("Unsupported or corrupted checkpoint")
    return load_model(body["model"], device=device), dict(body.get("metadata", {}))


class RunStore:
    def __init__(self, directory: str | Path, resume: bool = False):
        self.path = Path(directory)
        self.path.mkdir(parents=True, exist_ok=True)
        self.lock = self.path / ".writer.lock"
        try:
            fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as e:
            raise RuntimeError("Run directory is locked; never launch two writers into the same run. Remove a stale lock only after confirming its process exited") from e
        with os.fdopen(fd, "w") as f:
            f.write(str(os.getpid()))
        existing = (self.path / "resume.json").exists() or (self.path / "accepted.json").exists()
        if existing and not resume:
            self.close()
            raise FileExistsError("Run already exists; use --resume or choose a new output directory")
        if resume and not (self.path / "resume.json").exists():
            self.close()
            raise FileNotFoundError("No resumable state in this run")

    def append(self, event: dict[str, Any]) -> None:
        path = self.path / "events.jsonl"
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(fd, "ab") as f:
            f.write(canonical(event) + b"\n"); f.flush(); os.fsync(f.fileno())

    def close(self) -> None:
        self.lock.unlink(missing_ok=True)

    def __enter__(self) -> RunStore:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def atomic_jsonlines(path: str | Path, values: list[dict[str, Any]]) -> None:
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as f:
            for value in values:
                f.write(canonical(value) + b"\n")
            f.flush(); os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
