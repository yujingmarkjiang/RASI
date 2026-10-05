"""Validated, frozen settings. No experiment grids or hidden-outcome options."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import math
from typing import Any


def _finite(x: float, name: str, lower: float = 0.0, strict: bool = False) -> None:
    if not math.isfinite(x) or (x <= lower if strict else x < lower):
        raise ValueError(f"{name} must be finite and {'>' if strict else '>='} {lower}")


@dataclass(frozen=True)
class ModelConfig:
    kind: str = "linear"
    ridge: float = 0.01
    backend: str = "numpy"
    hidden_size: int = 64
    embedding_size: int = 4
    epochs: int = 200
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    early_stopping_patience: int = 20
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.kind not in {"linear", "gru"}:
            raise ValueError("Built-in models are 'linear' and 'gru'; register other adapters explicitly in Python")
        if self.backend not in {"numpy", "native"}:
            raise ValueError("linear backend must be numpy or native")
        _finite(self.ridge, "ridge")
        _finite(self.learning_rate, "learning_rate", strict=True)
        _finite(self.weight_decay, "weight_decay")
        for name in ("hidden_size", "embedding_size", "epochs", "batch_size", "early_stopping_patience"):
            if not isinstance(getattr(self, name), int) or isinstance(getattr(self, name), bool) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.kind == "linear" and self.device != "cpu":
            raise ValueError("The linear solver is CPU-only; it is not a CUDA implementation")


@dataclass(frozen=True)
class AuditConfig:
    size: int = 8                    # independent units, not correlated time points
    mode: str = "rotated"           # rotated, fixed, held_out
    allow_reuse: bool = True
    margin: str = "standard_error"  # main Methods; fixed = Supplementary Note 16
    epsilon: float = 0.0
    delta: float = 0.05
    sigma_c: float | None = None     # externally justified bound, never a fitted residual SD
    require_certificate: bool = False
    independent_units_asserted: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.size, int) or isinstance(self.size, bool) or self.size < 2:
            raise ValueError("audit.size must be at least two independent units")
        if self.mode not in {"rotated", "fixed", "held_out"}:
            raise ValueError("audit.mode must be rotated, fixed or held_out")
        for name in ("allow_reuse", "require_certificate", "independent_units_asserted"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"audit.{name} must be a JSON boolean")
        if self.mode == "fixed" and not self.allow_reuse:
            raise ValueError("A fixed audit is reused by definition; choose rotated for non-reused subsets")
        if self.margin not in {"standard_error", "fixed"}:
            raise ValueError("audit.margin must be standard_error or fixed")
        _finite(self.epsilon, "epsilon")
        if not 0 < self.delta < 1:
            raise ValueError("delta must lie strictly between zero and one")
        if self.sigma_c is not None:
            _finite(self.sigma_c, "sigma_c", strict=True)
        if self.require_certificate:
            if self.mode != "rotated" or self.allow_reuse or self.margin != "fixed":
                raise ValueError("A Note-16 certificate requires fresh, non-reused audits and a fixed epsilon")
            if self.sigma_c is None or not self.independent_units_asserted:
                raise ValueError("A certificate requires externally justified sigma_c and asserted independent units")


@dataclass(frozen=True)
class SelectionConfig:
    folds: int = 5
    lambda_grid: tuple[float, ...] = (0.3, 1.0, 3.0)
    k_grid: tuple[int, ...] = (1, 3, 5, 8)

    def __post_init__(self) -> None:
        object.__setattr__(self, "lambda_grid", tuple(self.lambda_grid))
        object.__setattr__(self, "k_grid", tuple(self.k_grid))
        if not isinstance(self.folds, int) or isinstance(self.folds, bool) or self.folds < 2:
            raise ValueError("Use at least two visible-data CV folds")
        if not self.lambda_grid or not self.k_grid:
            raise ValueError("Selection grids must not be empty")
        for x in self.lambda_grid:
            _finite(x, "lambda_grid entry", strict=True)
        if any(not isinstance(k, int) or isinstance(k, bool) or k < 1 for k in self.k_grid):
            raise ValueError("All retrieval k values must be positive integers")


@dataclass(frozen=True)
class RASIConfig:
    rounds: int = 15
    rollouts_per_arm: int = 5
    seed: int = 20261001
    regime: str = "replacement"
    synthetic_weight: float = 1.0
    rejection_patience: int = 10
    verifier_z: float = 3.0
    model: ModelConfig = field(default_factory=ModelConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.rounds, int) or isinstance(self.rounds, bool) or self.rounds < 0:
            raise ValueError("rounds must be a nonnegative integer")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not isinstance(self.rollouts_per_arm, int) or isinstance(self.rollouts_per_arm, bool) or self.rollouts_per_arm < 1:
            raise ValueError("rollouts_per_arm must be a positive integer")
        if self.regime not in {"replacement", "accumulation"}:
            raise ValueError("regime must be replacement or accumulation")
        if self.rejection_patience != 10:
            raise ValueError("Full RASI stops after ten consecutive rejected proposals")
        if self.verifier_z != 3.0:
            raise ValueError("Full RASI uses a three-anchor-standard-error verifier")
        _finite(self.synthetic_weight, "synthetic_weight", strict=True)
        if self.audit.require_certificate and self.rounds:
            penalty = 2 * self.audit.sigma_c * math.sqrt(math.log(2 * self.rounds / self.audit.delta) / self.audit.size)
            if self.audit.epsilon <= penalty:
                raise ValueError(f"epsilon must exceed the Note-16 uncertainty penalty {penalty:.8g}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RASIConfig:
        value = dict(value)
        for key, typ in (("model", ModelConfig), ("audit", AuditConfig), ("selection", SelectionConfig)):
            if key in value:
                value[key] = typ(**value[key])
        return cls(**value)


def preset_config(name: str) -> RASIConfig:
    """Return a documented starting configuration without experiment-specific tuning."""
    presets = {
        "linear-cpu": RASIConfig(),
        "linear-native": RASIConfig(model=ModelConfig(kind="linear", backend="native")),
        "gru-cpu": RASIConfig(rounds=10, model=ModelConfig(kind="gru", device="cpu")),
        "gru-cuda": RASIConfig(rounds=10, model=ModelConfig(kind="gru", device="cuda:0")),
    }
    try:
        return presets[name]
    except KeyError as exc:
        raise ValueError(f"Unknown preset {name!r}; choose one of {', '.join(sorted(presets))}") from exc
