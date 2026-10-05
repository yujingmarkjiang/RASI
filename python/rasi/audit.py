"""Reserved contrast auditing and the conditional Supplementary Note-16 bound.

The main-text gate uses the candidate audit MAE's standard error. The paired
improvement SE is also reported, but is not silently substituted for that rule.
A standard-error gate is not advertised as a high-probability guarantee.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from collections import defaultdict
import hashlib
import math
from typing import Any
import numpy as np
from .config import AuditConfig
from .data import Contrast, contrast_prediction
from .models.base import WorldModel


class AuditExhausted(RuntimeError):
    pass


@dataclass(frozen=True)
class Decision:
    accepted: bool
    reason: str
    incumbent_error: float | None
    candidate_error: float | None
    improvement: float | None
    candidate_standard_error: float | None
    improvement_standard_error: float | None
    margin: float
    units: int
    pairs: int
    subset_sha256: str
    maximum_reuse: int
    uncertainty_penalty: float | None
    conditional_certificate: bool
    certified_step: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AuditGate:
    """Only this component owns the audit outcomes; proposals receive no reference."""
    def __init__(self, pairs: tuple[Contrast, ...], config: AuditConfig, rounds: int, seed: int):
        self.config, self.rounds = config, rounds
        grouped: dict[str, list[Contrast]] = defaultdict(list)
        for pair in pairs:
            grouped[pair.unit_id].append(pair)
        if len(grouped) < config.size:
            raise ValueError(f"Audit requires {config.size} independent units but only {len(grouped)} are reserved")
        self._pairs = {k: tuple(v) for k, v in grouped.items()}
        self._units = sorted(grouped)
        self._sizes = {u: sum(len(c.values) for c in cs) for u, cs in self._pairs.items()}
        if config.require_certificate and len(set(self._sizes.values())) != 1:
            raise ValueError("The unweighted Note-16 formula requires equally weighted independent audit units; use empirical cluster SE for unequal units")
        self._rng = np.random.default_rng(seed)
        self._uses = {u: 0 for u in self._units}
        self._fixed: list[str] | None = None
        self._evaluations = 0
        self._certified_gain = 0.0
        self._accepted = 0

    def _draw(self) -> list[str]:
        if self.config.mode == "held_out" and self._evaluations:
            raise AuditExhausted("The held-out audit is single-use")
        if self.config.mode == "fixed" and self._fixed is not None:
            chosen = list(self._fixed)
        else:
            population = self._units if self.config.allow_reuse else [u for u in self._units if self._uses[u] == 0]
            if len(population) < self.config.size:
                raise AuditExhausted("Independent audit pool exhausted; no automatic reuse or undersized gate")
            chosen = self._rng.choice(population, self.config.size, replace=False).tolist()
            if self.config.mode == "fixed":
                self._fixed = list(chosen)
        for u in chosen:
            self._uses[u] += 1
        self._evaluations += 1
        return chosen

    def _errors(self, model: WorldModel, units: list[str]) -> np.ndarray:
        # Multiple times and multiple arms within one patient do not become
        # independent observations merely by flattening them into a long array.
        result = []
        for u in units:
            errors = [float(np.sum(np.abs(contrast_prediction(model, c) - c.values))) for c in self._pairs[u]]
            result.append(float(np.sum(errors)))
        values = np.asarray(result)
        if not np.all(np.isfinite(values)):
            raise ValueError("Nonfinite audit errors")
        return values

    def consider(self, incumbent: WorldModel, candidate: WorldModel) -> Decision:
        """Call only after the proposal is fitted/frozen. Both score the SAME draw."""
        chosen = self._draw()
        subset_hash = hashlib.sha256("\0".join(sorted(chosen)).encode()).hexdigest()
        n = len(chosen)
        reuse = max(self._uses[u] for u in chosen)
        pair_count = sum(len(self._pairs[u]) for u in chosen)
        penalty = None
        if self.config.sigma_c is not None:
            penalty = 2 * self.config.sigma_c * math.sqrt(math.log(2 * max(1, self.rounds) / self.config.delta) / n)
        certificate = (self.config.require_certificate and self.config.independent_units_asserted and
                       self.config.margin == "fixed" and self.config.mode == "rotated" and reuse == 1 and
                       not self.config.allow_reuse and penalty is not None and self.config.epsilon > penalty)
        # An invalid incumbent is fatal. An invalid proposal cannot be accepted.
        before = self._errors(incumbent, chosen)
        sizes = np.asarray([self._sizes[u] for u in chosen], dtype=float)
        mass = float(sizes.sum())
        before_mean = float(before.sum() / mass)
        try:
            after = self._errors(candidate, chosen)
        except (ValueError, FloatingPointError, RuntimeError):
            return Decision(False, "invalid_candidate_prediction", before_mean, None, None, None, None,
                            self.config.epsilon, n, pair_count, subset_hash, reuse, penalty, False, None)
        gain = before - after
        after_mean = float(after.sum() / mass)
        improvement = float(gain.sum() / mass)
        # Observation-weighted MAE (main Methods), with a cluster-robust ratio
        # standard error; correlated time points do not inflate sample size.
        se = float(math.sqrt(n / (n - 1) * np.sum((after - after_mean * sizes) ** 2)) / mass)
        gain_se = float(math.sqrt(n / (n - 1) * np.sum((gain - improvement * sizes) ** 2)) / mass)
        margin = max(self.config.epsilon, se) if self.config.margin == "standard_error" else self.config.epsilon
        accepted = improvement > margin if self.config.margin == "standard_error" else improvement >= margin
        certified_step = float(self.config.epsilon - penalty) if certificate else None
        if accepted:
            self._accepted += 1
            if certificate:
                self._certified_gain += certified_step
        return Decision(bool(accepted), "improved_beyond_margin" if accepted else "insufficient_contrast_improvement",
                        before_mean, after_mean, improvement, se, gain_se, float(margin), n,
                        pair_count, subset_hash, reuse, penalty, bool(certificate), certified_step)

    def certificate_summary(self) -> dict[str, Any]:
        conditional = bool(self.config.require_certificate)
        return {"conditional_on_declared_assumptions": conditional,
                "certified_error_upper_bound_relative_to_base": -self._certified_gain if conditional else None,
                "accepted_audit_evaluations": self._accepted,
                "audit_evaluations": self._evaluations,
                "maximum_unit_uses": max(self._uses.values(), default=0),
                "sigma_c": self.config.sigma_c,
                "delta": self.config.delta,
                "assumptions": ["independent reserved audit units", "fresh audits independent of adaptive proposals",
                                "externally justified sub-Gaussian error parameter", "fixed predeclared epsilon"] if conditional else [],
                "warning": None if conditional else "Empirical acceptance only; a reused holdout or standard-error margin does not establish Note 16"}

    def state_dict(self) -> dict[str, Any]:
        return {"uses": dict(self._uses), "fixed": self._fixed, "evaluations": self._evaluations,
                "accepted": self._accepted, "certified_gain": self._certified_gain, "rng": self._rng.bit_generator.state}

    def restore(self, state: dict[str, Any]) -> None:
        if set(state["uses"]) != set(self._units):
            raise ValueError("Resume audit units differ from the original reserved pool")
        uses = dict(state["uses"])
        if any(not isinstance(v, int) or v < 0 for v in uses.values()):
            raise ValueError("Invalid audit reuse counters")
        self._uses = uses
        self._fixed = state["fixed"]
        self._evaluations, self._accepted = int(state["evaluations"]), int(state["accepted"])
        self._certified_gain = float(state["certified_gain"])
        self._rng.bit_generator.state = state["rng"]
