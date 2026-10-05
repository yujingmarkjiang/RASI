"""Explicit factual trajectories, contrast estimands and leakage-safe partitions.

A parallel-trial contrast is an average of active predictions minus an average
of comparator predictions. It is never expanded into individual causal labels.
Missing observations are omitted; no missing outcome is converted to zero.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable
import json
import math
import numpy as np


def numbers(values: Iterable[float], name: str) -> tuple[float, ...]:
    out = tuple(float(x) for x in values)
    if not out or any(not math.isfinite(x) for x in out):
        raise ValueError(f"{name} must be a nonempty finite vector")
    return out


@dataclass(frozen=True)
class Query:
    id: str
    subject: str
    treatment: str
    times: tuple[float, ...]
    context: tuple[tuple[float, ...], ...] = ()
    stratum: str = "all"
    initial_value: float | None = None

    def __post_init__(self) -> None:
        if not all(isinstance(x, str) and x.strip() for x in (self.id, self.subject, self.treatment, self.stratum)):
            raise ValueError("query id, subject, treatment and stratum must be nonempty strings")
        object.__setattr__(self, "times", numbers(self.times, "times"))
        if any(b <= a for a, b in zip(self.times, self.times[1:])):
            raise ValueError(f"Query {self.id}: times must be strictly increasing")
        context = tuple(tuple(float(x) for x in row) for row in self.context)
        if not context:
            context = tuple(() for _ in self.times)
        if len(context) != len(self.times) or len({len(row) for row in context}) != 1:
            raise ValueError(f"Query {self.id}: context must have one equal-width row per time")
        if any(not math.isfinite(x) for row in context for x in row):
            raise ValueError(f"Query {self.id}: nonfinite context")
        object.__setattr__(self, "context", context)
        if self.initial_value is not None and not math.isfinite(self.initial_value):
            raise ValueError("initial_value must be finite or null")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Query:
        return cls(**d)


@dataclass(frozen=True)
class Trajectory:
    query: Query
    values: tuple[float, ...]
    weight: float = 1.0
    origin: str = "real"
    donors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", numbers(self.values, "trajectory values"))
        object.__setattr__(self, "donors", tuple(self.donors))
        if len(self.values) != len(self.query.times):
            raise ValueError(f"Trajectory {self.query.id}: values and times differ in length")
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("Trajectory weights must be finite and positive")
        if self.origin not in {"real", "synthetic"}:
            raise ValueError("Trajectory origin must be real or synthetic")
        if self.origin == "real" and self.donors:
            raise ValueError("A real observation cannot carry synthetic donor provenance")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Trajectory:
        d = dict(d)
        d["query"] = Query.from_dict(d["query"])
        return cls(**d)


PROVENANCE = {"randomised_crossover", "randomised_trial", "negative_control", "injected_effect", "quasi_randomised"}


@dataclass(frozen=True)
class Contrast:
    id: str
    active: tuple[Query, ...]
    comparator: tuple[Query, ...]
    values: tuple[float, ...]
    unit_id: str
    provenance: str
    scope: str = "individual"
    standard_errors: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "active", tuple(self.active))
        object.__setattr__(self, "comparator", tuple(self.comparator))
        object.__setattr__(self, "values", numbers(self.values, "contrast values"))
        if not self.id or not self.unit_id or not self.active:
            raise ValueError("A contrast requires id, unit_id and active queries")
        if self.provenance not in PROVENANCE:
            raise ValueError(f"Unrecognised causal provenance {self.provenance!r}")
        if self.scope not in {"individual", "stratum"}:
            raise ValueError("contrast.scope must be individual or stratum")
        if self.scope == "individual" and (len(self.active) != 1 or len(self.comparator) > 1):
            raise ValueError("An individual contrast has one active and at most one comparator")
        if self.scope == "individual" and self.comparator and self.active[0].subject != self.comparator[0].subject:
            raise ValueError("Within-patient contrasts must refer to the same subject")
        if self.provenance in {"randomised_trial", "quasi_randomised"} and self.scope != "stratum":
            raise ValueError("Parallel/quasi-randomised evidence identifies stratum means, not individual effects")
        queries = self.active + self.comparator
        if any(q.times != queries[0].times for q in queries) or len(self.values) != len(queries[0].times):
            raise ValueError(f"Contrast {self.id}: all predictions and values require identical times")
        if len({q.id for q in self.active}) != len(self.active) or len({q.id for q in self.comparator}) != len(self.comparator):
            raise ValueError("Duplicate query within a contrast arm")
        if self.standard_errors is not None:
            se = numbers(self.standard_errors, "standard_errors")
            if len(se) != len(self.values) or any(x < 0 for x in se):
                raise ValueError("Contrast standard errors must be nonnegative and match its times")
            object.__setattr__(self, "standard_errors", se)

    @property
    def queries(self) -> tuple[Query, ...]:
        return self.active + self.comparator

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Contrast:
        d = dict(d)
        d["active"] = tuple(Query.from_dict(q) for q in d["active"])
        d["comparator"] = tuple(Query.from_dict(q) for q in d.get("comparator", []))
        return cls(**d)


@dataclass(frozen=True)
class Target:
    id: str
    active: Query
    comparator: Query | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("target id must not be empty")
        if self.comparator is not None:
            if self.active.subject != self.comparator.subject or self.active.times != self.comparator.times:
                raise ValueError("Target treatment alternatives must share subject and scheduled times")
            if self.active.treatment == self.comparator.treatment:
                raise ValueError("Target arms must be different treatments")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Target:
        return cls(d["id"], Query.from_dict(d["active"]), Query.from_dict(d["comparator"]) if d.get("comparator") else None)


@dataclass(frozen=True)
class Anchor:
    id: str
    treatment: str
    comparator: str | None
    times: tuple[float, ...]
    values: tuple[float, ...]
    standard_errors: tuple[float, ...]
    provenance: str
    source: str
    population: tuple[Target, ...]
    scope: str = "stratum"
    stratum: str = "all"

    def __post_init__(self) -> None:
        for name in ("times", "values", "standard_errors"):
            object.__setattr__(self, name, numbers(getattr(self, name), f"anchor {name}"))
        object.__setattr__(self, "population", tuple(self.population))
        if not self.id or not self.source or self.provenance not in PROVENANCE:
            raise ValueError("Anchor needs an id, a source and valid evidence provenance")
        if self.scope not in {"individual", "stratum"} or not self.population:
            raise ValueError("Anchor needs an explicit individual or stratum population")
        if self.scope == "individual" and len(self.population) != 1:
            raise ValueError("An individual anchor must name exactly one target")
        if self.provenance in {"randomised_trial", "quasi_randomised"} and self.scope != "stratum":
            raise ValueError("A parallel-trial mean is not an individual treatment effect")
        if not (len(self.times) == len(self.values) == len(self.standard_errors)) or any(s < 0 for s in self.standard_errors):
            raise ValueError("Anchor standard errors must be nonnegative and aligned with values")
        if len({t.active.subject for t in self.population}) != len(self.population):
            raise ValueError("An anchor population must have at most one target per subject")
        for t in self.population:
            comp = t.comparator.treatment if t.comparator else None
            if (t.active.treatment, comp, t.active.times, t.active.stratum) != (self.treatment, self.comparator, self.times, self.stratum):
                raise ValueError(f"Anchor {self.id}: population does not match the named estimand")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Anchor:
        d = dict(d)
        d["population"] = tuple(Target.from_dict(t) for t in d["population"])
        return cls(**d)


@dataclass(frozen=True)
class TrainingView:
    """The complete data capability given to a proposal. No audit fields exist."""
    real: tuple[Trajectory, ...]
    contrasts: tuple[Contrast, ...]
    targets: tuple[Target, ...]
    anchors: tuple[Anchor, ...]
    validation: tuple[Trajectory, ...] = ()
    representation: str = "outcome"


@dataclass(frozen=True)
class Problem:
    training: TrainingView
    audit: tuple[Contrast, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        v = self.training
        if v.representation not in {"outcome", "contrast"}:
            raise ValueError("representation must be outcome or contrast")
        if not v.real or not v.contrasts or not self.audit:
            raise ValueError("Real training data, real training contrasts and a separate contrast audit are required")
        if any(x.origin != "real" for x in v.real + v.validation):
            raise ValueError("Input training/validation records must be real")
        for records in (v.real, v.validation):
            if len({x.query.id for x in records}) != len(records):
                raise ValueError("Duplicate real trajectory id")
        tr_ids = {x.query.id for x in v.real}
        val_ids = {x.query.id for x in v.validation}
        au_ids = {q.id for c in self.audit for q in c.queries}
        target_ids = {t.active.id for t in v.targets}
        if len(target_ids) != len(v.targets) or len({t.id for t in v.targets}) != len(v.targets):
            raise ValueError("Each synthetic target must name a distinct unobserved active query")
        if tr_ids & val_ids or (tr_ids | val_ids | target_ids) & au_ids or target_ids & (tr_ids | val_ids):
            raise ValueError("Training, validation, targets and audit trajectories must be disjoint")
        train_contrast_ids = {c.id for c in v.contrasts}
        if len(train_contrast_ids) != len(v.contrasts) or len({c.id for c in self.audit}) != len(self.audit):
            raise ValueError("Duplicate contrast id")
        if train_contrast_ids & {c.id for c in self.audit}:
            raise ValueError("Audit and training contrast IDs overlap")
        recorded = {o.query.id: np.asarray(o.values) for o in v.real}
        for c in v.contrasts:
            if not {q.id for q in c.queries} <= tr_ids:
                raise ValueError(f"Training contrast {c.id} references a non-training trajectory")
            active = np.mean([recorded[q.id] for q in c.active], axis=0)
            comparator = np.mean([recorded[q.id] for q in c.comparator], axis=0) if c.comparator else np.zeros_like(active)
            if not np.allclose(active - comparator, c.values, rtol=1e-8, atol=1e-8):
                raise ValueError(f"Training contrast {c.id} is not the recorded active-minus-comparator contrast; published external estimates belong in anchors")
        # Independent statistical units can contain several correlated pairs and time points.
        owners: dict[str, str] = {}
        for c in self.audit:
            for q in c.queries:
                previous = owners.setdefault(q.subject, c.unit_id)
                if previous != c.unit_id:
                    raise ValueError("Correlated audit pairs for one subject must share a unit_id")
        train_units = {c.unit_id for c in v.contrasts}
        if train_units & {c.unit_id for c in self.audit}:
            raise ValueError("Training and audit unit_id values overlap")
        proposal_queries = [x.query for x in v.real + v.validation]
        proposal_queries += [q for t in v.targets for q in ((t.active, t.comparator) if t.comparator else (t.active,))]
        proposal_queries += [q for a in v.anchors for t in a.population for q in ((t.active, t.comparator) if t.comparator else (t.active,))]
        if {q.id for q in proposal_queries} & au_ids:
            raise ValueError("An audit input was exposed through a target or an anchor population")
        # Identity changes must not be used to bypass ID-based exclusion.
        audit_keys = {(q.subject, q.treatment, t) for c in self.audit for q in c.queries for t in q.times}
        if any((q.subject, q.treatment, t) in audit_keys for q in proposal_queries for t in q.times):
            raise ValueError("Reserved audit trajectory reappears under another id")
        all_queries = proposal_queries + [q for c in v.contrasts + self.audit for q in c.queries]
        known: dict[str, Query] = {}
        for q in all_queries:
            if known.setdefault(q.id, q) != q:
                raise ValueError(f"Inconsistent definitions for query {q.id}")
        if len({len(q.context[0]) for q in all_queries}) != 1:
            raise ValueError("All context vectors must have the same width")
        for c in v.contrasts + self.audit:
            if v.representation == "outcome" and not c.comparator:
                raise ValueError("Outcome mode requires real comparator queries; zero is not a fabricated comparator")
            if v.representation == "contrast" and c.comparator:
                raise ValueError("Direct-contrast mode uses an algebraic zero reference, not a second outcome model")
        if v.representation == "outcome" and any(t.comparator is None for t in v.targets):
            raise ValueError("Outcome-mode targets require a comparator query")
        if v.representation == "contrast" and any(t.comparator is not None for t in v.targets):
            raise ValueError("Contrast-mode targets must have comparator=null")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Problem:
        allowed = {"schema_version", "representation", "training", "contrasts", "targets", "anchors", "validation", "audit", "metadata"}
        if set(d) - allowed:
            raise ValueError(f"Unknown dataset fields: {sorted(set(d) - allowed)}")
        if d.get("schema_version") != 1:
            raise ValueError("Unsupported dataset schema_version")
        view = TrainingView(
            tuple(Trajectory.from_dict(x) for x in d["training"]),
            tuple(Contrast.from_dict(x) for x in d["contrasts"]),
            tuple(Target.from_dict(x) for x in d.get("targets", [])),
            tuple(Anchor.from_dict(x) for x in d.get("anchors", [])),
            tuple(Trajectory.from_dict(x) for x in d.get("validation", [])),
            d.get("representation", "outcome"),
        )
        return cls(view, tuple(Contrast.from_dict(x) for x in d["audit"]), dict(d.get("metadata", {})))

    @classmethod
    def load(cls, path: str | Path) -> Problem:
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict[str, Any]:
        v = self.training
        return {"schema_version": 1, "representation": v.representation,
                "training": [asdict(x) for x in v.real], "contrasts": [asdict(x) for x in v.contrasts],
                "targets": [asdict(x) for x in v.targets], "anchors": [asdict(x) for x in v.anchors],
                "validation": [asdict(x) for x in v.validation], "audit": [asdict(x) for x in self.audit],
                "metadata": self.metadata}


def contrast_prediction(model: Any, c: Contrast) -> np.ndarray:
    active = np.mean([model.predict(q) for q in c.active], axis=0)
    comparator = np.mean([model.predict(q) for q in c.comparator], axis=0) if c.comparator else np.zeros_like(active)
    result = active - comparator
    if result.shape != (len(c.values),) or not np.all(np.isfinite(result)):
        raise ValueError("Invalid model contrast prediction")
    return result


def target_prediction(model: Any, t: Target) -> np.ndarray:
    active = model.predict(t.active)
    return active - model.predict(t.comparator) if t.comparator else active


def factual_mae(model: Any, observations: tuple[Trajectory, ...]) -> float | None:
    if not observations:
        return None
    errors = [np.abs(model.predict(o.query) - o.values) for o in observations]
    value = float(np.mean(np.concatenate(errors)))
    if not math.isfinite(value):
        raise ValueError("Nonfinite factual prediction")
    return value


def schema_queries(v: TrainingView) -> tuple[Query, ...]:
    """Only the non-audit input support is used to initialise an adapter."""
    qs = {o.query.id: o.query for o in v.real}
    for t in v.targets:
        qs[t.active.id] = t.active
        if t.comparator:
            qs[t.comparator.id] = t.comparator
    for a in v.anchors:
        for t in a.population:
            qs[t.active.id] = t.active
            if t.comparator:
                qs[t.comparator.id] = t.comparator
    return tuple(qs.values())
