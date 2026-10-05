"""Five-member disagreement allocation and averaged real-neighbour proposals."""
from __future__ import annotations
from dataclasses import dataclass, replace
from collections import Counter
from typing import Any
import numpy as np
from .data import Query, Trajectory, Target, TrainingView, target_prediction
from .models.base import FitRequest, ModelFactory, WorldModel


@dataclass(frozen=True)
class Allocation:
    counts: tuple[int, ...]
    disagreement: tuple[float, ...]
    probabilities: tuple[float, ...]
    total_budget: int


def allocate(factory: ModelFactory, view: TrainingView, current_data: tuple[Trajectory, ...],
             lam: float, rollouts: int, rng: np.random.Generator) -> Allocation:
    if not view.targets:
        return Allocation((), (), (), 0)
    subjects = sorted({o.query.subject for o in current_data})
    real_subjects = {o.query.subject for o in view.real}
    predictions = []
    for _ in range(5):
        # Cluster bootstrap supplies independent fits even for deterministic
        # linear models. Neural fits also use independent initialisation seeds.
        while True:
            weights = dict(Counter(rng.choice(subjects, size=len(subjects), replace=True).tolist()))
            if real_subjects & weights.keys():
                break
        member = factory(int(rng.integers(0, 2**31 - 1)))
        member.fit(FitRequest(current_data, view.contrasts, lam, view.validation, weights))
        predictions.append([target_prediction(member, t) for t in view.targets])
    variance = np.asarray([np.var(np.stack([member[j] for member in predictions]), axis=0, ddof=1).mean()
                           for j in range(len(view.targets))], dtype=float)
    if not np.all(np.isfinite(variance)) or np.any(variance < 0):
        raise FloatingPointError("Nonfinite ensemble contrast disagreement")
    total = rollouts * len(view.targets)
    # Multinomial resampling preserves E[m_j] proportional to v_j and the exact
    # total budget; zero allocations are valid. No artificial one-per-arm floor.
    probabilities = variance / variance.sum() if variance.sum() > 0 else np.full(len(variance), 1 / len(variance))
    counts = rng.multinomial(total, probabilities)
    return Allocation(tuple(int(x) for x in counts), tuple(float(x) for x in variance),
                      tuple(float(x) for x in probabilities), total)


def aligned_donor(donor: Trajectory, query: Query) -> Trajectory | None:
    """Select recorded scheduled times only; do not interpolate missing outcomes."""
    index = {t: j for j, t in enumerate(donor.query.times)}
    if not all(t in index for t in query.times):
        return None
    js = [index[t] for t in query.times]
    q = replace(donor.query, times=query.times, context=tuple(donor.query.context[j] for j in js))
    return replace(donor, query=q, values=tuple(donor.values[j] for j in js))


def nearest_mean(model: WorldModel, query: Query, real: tuple[Trajectory, ...], k: int) -> tuple[np.ndarray, tuple[str, ...]] | None:
    target_embedding = np.asarray(model.encode(query), dtype=float)
    if target_embedding.ndim != 1 or not np.all(np.isfinite(target_embedding)):
        raise ValueError("Model encode() must return a finite one-dimensional representation")
    candidates: dict[str, tuple[float, str, Trajectory]] = {}
    for row in real:
        if row.origin != "real":
            raise ValueError("Synthetic observations cannot enter the retrieval index")
        q = row.query
        if q.subject == query.subject or q.treatment != query.treatment or q.stratum != query.stratum:
            continue
        aligned = aligned_donor(row, query)
        if aligned is None:
            continue
        emb = np.asarray(model.encode(aligned.query), dtype=float)
        if emb.shape != target_embedding.shape or not np.all(np.isfinite(emb)):
            raise ValueError("Donor representations must be finite and have a consistent shape")
        distance = float(np.mean((emb - target_embedding) ** 2))
        candidate = (distance, q.id, aligned)
        previous = candidates.get(q.subject)
        if previous is None or candidate[:2] < previous[:2]:
            candidates[q.subject] = candidate
    ranked = sorted(candidates.values(), key=lambda x: (x[0], x[1]))
    if len(ranked) < k:
        return None
    neighbours = [x[2] for x in ranked[:k]]
    # Full RASI averages k real patients. The demonstration's single-donor draw,
    # sensitivity shift, and Gaussian fallback are intentionally not used here.
    return np.mean([row.values for row in neighbours], axis=0), tuple(row.query.id for row in neighbours)


@dataclass(frozen=True)
class Proposal:
    target: Target
    trajectory: Trajectory
    allocated: int


def retrieve(model: WorldModel, view: TrainingView, allocation: Allocation, k: int,
             synthetic_weight: float) -> tuple[tuple[Proposal, ...], dict[str, int]]:
    proposals = []
    skipped = {"zero_allocation": 0, "insufficient_real_donors": 0}
    for target, count in zip(view.targets, allocation.counts):
        if count == 0:
            skipped["zero_allocation"] += 1
            continue
        retrieved = nearest_mean(model, target.active, view.real, k)
        if retrieved is None:
            skipped["insufficient_real_donors"] += count
            continue
        values, donor_ids = retrieved
        # Identical deterministic kNN means are represented by multiplicity,
        # preserving targeted budget weights without materialising duplicate rows.
        row = Trajectory(target.active, tuple(float(x) for x in values), count * synthetic_weight, "synthetic", donor_ids)
        proposals.append(Proposal(target, row, count))
    return tuple(proposals), skipped


def verify(model: WorldModel, view: TrainingView, proposals: tuple[Proposal, ...], z: float = 3.0) -> tuple[tuple[Proposal, ...], dict[str, int]]:
    pending = {p.target.active.id: p for p in proposals}
    reasons = {"missing_anchor": 0, "anchor_inconsistent": 0}
    covered = {t.active.id for a in view.anchors for t in a.population}
    for qid in list(pending):
        if qid not in covered:
            reasons["missing_anchor"] += pending.pop(qid).allocated
    # Population anchors are checked on their defined population, not on one
    # patient's outcome and not on a rollout-count-weighted mean. Failing groups
    # are removed together. Recheck overlapping anchors after every removal.
    while pending:
        remove: set[str] = set()
        for anchor in view.anchors:
            affected = {t.active.id for t in anchor.population} & pending.keys()
            if not affected:
                continue
            contrasts = []
            for t in anchor.population:
                p = pending.get(t.active.id)
                active = np.asarray(p.trajectory.values) if p else model.predict(t.active)
                comparator = model.predict(t.comparator) if t.comparator else np.zeros_like(active)
                contrasts.append(active - comparator)
            implied = np.mean(contrasts, axis=0)
            deviation = np.abs(implied - anchor.values)
            tolerance = z * np.asarray(anchor.standard_errors)
            # Zero SE is an exact constraint, not a licence to use residual SD.
            if not np.all(np.isfinite(implied)) or np.any(deviation > tolerance):
                remove.update(affected)
        if not remove:
            break
        for qid in remove:
            reasons["anchor_inconsistent"] += pending.pop(qid).allocated
    return tuple(p for p in proposals if p.target.active.id in pending), reasons
