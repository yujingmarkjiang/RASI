"""Select lambda and retrieval k once, using only visible training contrasts."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from collections import defaultdict
import hashlib
import json
import numpy as np
from .config import RASIConfig
from .data import TrainingView, contrast_prediction, schema_queries
from .models import FitRequest, make_factory
from .retrieval import nearest_mean


@dataclass(frozen=True)
class Selection:
    contrast_weight: float
    neighbours: int
    source: str
    training_sha256: str
    scores: dict

    def __post_init__(self) -> None:
        if not np.isfinite(self.contrast_weight) or self.contrast_weight <= 0 or self.neighbours < 1:
            raise ValueError("Full RASI needs positive lambda and a positive retrieval k")
        if not self.source or not self.training_sha256:
            raise ValueError("Frozen settings require recorded selection provenance")

    def to_dict(self) -> dict:
        return asdict(self)


def training_digest(view: TrainingView) -> str:
    body = {"real": [asdict(x) for x in view.real], "contrasts": [asdict(x) for x in view.contrasts]}
    return hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest()


def select(view: TrainingView, config: RASIConfig) -> Selection:
    groups: dict[str, set[str]] = defaultdict(set)
    for c in view.contrasts:
        groups[c.unit_id].update(q.subject for q in c.queries)
    if len(groups) < 2:
        raise ValueError("Visible-pair CV needs at least two independent units, or a previously frozen selection file")
    units = sorted(groups)
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 101]))
    rng.shuffle(units)
    nfolds = min(config.selection.folds, len(units))
    lambda_errors = {lam: [] for lam in config.selection.lambda_grid}
    k_errors = {k: [] for k in config.selection.k_grid}
    fold_records = []
    queries = schema_queries(view)
    for fold in range(nfolds):
        held_units = set(units[fold::nfolds])
        held_subjects = set().union(*(groups[u] for u in held_units))
        real = tuple(o for o in view.real if o.query.subject not in held_subjects)
        pairs = tuple(c for c in view.contrasts if not any(q.subject in held_subjects for q in c.queries))
        held = tuple(c for c in view.contrasts if c.unit_id in held_units)
        if not real or not pairs or not held:
            raise ValueError("CV folds have no training support; provide more disjoint visible units or frozen settings")
        factory = make_factory(config.model, queries, real)
        models = {}
        for lam in config.selection.lambda_grid:
            model = factory(config.seed + 1009 * (fold + 1))
            model.fit(FitRequest(real, pairs, lam, view.validation))
            models[lam] = model
            for c in held:
                lambda_errors[lam].extend(np.abs(contrast_prediction(model, c) - c.values).tolist())
        fold_records.append((real, held, models))
    best_lambda = min(config.selection.lambda_grid, key=lambda lam: (np.mean(lambda_errors[lam]), lam))
    coverage = {k: 0 for k in config.selection.k_grid}
    total_held = sum(len(held) for _, held, _ in fold_records)
    for real, held, models in fold_records:
        model = models[best_lambda]
        for k in config.selection.k_grid:
            for c in held:
                active = [nearest_mean(model, q, real, k) for q in c.active]
                if any(x is None for x in active):
                    continue
                prediction = np.mean([x[0] for x in active], axis=0)
                if c.comparator:
                    prediction -= np.mean([model.predict(q) for q in c.comparator], axis=0)
                k_errors[k].extend(np.abs(prediction - c.values).tolist())
                coverage[k] += 1
    # Comparing k on different supported subsets would give a biased selection.
    valid_k = [k for k in config.selection.k_grid if coverage[k] == total_held]
    if not valid_k:
        raise ValueError("No retrieval k has complete visible-fold donor support. Supply additional real donors or a justified frozen selection")
    best_k = min(valid_k, key=lambda k: (np.mean(k_errors[k]), k))
    return Selection(float(best_lambda), int(best_k), "visible-unit cross-validation, frozen before round 0", training_digest(view),
                     {"folds": nfolds, "lambda_mae": {str(k): float(np.mean(v)) for k, v in lambda_errors.items()},
                      "retrieval_mae": {str(k): (float(np.mean(v)) if v and coverage[k] == total_held else None) for k, v in k_errors.items()},
                      "retrieval_coverage": {str(k): n for k, n in coverage.items()}, "held_pairs": total_held})
