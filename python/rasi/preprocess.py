"""Real-data importers; no artificial masks, simulations or experimental sweeps."""
from __future__ import annotations
from dataclasses import replace
from collections import defaultdict
from pathlib import Path
from typing import Any
import csv
import math
import numpy as np
from .data import Query, Trajectory, Contrast, Target, Anchor, TrainingView, Problem


def read_csv(path: str | Path, required: set[str]) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError(f"{path}: required columns are {', '.join(sorted(required))}")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("Duplicate CSV column name")
        rows = list(reader)
    if not rows or any(None in row or any(row.get(k) in (None, "") for k in required) for row in rows):
        raise ValueError("CSV contains no rows, ragged rows or missing required values")
    return rows


def finite(value: str, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid numeric {name}: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite {name}")
    return result


def _reserve(subjects: set[str], protect: set[str], audit_subjects: int, validation_subjects: int, seed: int) -> tuple[set[str], set[str]]:
    if audit_subjects < 2 or validation_subjects < 0:
        raise ValueError("Reserve at least two audit subjects and a nonnegative number for factual validation")
    eligible = sorted(subjects - protect)
    if len(eligible) < audit_subjects + validation_subjects or len(subjects) - audit_subjects - validation_subjects < 3:
        raise ValueError("Not enough subjects for the requested audit/validation reserves and visible training support")
    rng = np.random.default_rng(seed)
    rng.shuffle(eligible)
    return set(eligible[:audit_subjects]), set(eligible[audit_subjects:audit_subjects + validation_subjects])


def derived_crossover_anchors(real: tuple[Trajectory, ...], contrasts: tuple[Contrast, ...],
                              targets: tuple[Target, ...], source: str) -> tuple[Anchor, ...]:
    groups: dict[tuple, list[Contrast]] = defaultdict(list)
    for c in contrasts:
        if c.scope != "individual" or c.provenance != "randomised_crossover":
            continue
        q = c.active[0]
        comp = c.comparator[0].treatment if c.comparator else None
        groups[(q.treatment, comp, q.stratum, q.times)].append(c)
    result = []
    for key, pairs in groups.items():
        treatment, comparator, stratum, times = key
        by_subject: dict[str, list[np.ndarray]] = defaultdict(list)
        for c in pairs:
            by_subject[c.active[0].subject].append(np.asarray(c.values))
        if len(by_subject) < 2:
            continue
        values = np.stack([np.mean(rows, axis=0) for rows in by_subject.values()])
        mean, se = values.mean(axis=0), values.std(axis=0, ddof=1) / math.sqrt(len(values))
        population = {c.active[0].subject: Target("population:" + c.id, c.active[0], c.comparator[0] if c.comparator else None) for c in pairs}
        for t in targets:
            comp = t.comparator.treatment if t.comparator else None
            if (t.active.treatment, comp, t.active.stratum, t.active.times) == key:
                population[t.active.subject] = t
        result.append(Anchor("visible:" + treatment + ":" + stratum + ":" + str(times), treatment, comparator, times,
                             tuple(float(x) for x in mean), tuple(float(x) for x in se), "randomised_crossover",
                             source + "; mean and between-subject SE from visible training contrasts only",
                             tuple(population.values()), "stratum", stratum))
    return tuple(result)


def _assemble(records: list[Trajectory], pairs: list[Contrast], targets: list[Target], audit_subjects: int,
              validation_subjects: int, seed: int, representation: str, metadata: dict[str, Any]) -> Problem:
    subjects = {o.query.subject for o in records}
    protect = {t.active.subject for t in targets} & subjects
    audit, validation = _reserve(subjects, protect, audit_subjects, validation_subjects, seed)
    train = tuple(o for o in records if o.query.subject not in audit | validation)
    train_ids = {o.query.id for o in train}
    train_pairs = tuple(c for c in pairs if {q.id for q in c.queries} <= train_ids)
    audit_pairs = tuple(c for c in pairs if all(q.subject in audit for q in c.queries))
    val = tuple(o for o in records if o.query.subject in validation)
    targets_t = tuple(targets)
    anchors = derived_crossover_anchors(train, train_pairs, targets_t, metadata["source"])
    metadata = dict(metadata, split_seed=seed, audit_subjects=len(audit), validation_subjects=len(validation),
                    training_subjects=len(subjects - audit - validation), artificial_hidden_masks=False,
                    anchor_transport_assumption="Visible randomised crossover population applies to the explicitly named target population")
    return Problem(TrainingView(train, train_pairs, targets_t, anchors, val, representation), audit_pairs, metadata)


def prepare_contrasts(path: str | Path, audit_subjects: int = 12, validation_subjects: int = 2,
                      seed: int = 20261001, targets_path: str | Path | None = None) -> Problem:
    """Import the supplied contrast table without pretending it contains factual QT.

    Concentration is NOT automatically used: concentrations of a missing arm may
    be unavailable at decision time. Supply declared context through the JSON API
    for a concentration-conditioned model. Baseline is a pre-treatment covariate.
    """
    rows = read_csv(path, {"participant", "drug", "time", "contrast"})
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["participant"], row["drug"])].append(row)
    has_baseline = "baseline" in rows[0]
    if has_baseline and any(not r.get("baseline") for r in rows):
        raise ValueError("Missing baseline covariates must be handled explicitly, not filled with zero")
    records, pairs = [], []
    for (subject, treatment), cells in sorted(groups.items()):
        cells = sorted(cells, key=lambda r: finite(r["time"], "time"))
        times = tuple(finite(r["time"], "time") for r in cells)
        context = tuple((finite(r["baseline"], "baseline"),) if has_baseline else () for r in cells)
        strata = {r.get("stratum", "all") for r in cells}
        if len(strata) != 1:
            raise ValueError("A trajectory must have one declared pre-treatment stratum")
        q = Query(subject + "::" + treatment, subject, treatment, times, context, strata.pop())
        values = tuple(finite(r["contrast"], "contrast") for r in cells)
        records.append(Trajectory(q, values))
        pairs.append(Contrast("contrast:" + q.id, (q,), (), values, subject, "randomised_crossover"))
    targets = []
    if targets_path is not None:
        target_rows = read_csv(targets_path, {"participant", "drug", "time"})
        if any(k.lower() in {"contrast", "outcome", "y", "qt", "rr"} for k in target_rows[0]):
            raise ValueError("Target files must contain inputs only, never hidden outcomes")
        target_groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in target_rows:
            target_groups[(row["participant"], row["drug"])].append(row)
        for (subject, treatment), cells in sorted(target_groups.items()):
            if (subject, treatment) in groups:
                raise ValueError("Target is already observed; preprocessing does not create hidden-arm experiments")
            cells = sorted(cells, key=lambda r: finite(r["time"], "time"))
            context = tuple((finite(r.get("baseline", ""), "target baseline"),) if has_baseline else () for r in cells)
            strata = {r.get("stratum", "all") for r in cells}
            if len(strata) != 1:
                raise ValueError("Target stratum changes within a trajectory")
            q = Query(subject + "::" + treatment, subject, treatment,
                      tuple(finite(r["time"], "target time") for r in cells), context, strata.pop())
            targets.append(Target("target:" + q.id, q))
    else:
        subjects = sorted({o.query.subject for o in records})
        treatments = sorted({o.query.treatment for o in records})
        for subject in subjects:
            own = [o for o in records if o.query.subject == subject]
            for treatment in treatments:
                if (subject, treatment) in groups:
                    continue
                schedules = {o.query.times for o in records if o.query.treatment == treatment}
                strata = {o.query.stratum for o in own}
                if len(schedules) != 1 or len(strata) != 1:
                    raise ValueError("Missing arm has ambiguous schedule/stratum; supply an input-only targets CSV")
                times = next(iter(schedules))
                # A subject's known pre-treatment baselines are inputs, not hidden outcomes.
                baseline = float(np.mean([r[0] for o in own for r in o.query.context])) if has_baseline else None
                context = tuple((baseline,) if has_baseline else () for _ in times)
                q = Query(subject + "::" + treatment, subject, treatment, times, context, next(iter(strata)))
                targets.append(Target("target:" + q.id, q))
    return _assemble(records, pairs, targets, audit_subjects, validation_subjects, seed, "contrast",
                     {"source": str(Path(path).name), "input_kind": "precomputed_placebo_corrected_contrasts",
                      "factual_outcomes_available": False, "reference": "algebraic zero in contrast coordinates, not a fabricated placebo outcome",
                      "measurement_floor": None, "measurement_floor_reason": "Replicate ECG measurements were not supplied",
                      "evidence_status": "Crossover provenance is supplied by the manuscript, not independently verified by this importer"})


def prepare_qt(path: str | Path, placebo: str = "Placebo", audit_subjects: int = 12,
               validation_subjects: int = 2, seed: int = 20261001) -> Problem:
    """QT/(RR/1000)^(1/3), replicate means, period baselines and placebo correction.

    Required CSV: participant, period, drug, time, QT, RR (QT and RR in ms).
    Time <= 0 denotes pre-dose measurements. A single placebo period per subject
    and at most one period per drug are required; ambiguous matching is refused.
    """
    rows = read_csv(path, {"participant", "period", "drug", "time"})
    if not {"QT", "RR"} <= set(rows[0]):
        raise ValueError("The raw ECG importer requires QT and RR columns in milliseconds")
    cells: dict[tuple[str, str, str, float], list[float]] = defaultdict(list)
    excluded = 0
    period_drug: dict[tuple[str, str], str] = {}
    for r in rows:
        try:
            qt, rr = finite(r["QT"], "QT"), finite(r["RR"], "RR")
        except ValueError:
            excluded += 1
            continue
        if qt <= 0 or rr <= 0:
            excluded += 1
            continue
        key = (r["participant"], r["period"])
        if period_drug.setdefault(key, r["drug"]) != r["drug"]:
            raise ValueError("One crossover period cannot have multiple treatment labels")
        time = finite(r["time"], "time")
        cells[(r["participant"], r["period"], r["drug"], time)].append(qt / (rr / 1000.0) ** (1.0 / 3.0))
    periods: dict[tuple[str, str, str], dict[float, tuple[float, float | None]]] = defaultdict(dict)
    for (subject, period, drug, t), values in cells.items():
        variance = float(np.var(values, ddof=1) / len(values)) if len(values) >= 2 else None
        periods[(subject, period, drug)][t] = (float(np.mean(values)), variance)
    by_subject: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for subject, period, drug in periods:
        by_subject[subject].append((period, drug))
    records, pairs, floors = [], [], []
    observed: dict[tuple[str, str], Query] = {}
    for subject, exposures in sorted(by_subject.items()):
        controls = [(p, d) for p, d in exposures if d.casefold() == placebo.casefold()]
        if len(controls) != 1 or len({d for _, d in exposures}) != len(exposures):
            raise ValueError(f"Subject {subject}: supply an unambiguous one-period-per-treatment crossover with one placebo")
        summary = {}
        for period, drug in exposures:
            means = periods[(subject, period, drug)]
            predose = [v for t, v in means.items() if t <= 0]
            if not predose:
                raise ValueError(f"Subject {subject}, period {period}: no pre-dose baseline")
            baseline = float(np.mean([m for m, _ in predose]))
            baseline_variance = sum(v for _, v in predose) / len(predose) ** 2 if all(v is not None for _, v in predose) else None
            summary[drug] = (period, means, baseline, baseline_variance)
        schedules = [set(t for t in data[1] if t > 0) for data in summary.values()]
        times = tuple(sorted(set.intersection(*schedules)))
        if not times:
            raise ValueError(f"Subject {subject}: no shared post-dose schedule")
        per_subject = {}
        for drug, (period, means, baseline, base_var) in summary.items():
            q = Query(subject + "::" + period + "::" + drug, subject, drug, times,
                      tuple((baseline,) for _ in times), "all", 0.0)
            values = tuple(means[t][0] - baseline for t in times)
            row = Trajectory(q, values)
            records.append(row); per_subject[drug] = row; observed[(subject, drug)] = q
        control_drug = controls[0][1]
        control = per_subject[control_drug]
        for drug, row in per_subject.items():
            if drug == control_drug:
                continue
            values = tuple(a - b for a, b in zip(row.values, control.values))
            active_summary, control_summary = summary[drug], summary[control_drug]
            variances = []
            for t in times:
                terms = (active_summary[1][t][1], active_summary[3], control_summary[1][t][1], control_summary[3])
                variance = sum(terms) if all(x is not None for x in terms) else None
                variances.append(variance)
                if variance is not None:
                    floors.append(math.sqrt(2 / math.pi) * math.sqrt(variance))
            se = tuple(math.sqrt(x) for x in variances) if all(x is not None for x in variances) else None
            pairs.append(Contrast("contrast:" + row.query.id, (row.query,), (control.query,), values, subject,
                                  "randomised_crossover", "individual", se))
    treatments = sorted({q.treatment for q in observed.values() if q.treatment.casefold() != placebo.casefold()})
    targets = []
    for subject in sorted(by_subject):
        controls = [q for (s, d), q in observed.items() if s == subject and d.casefold() == placebo.casefold()]
        control = controls[0]
        for drug in treatments:
            if (subject, drug) not in observed:
                q = replace(control, id=subject + "::unobserved::" + drug, treatment=drug)
                targets.append(Target("target:" + q.id, q, control))
    return _assemble(records, pairs, targets, audit_subjects, validation_subjects, seed, "outcome",
                     {"source": str(Path(path).name), "input_kind": "replicate_QT_RR", "factual_outcomes_available": True,
                      "QT_and_RR_units": "ms", "reference": "period-centred QTcF(active) minus period-centred QTcF(placebo)",
                      "excluded_invalid_recordings": excluded, "measurement_floor": float(np.mean(floors)) if floors else None,
                      "measurement_floor_observations": len(floors), "measurement_floor_complete": len(floors) == sum(len(c.values) for c in pairs),
                      "evidence_status": "Randomisation, washout and absence of carry-over remain declared design assumptions"})
