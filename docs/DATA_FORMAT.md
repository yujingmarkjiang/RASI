# Data and estimand contract

The CLI reads UTF-8 JSON with `schema_version: 1`. Run `rasi validate --data PATH` before training. Python callers can construct the frozen dataclasses in `rasi.data`; the same `Problem` validation applies. The canonical serialization is `Problem.to_dict()`.

## Top-level document

| Field | Meaning |
|---|---|
| `schema_version` | Integer `1`. |
| `representation` | `"outcome"` for actual outcomes; `"contrast"` only for already corrected contrast targets. |
| `training` | Nonempty list of real `Trajectory` objects. Never audit rows or generated labels. |
| `contrasts` | Nonempty list of visible recorded `Contrast` objects. |
| `targets` | Input-only `Target` objects for genuinely unobserved arms; may be empty. |
| `anchors` | `Anchor` objects derived from visible randomized evidence or external, appropriately matched trials. Uncovered proposals are rejected. |
| `validation` | Separate visible `Trajectory` objects for early stopping; required by the GRU. |
| `audit` | Nonempty reserved `Contrast` objects. These are not supplied to the fitting, retrieval or proposal interfaces. |
| `metadata` | Optional JSON object for design assumptions, units, preprocessing and provenance. |

Arrays are required where listed; `targets`, `anchors` and `validation` default to empty. Unknown top-level fields are rejected. Do not put protected labels in `metadata`, contexts, identifiers, anchor estimates or target files. The program checks structural separation, not whether a caller has disguised an outcome as a covariate.

## Query: only information available before prediction

A `Query` has:

- `id`: globally consistent trajectory identifier; `subject`: patient/independent sampling-cluster identifier; `treatment`: treatment name. All are nonempty strings.
- `times`: a nonempty, finite, strictly increasing numeric vector in a declared common time unit.
- `context`: one equal-width numeric row per time, containing only available covariates/intervention inputs. Omit it for zero context features. All queries in one problem must have the same feature width.
- `stratum`: predeclared treatment-effect stratum, default `"all"`.
- `initial_value`: optional observed initial state, or `null`. It is an input, not the first hidden outcome. The raw QT importer uses zero for a period-centred pre-dose state.

A query describes a complete scheduled prediction, not an outcome. Repeated appearances of the same ID must have identical definitions. In particular, do not vary the context of a query between the training contrast and its factual trajectory.

The built-in linear model interpolates its learned time basis inside the known schedule and rejects extrapolation. Retrieval does not interpolate recorded outcomes: a donor must have all requested scheduled times. The GRU supports explicit scheduled times as inputs; this does not establish its accuracy outside the training distribution.

## Trajectory: a recorded outcome or accepted retrieved target

A `Trajectory` contains `query`, `values`, optional positive `weight` (default `1.0`), `origin` (default `"real"`) and `donors` (default empty). The finite `values` vector has exactly one value per query time.

All input training/validation observations must be real. The controller creates `origin: "synthetic"` observations internally, with donor provenance. A repeated allocation to the same deterministic neighbour mean is represented by its multiplicity in the observation weight, not by unreported Gaussian perturbations. All real observations remain in the training set under both recursion regimes.

Missing outcomes are absent observations, never zero-valued observations. Omit unsupported schedules or supply a properly defined shorter query and matching estimand. Treatment/comparator times within a contrast must match exactly.

## Contrast: the recorded causal estimand

Required fields: `id`, `active`, `comparator`, `values`, `unit_id` and `provenance`. `active` and `comparator` contain full Query objects, not lookup strings. Optional fields are `scope` (default `"individual"`) and `standard_errors` (nonnegative, aligned vector or `null`).

The prediction is

```
mean(model.predict(q) for q in active)
    - mean(model.predict(q) for q in comparator).
```

The averages are over patients, separately within each arm. An individual crossover contrast has one active query and one comparator query for the same patient. A parallel-trial contrast is a stratum mean, with separate active and comparator populations. Its members are not assigned the trial's average effect as patient-specific labels.

Allowed provenance strings are `randomised_crossover`, `randomised_trial`, `quasi_randomised`, `negative_control` and `injected_effect`. The latter is an input contract, not a generator or an experiment implementation. Parallel and quasi-randomised trial provenance requires `scope: "stratum"`. Provenance labels do not verify a study's actual design.

In **outcome mode**, both arms are required. Every visible training contrast must refer to real training trajectories, and `values` must equal their recorded mean active-minus-comparator difference. Published external effects belong in `anchors`, not in invented factual records.

In **contrast mode**, `comparator` is empty because `values` already describe a contrast. The zero reference is algebraic, not a fictitious placebo patient. Factual QT reconstruction and a factual non-degradation metric are unavailable in this mode.

`unit_id` is the independent audit/CV sampling unit. All contrasts sharing a patient must share an audit unit. Keep repeated times and effects sharing one comparator patient in the same unit; they are not independent observations. Nested clustering or overlapping trial populations may require a coarser unit, which the data provider must specify. Main audit MAE is observation-weighted; its uncertainty is clustered by these units.

## Target: unobserved treatment alternatives

A `Target` contains `id`, `active` and `comparator`. Each arm is a Query; `comparator` must be present in outcome mode and must be `null`/omitted in contrast mode. Outcome alternatives share the same subject and scheduled times but have different treatment names. The active query must not already be a real training or validation observation. The comparator can be a known real query. Targets contain no labels.

## Anchor: an estimate with a declared population and uncertainty

An `Anchor` contains:

| Field | Requirement |
|---|---|
| `id`, `source` | Nonempty identifiers and a traceable study/estimate source. |
| `treatment`, `comparator` | The named treatment contrast; comparator is `null` in contrast coordinates. |
| `times`, `values`, `standard_errors` | Equal-length finite vectors; SEs are nonnegative. |
| `provenance` | One of the evidence types above. |
| `population` | Nonempty list of Target objects defining the patients over whom to check the estimate. At most one target per patient. |
| `scope`, `stratum` | `"stratum"` and `"all"` by default; individual scope permits exactly one named patient. |

Every population member must match the anchor's treatment, comparator, stratum and times. The anchor estimate and its SE must be obtained without reserved audit outcomes. Trial eligibility, outcome scale, follow-up time and target-population transportability must be justified by the data provider. A published overall effect is not automatically an eligible subgroup effect.

The verifier forms the anchor-population mean using proposed active outcomes where available and incumbent predictions elsewhere, including comparator predictions. It checks every time against the recorded estimate ± three supplied anchor SEs. A zero SE gives a zero-width interval; there is no invented minimum uncertainty. A failed population check rejects all contributing proposals. Individual anchors check the named patient's proposed contrast. After removals, overlapping checks are reevaluated.

## Partition and leakage checks

The loader checks real/validation/target/audit query separation, consistent query definitions, visible-contrast consistency and matching estimands. Audit trajectories cannot reappear in proposal inputs under a new identifier with the same `(subject, treatment, time)`. No audit targets are included in retrieval indices or feature-schema initialization. Unit/subject overlap is handled conservatively; full-audit subject exclusion is additionally required for conditional-certificate mode.

For the strongest separation, use entirely different subjects for training, validation and audit. The CSV importers do so, keeping patients with naturally absent target arms outside audit and validation. With a caller-defined partial-arm split, the structural checker cannot establish causal independence or defend against arbitrary executable proposal code reading files. See `ADAPTERS.md` for the trust boundary.

## CSV importers

### Precomputed crossover contrasts

Required columns: `participant,drug,time,contrast`; optional `baseline` and `stratum`. One record per patient/drug/time is allowed. All supplied values must be finite. Other columns such as post-treatment concentration are not silently used as target covariates.

An optional input-only target CSV has `participant,drug,time` and, when used by training, `baseline`; `stratum` is optional. Columns containing `contrast`, `outcome`, `y`, `QT` or `RR` are rejected. Do not use this path to hide known arms for a benchmark: observed targets are rejected. Without that file, the importer discovers genuinely absent arms and uses known pre-treatment baselines, requiring an unambiguous schedule/stratum.

### Raw replicate ECGs

Required columns: `participant,period,drug,time,QT,RR`. QT and RR are milliseconds; time is a consistent scheduled unit, with `time <= 0` pre-dose. Each row is one recording replicate. An invalid/nonpositive/missing QT or RR is excluded and counted. Identifiers and time must be valid.

For each replicate, the importer computes `QTcF = QT / (RR/1000)^(1/3)`, then averages replicates at each nominal time. The baseline is the mean of available pre-dose nominal-time means within that period. It subtracts this period baseline from factual trajectories, then subtracts centred placebo from centred active to obtain contrasts. It uses only common observed post-dose times, with no zero filling or outcome interpolation.

One placebo period and at most one period per treatment per patient are required. A study with repeated same-treatment periods or a different pairing/washout design requires an explicit study-specific importer, not silent averaging. The importer assumes all rows otherwise pass the study's required clinical quality filters; it does not infer washout, carry-over, randomisation or baseline eligibility from measurements.

Replicate mean variances use sample variance divided by replicate count, propagate through the pre-dose mean and active-minus-placebo difference, and yield a Gaussian expected-absolute-error floor `sqrt(2/pi)*SD`. Unavailable variances remain unavailable; coverage is reported. Independence of the relevant measurement-error terms is an assumption, not estimated from one replicate. Population anchors use training-only contrast means and between-patient standard errors, separately from the measurement floor.
