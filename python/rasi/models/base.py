"""The adapter contract shared by linear, recurrent and external world models."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Protocol, Any, Callable
import numpy as np
from ..data import Query, Trajectory, Contrast
from ..config import ModelConfig


@dataclass(frozen=True)
class FitRequest:
    observations: tuple[Trajectory, ...]
    contrasts: tuple[Contrast, ...]
    contrast_weight: float
    validation: tuple[Trajectory, ...] = ()
    bootstrap: dict[str, int] | None = None

    def observation_weight(self, row: Trajectory) -> float:
        return row.weight * (self.bootstrap.get(row.query.subject, 0) if self.bootstrap is not None else 1)

    def contrast_sample_weight(self, pair: Contrast) -> float:
        if self.bootstrap is None:
            return 1.0
        # Individual paired records share their patient's bootstrap count.
        # A separately estimated trial/stratum constraint is not an individual row.
        if pair.scope == "stratum":
            return 1.0
        return float(self.bootstrap.get(pair.active[0].subject, 0))


class WorldModel(Protocol):
    def fork(self, seed: int) -> WorldModel: ...
    def fit(self, request: FitRequest) -> None: ...
    def predict(self, query: Query) -> np.ndarray: ...
    def encode(self, query: Query) -> np.ndarray: ...
    def state_dict(self) -> dict[str, Any]: ...


ModelFactory = Callable[[int], WorldModel]


@dataclass(frozen=True)
class FeatureSchema:
    subjects: tuple[str, ...]
    treatments: tuple[str, ...]
    times: tuple[float, ...]
    context_mean: tuple[float, ...]
    context_scale: tuple[float, ...]

    @classmethod
    def build(cls, queries: tuple[Query, ...], real: tuple[Trajectory, ...]) -> FeatureSchema:
        if not queries or not real:
            raise ValueError("A model needs real data and nonempty input support")
        width = len(real[0].query.context[0])
        if any(len(q.context[0]) != width for q in queries):
            raise ValueError("Inconsistent context width")
        x = np.asarray([row for o in real for row in o.query.context], dtype=float).reshape(-1, width) if width else np.zeros((1, 0))
        scale = np.std(x, axis=0)
        return cls(tuple(sorted({q.subject for q in queries})), tuple(sorted({q.treatment for q in queries})),
                   tuple(sorted({t for q in queries for t in q.times})), tuple(np.mean(x, axis=0)),
                   tuple(np.where(scale > 1e-12, scale, 1.0)))

    def context(self, query: Query) -> np.ndarray:
        if len(query.context[0]) != len(self.context_mean):
            raise ValueError("Query context width differs from the fitted schema")
        return (np.asarray(query.context) - self.context_mean) / self.context_scale

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FeatureSchema:
        return cls(**{key: tuple(value) for key, value in d.items()})


def make_factory(config: ModelConfig, queries: tuple[Query, ...], real: tuple[Trajectory, ...]) -> ModelFactory:
    schema = FeatureSchema.build(queries, real)
    if config.kind == "linear":
        from .linear import LinearWorldModel
        return lambda seed: LinearWorldModel(config, schema, seed)
    from .gru import GRUWorldModel
    return lambda seed: GRUWorldModel(config, schema, seed)


def load_model(state: dict[str, Any], device: str | None = None) -> WorldModel:
    """Do not import arbitrary code or unpickle objects named by a checkpoint."""
    if state.get("type") == "linear":
        from .linear import LinearWorldModel
        return LinearWorldModel.from_state(state)
    if state.get("type") == "gru":
        from .gru import GRUWorldModel
        return GRUWorldModel.from_state(state, device=device)
    raise ValueError("Unrecognised checkpoint model type; external adapters need an explicit application loader")
