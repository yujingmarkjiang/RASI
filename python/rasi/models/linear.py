"""Exact normalised factual + active-minus-comparator contrast objective.

No replication of factual labels is used as a substitute for D = X_a - X_a'.
The NumPy implementation solves the augmented least-squares problem, not an
explicit matrix inverse. The optional C++ backend evaluates the same equations.
"""
from __future__ import annotations
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
import ctypes
import copy
import os
import numpy as np
from .base import FeatureSchema, FitRequest
from ..config import ModelConfig
from ..data import Query


class LinearWorldModel:
    def __init__(self, config: ModelConfig, schema: FeatureSchema, seed: int = 0):
        self.config, self.schema, self.seed = config, schema, seed
        self._p = {p: i for i, p in enumerate(schema.subjects)}
        self._a = {a: i for i, a in enumerate(schema.treatments)}
        self.na, self.nt, self.np = len(self._a), len(schema.times), len(self._p)
        self.nc = len(schema.context_mean)
        self.size = self.na * self.nt + self.np + self.np * self.na + self.nc * (self.na + 1)
        self.theta = np.zeros(self.size, dtype=np.float64)
        self.fitted = False

    def design(self, q: Query) -> np.ndarray:
        if q.treatment not in self._a:
            raise ValueError(f"No treatment support for {q.treatment!r}")
        a = self._a[q.treatment]
        x = np.zeros((len(q.times), self.size))
        times = np.asarray(self.schema.times)
        for row, t in enumerate(q.times):
            if t < times[0] or t > times[-1]:
                raise ValueError("The linear time-cell model does not extrapolate outside its fitted support")
            j = int(np.searchsorted(times, t))
            if j < len(times) and t == times[j]:
                x[row, a * self.nt + j] = 1.0
            else:
                frac = (t - times[j - 1]) / (times[j] - times[j - 1])
                x[row, a * self.nt + j - 1] = 1.0 - frac
                x[row, a * self.nt + j] = frac
        if q.subject in self._p:
            p = self._p[q.subject]
            x[:, self.na * self.nt + p] = 1
            x[:, self.na * self.nt + self.np + p * self.na + a] = 1
        # An unseen patient has zero learned person/deviation terms, not a made-up embedding.
        offset = self.na * self.nt + self.np + self.np * self.na
        context = self.schema.context(q)
        x[:, offset:offset + self.nc] = context
        x[:, offset + self.nc * (a + 1):offset + self.nc * (a + 2)] = context
        return x

    def fork(self, seed: int) -> LinearWorldModel:
        """Independent proposal snapshot; no mutable weights are shared."""
        model = copy.deepcopy(self)
        model.seed = seed
        return model

    def fit(self, request: FitRequest) -> None:
        xs, ys, ws = [], [], []
        for o in request.observations:
            w = request.observation_weight(o)
            if w > 0:
                xs.append(self.design(o.query)); ys.extend(o.values); ws.extend([w] * len(o.values))
        if not xs:
            raise ValueError("No positive-weight factual/synthetic observations")
        x, y, w = np.vstack(xs), np.asarray(ys), np.asarray(ws)
        ds, targets, cweights = [], [], []
        for c in request.contrasts:
            cw = request.contrast_sample_weight(c)
            if cw <= 0:
                continue
            active = np.mean([self.design(q) for q in c.active], axis=0)
            comparator = np.mean([self.design(q) for q in c.comparator], axis=0) if c.comparator else np.zeros_like(active)
            ds.append(active - comparator)
            targets.extend(c.values); cweights.extend([cw] * len(c.values))
        d = np.vstack(ds) if ds else np.empty((0, self.size))
        target, cw = np.asarray(targets), np.asarray(cweights)
        if request.contrast_weight > 0 and not len(d):
            raise ValueError("Positive lambda requires visible recorded contrasts")
        if self.config.backend == "native":
            theta = _native_solve(x, y, w, d, target, cw, request.contrast_weight, self.config.ridge)
        else:
            aw = np.sqrt(w / np.sum(w))
            matrices, vectors = [x * aw[:, None]], [y * aw]
            if len(d) and request.contrast_weight > 0:
                dw = np.sqrt(request.contrast_weight * cw / np.sum(cw))
                matrices.append(d * dw[:, None]); vectors.append(target * dw)
            if self.config.ridge > 0:
                matrices.append(np.eye(self.size) * np.sqrt(self.config.ridge)); vectors.append(np.zeros(self.size))
            theta = np.linalg.lstsq(np.vstack(matrices), np.concatenate(vectors), rcond=None)[0]
        if not np.all(np.isfinite(theta)):
            raise FloatingPointError("Nonfinite fitted coefficients")
        self.theta, self.fitted = theta, True

    def predict(self, query: Query) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Model has not been fitted")
        return self.design(query) @ self.theta

    def encode(self, query: Query) -> np.ndarray:
        # The current model's response representation; no future recorded values.
        response = [float(np.mean(self.predict(replace(query, treatment=a)))) for a in self.schema.treatments]
        return np.concatenate([response, np.mean(self.schema.context(query), axis=0)])

    def state_dict(self) -> dict[str, Any]:
        if not self.fitted:
            raise RuntimeError("Cannot checkpoint an unfitted model")
        return {"type": "linear", "config": asdict(self.config), "schema": self.schema.to_dict(),
                "seed": self.seed, "theta": self.theta.tolist()}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> LinearWorldModel:
        obj = cls(ModelConfig(**state["config"]), FeatureSchema.from_dict(state["schema"]), state["seed"])
        theta = np.asarray(state["theta"], dtype=np.float64)
        if theta.shape != (obj.size,) or not np.all(np.isfinite(theta)):
            raise ValueError("Malformed linear model checkpoint")
        obj.theta, obj.fitted = theta, True
        return obj


def _native_solve(x: np.ndarray, y: np.ndarray, w: np.ndarray, d: np.ndarray, target: np.ndarray,
                  cw: np.ndarray, lam: float, eta: float) -> np.ndarray:
    default = Path(__file__).resolve().parents[3] / "cpp" / "build" / "librasi.so"
    path = os.environ.get("RASI_NATIVE_LIB", str(default))
    try:
        lib = ctypes.CDLL(path)
    except OSError as error:
        raise RuntimeError("Native backend missing: run make native, set RASI_NATIVE_LIB, or use backend=numpy") from error
    ptr = ctypes.POINTER(ctypes.c_double)
    lib.rasi_solve.argtypes = [ctypes.c_int, ctypes.c_int, ptr, ptr, ptr, ctypes.c_int, ptr, ptr, ptr,
                               ctypes.c_double, ctypes.c_double, ptr]
    lib.rasi_solve.restype = ctypes.c_int
    arrays = [np.ascontiguousarray(a, dtype=np.float64) for a in (x, y, w, d, target, cw)]
    out = np.zeros(x.shape[1], dtype=np.float64)
    xp, yp, wp, dp, tp, cp = [a.ctypes.data_as(ptr) for a in arrays]
    status = lib.rasi_solve(len(x), x.shape[1], xp, yp, wp, len(d), dp, tp, cp, lam, eta, out.ctypes.data_as(ptr))
    if status:
        raise RuntimeError(f"Native linear solver failed (status {status}); for rank-deficient problems use the NumPy SVD backend")
    return out
