"""A real autoregressive GRU world model, using the common RASI interface.

Training uses AdamW, cosine decay, gradient clipping at 1, and early stopping
on separate visible factual validation data (patience 20 by default). Contrast
predictions are free rollouts, never teacher-forced with the contrast outcome.
"""
from __future__ import annotations
from dataclasses import asdict, replace
from typing import Any
import math
import copy
import numpy as np
from .base import FeatureSchema, FitRequest
from ..config import ModelConfig
from ..data import Query

try:
    import torch
    from torch import nn
except ImportError as error:
    raise ImportError("The GRU adapter needs the neural extra: pip install -e '.[neural]'") from error


class _Network(nn.Module):
    def __init__(self, schema: FeatureSchema, config: ModelConfig):
        super().__init__()
        e = config.embedding_size
        self.subject = nn.Embedding(len(schema.subjects) + 1, e, padding_idx=0)
        self.treatment = nn.Embedding(len(schema.treatments), e)
        self.cell = nn.GRUCell(2 * e + len(schema.context_mean) + 2, config.hidden_size)
        self.head = nn.Linear(config.hidden_size, 1)
        self.initial_token = nn.Parameter(torch.zeros(()))


class GRUWorldModel:
    def __init__(self, config: ModelConfig, schema: FeatureSchema, seed: int = 0):
        self.config, self.schema, self.seed = config, schema, seed
        self.device = torch.device(config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("A CUDA device was requested but PyTorch cannot use CUDA; no CPU fallback is assumed")
        self._p = {p: i + 1 for i, p in enumerate(schema.subjects)}
        self._a = {a: i for i, a in enumerate(schema.treatments)}
        torch.manual_seed(seed)
        self.net = _Network(schema, config).to(self.device)
        self.location, self.scale = 0.0, 1.0
        self.fitted = False
        self.fit_metadata: dict[str, Any] = {}

    def _forward(self, q: Query) -> tuple[torch.Tensor, torch.Tensor]:
        if q.treatment not in self._a:
            raise ValueError(f"No treatment support for {q.treatment!r}")
        p = torch.tensor([self._p.get(q.subject, 0)], device=self.device)
        a = torch.tensor([self._a[q.treatment]], device=self.device)
        emb = torch.cat((self.net.subject(p), self.net.treatment(a)), dim=-1)
        context = torch.tensor(self.schema.context(q), dtype=torch.float32, device=self.device)
        h = torch.zeros((1, self.config.hidden_size), device=self.device)
        previous = self.net.initial_token.reshape(1, 1) if q.initial_value is None else torch.tensor(
            [[(q.initial_value - self.location) / self.scale]], dtype=torch.float32, device=self.device)
        time_scale = max(abs(t) for t in self.schema.times) or 1.0
        result, hidden = [], []
        for j, t in enumerate(q.times):
            time = torch.tensor([[t / time_scale]], dtype=torch.float32, device=self.device)
            x = torch.cat((emb, context[j:j + 1], time, previous), dim=-1)
            h = self.net.cell(x, h)
            previous = self.net.head(h)
            result.append(previous.reshape(())); hidden.append(h.reshape(-1))
        return torch.stack(result) * self.scale + self.location, torch.stack(hidden).mean(dim=0)

    def _contrast(self, pair: Any) -> torch.Tensor:
        active = torch.stack([self._forward(q)[0] for q in pair.active]).mean(dim=0)
        comparator = torch.stack([self._forward(q)[0] for q in pair.comparator]).mean(dim=0) if pair.comparator else 0.0
        return active - comparator

    def fork(self, seed: int) -> GRUWorldModel:
        """Independent proposal snapshot; no mutable weights are shared."""
        model = copy.deepcopy(self)
        model.seed = seed
        return model

    def fit(self, request: FitRequest) -> None:
        if not request.validation:
            raise ValueError("GRU training requires separate visible factual validation trajectories; the audit cannot substitute for them")
        observations = [(o, request.observation_weight(o)) for o in request.observations if request.observation_weight(o) > 0]
        contrasts = [(c, request.contrast_sample_weight(c)) for c in request.contrasts if request.contrast_sample_weight(c) > 0]
        if not observations or (request.contrast_weight > 0 and not contrasts):
            raise ValueError("Empty positive-weight training data")
        values = [v for o, _ in observations if o.origin == "real" for v in o.values]
        if not values:
            raise ValueError("GRU scaling requires real training outcomes")
        self.location, self.scale = float(np.mean(values)), max(float(np.std(values)), 1e-6)
        factual_mass = sum(w * len(o.values) for o, w in observations)
        contrast_mass = sum(w * len(c.values) for c, w in contrasts)
        optimiser = torch.optim.AdamW(self.net.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=self.config.epochs)
        best_loss, bad_epochs, best_state, best_epoch = float("inf"), 0, None, 0
        rng = np.random.default_rng(self.seed)
        # Microbatch gradient accumulation exactly preserves the two separately
        # normalised loss terms. Every optimiser step sees the complete objective.
        for epoch in range(1, self.config.epochs + 1):
            self.net.train()
            optimiser.zero_grad(set_to_none=True)
            order = rng.permutation(len(observations))
            for start in range(0, len(order), self.config.batch_size):
                loss = torch.zeros((), device=self.device)
                for i in order[start:start + self.config.batch_size]:
                    o, w = observations[int(i)]
                    y = torch.tensor(o.values, dtype=torch.float32, device=self.device)
                    loss = loss + w * ((self._forward(o.query)[0] - y) ** 2).sum() / factual_mass
                (loss / self.scale ** 2).backward()
            if request.contrast_weight > 0:
                order_c = rng.permutation(len(contrasts))
                for start in range(0, len(order_c), self.config.batch_size):
                    loss = torch.zeros((), device=self.device)
                    for i in order_c[start:start + self.config.batch_size]:
                        c, w = contrasts[int(i)]
                        y = torch.tensor(c.values, dtype=torch.float32, device=self.device)
                        loss = loss + request.contrast_weight * w * ((self._contrast(c) - y) ** 2).sum() / contrast_mass
                    (loss / self.scale ** 2).backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0, error_if_nonfinite=True)
            optimiser.step(); scheduler.step()
            self.net.eval()
            with torch.no_grad():
                val_sum, val_n = 0.0, 0
                for o in request.validation:
                    target = torch.tensor(o.values, dtype=torch.float32, device=self.device)
                    val_sum += float(((self._forward(o.query)[0] - target) ** 2).sum().cpu())
                    val_n += len(o.values)
                val_loss = val_sum / val_n
            if not math.isfinite(val_loss):
                raise FloatingPointError("Nonfinite neural validation loss")
            if val_loss < best_loss:
                best_loss, bad_epochs, best_epoch = val_loss, 0, epoch
                best_state = {name: tensor.detach().clone() for name, tensor in self.net.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= self.config.early_stopping_patience:
                    break
        if best_state is None:
            raise RuntimeError("No finite neural checkpoint")
        self.net.load_state_dict(best_state)
        self.net.eval()
        self.fitted = True
        self.fit_metadata = {"epochs_completed": epoch, "best_epoch": best_epoch, "visible_validation_mse": best_loss,
                             "optimizer": "AdamW", "schedule": "cosine", "gradient_clip": 1.0}

    def predict(self, query: Query) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Model has not been fitted")
        self.net.eval()
        with torch.no_grad():
            result = self._forward(query)[0].cpu().numpy().astype(np.float64)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("Nonfinite GRU rollout")
        return result

    def encode(self, query: Query) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Model has not been fitted")
        self.net.eval()
        with torch.no_grad():
            return self._forward(query)[1].cpu().numpy().astype(np.float64)

    def state_dict(self) -> dict[str, Any]:
        if not self.fitted:
            raise RuntimeError("Cannot checkpoint an unfitted model")
        return {"type": "gru", "config": asdict(self.config), "schema": self.schema.to_dict(), "seed": self.seed,
                "location": self.location, "scale": self.scale, "fit_metadata": self.fit_metadata,
                "weights": {k: v.detach().cpu().tolist() for k, v in self.net.state_dict().items()}}

    @classmethod
    def from_state(cls, state: dict[str, Any], device: str | None = None) -> GRUWorldModel:
        config = ModelConfig(**state["config"])
        if device is not None:
            config = replace(config, device=device)
        obj = cls(config, FeatureSchema.from_dict(state["schema"]), state["seed"])
        obj.location, obj.scale = float(state["location"]), float(state["scale"])
        if not math.isfinite(obj.location) or not math.isfinite(obj.scale) or obj.scale <= 0:
            raise ValueError("Invalid GRU normalisation state")
        weights = {k: torch.tensor(v, dtype=torch.float32, device=obj.device) for k, v in state["weights"].items()}
        if any(not bool(torch.isfinite(v).all()) for v in weights.values()):
            raise ValueError("Nonfinite GRU parameters")
        obj.net.load_state_dict(weights, strict=True)
        obj.net.eval(); obj.fitted = True
        obj.fit_metadata = dict(state.get("fit_metadata", {}))
        return obj
