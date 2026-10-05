"""One RASI loop for every world-model adapter.

Allocation -> real-neighbour means -> contrast verifier -> contrast-penalised
fit -> fresh audit -> accept or restore. The current accepted model, not a
retrospectively selected 'best' model on the entire audit pool, is deployed.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Any
import copy
import json
import numpy as np
from .audit import AuditGate, AuditExhausted
from .config import RASIConfig
from .data import Problem, TrainingView, Trajectory, factual_mae, schema_queries
from .models import FitRequest, WorldModel, ModelFactory, make_factory, load_model
from .retrieval import allocate, retrieve, verify
from .selection import Selection, select, training_digest
from .storage import RunStore, atomic_json, atomic_jsonlines, checkpoint, read_checkpoint, digest


@dataclass(frozen=True)
class ProposalContext:
    """For a trusted agent/architecture or aggregation proposal; never includes audit data."""
    training: TrainingView
    observations: tuple[Trajectory, ...]
    selection: Selection
    round: int
    seed: int


ProposalFunction = Callable[[ProposalContext, WorldModel], WorldModel]


class RASI:
    def __init__(self, problem: Problem, config: RASIConfig, selection: Selection | None = None,
                 factory: ModelFactory | None = None, proposer: ProposalFunction | None = None):
        self.problem, self.config, self.selection = problem, config, selection
        self.factory = factory or make_factory(config.model, schema_queries(problem.training), problem.training.real)
        self.proposer = proposer
        self.external_factory = factory is not None
        if self.external_factory and selection is None:
            raise ValueError("External adapters must supply settings cross-validated on visible training data")
        if config.audit.require_certificate:
            reserved_subjects = {q.subject for c in problem.audit for q in c.queries}
            exposed_subjects = {q.subject for q in schema_queries(problem.training)}
            exposed_subjects |= {o.query.subject for o in problem.training.validation}
            if reserved_subjects & exposed_subjects:
                raise ValueError("A conditional certificate additionally requires subject-disjoint audit data")
        self.dataset_sha = digest(problem.to_dict())
        self.config_sha = digest(config.to_dict())

    def run(self, directory: str | Path, resume: bool = False,
            progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
        cfg, view = self.config, self.problem.training
        gate_seed = int(np.random.SeedSequence([cfg.seed, 7717]).generate_state(1)[0])
        gate = AuditGate(self.problem.audit, cfg.audit, cfg.rounds, gate_seed)
        rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 9811]))
        with RunStore(directory, resume=resume) as store:
            if resume:
                if self.external_factory:
                    raise ValueError("External adapter resume requires an application-specific trusted loader")
                state = json.loads((store.path / "resume.json").read_text())
                if state["dataset_sha256"] != self.dataset_sha or state["config_sha256"] != self.config_sha:
                    raise ValueError("Resume requires exactly the original data, audit pool and configuration")
                chosen = Selection(**state["selection"])
                if self.selection is not None and self.selection != chosen:
                    raise ValueError("Resume selection differs from the original frozen settings")
                current = load_model(state["working_model"], device=cfg.model.device)
                deployed = load_model(state["deployed_model"], device=cfg.model.device)
                base, _ = read_checkpoint(store.path / "base.json", device=cfg.model.device)
                synthetic = tuple(Trajectory.from_dict(x) for x in state["synthetic"])
                gate.restore(state["audit"]); rng.bit_generator.state = state["rng"]
                history = state["history"]
                completed, rejections = state["round"], state["consecutive_rejections"]
                accepted, accepted_round = state["accepted_rounds"], state["last_accepted_round"]
                stop_reason = state["stop_reason"]
            else:
                if self.selection is not None:
                    chosen = self.selection
                elif not cfg.rounds or not view.targets:
                    chosen = Selection(cfg.selection.lambda_grid[0], cfg.selection.k_grid[0],
                                       "unused: no recursive proposals requested", training_digest(view), {})
                else:
                    chosen = select(view, cfg)
                base = self.factory(cfg.seed)
                # Round zero is the real-data-only baseline, lambda=0.
                base.fit(FitRequest(view.real, view.contrasts, 0.0, view.validation))
                current, deployed = base, base
                synthetic: tuple[Trajectory, ...] = ()
                history: list[dict[str, Any]] = []
                completed = rejections = accepted = accepted_round = 0
                stop_reason = None
                atomic_json(store.path / "selection.json", chosen.to_dict())
                checkpoint(store.path / "base.json", base, {"round": 0, "source": "real_data_only", "dataset_sha256": self.dataset_sha,
                                                          "representation": view.representation})
            self.selection = chosen

            def metadata() -> dict[str, Any]:
                return {"last_accepted_round": accepted_round, "rounds_completed": completed,
                        "accepted_rounds": accepted, "consecutive_rejections": rejections,
                        "stop_reason": stop_reason, "dataset_sha256": self.dataset_sha,
                        "representation": view.representation, "selection": chosen.to_dict(),
                        "config": cfg.to_dict(), "guarantee": gate.certificate_summary(),
                        "deployment_rule": "last accepted model; no best-of-full-pool selection",
                        "factual_metric": "outcome MAE" if view.representation == "outcome" else "unavailable: input contains contrasts only"}

            def persist() -> None:
                # resume.json is the transaction commit. Other artefacts can be
                # reconstructed from it after a crash between atomic file writes.
                body = {"schema_version": 1, "dataset_sha256": self.dataset_sha, "config_sha256": self.config_sha,
                        "selection": chosen.to_dict(), "working_model": current.state_dict(), "deployed_model": deployed.state_dict(),
                        "synthetic": [asdict(x) for x in synthetic], "audit": gate.state_dict(),
                        "rng": rng.bit_generator.state, "history": history, "round": completed,
                        "consecutive_rejections": rejections, "accepted_rounds": accepted,
                        "last_accepted_round": accepted_round, "stop_reason": stop_reason}
                atomic_json(store.path / "resume.json", body)
                checkpoint(store.path / "accepted.json", deployed, metadata())
                atomic_jsonlines(store.path / "events.jsonl", history)

            persist()
            if not view.targets:
                stop_reason = "no_unobserved_arms"
            if rejections >= cfg.rejection_patience:
                stop_reason = "ten_consecutive_rejections"
            if stop_reason is None:
                for round_index in range(completed + 1, cfg.rounds + 1):
                    existing_data = view.real + synthetic
                    budget = allocate(self.factory, view, existing_data, chosen.contrast_weight, cfg.rollouts_per_arm, rng)
                    raw, retrieval_skips = retrieve(current, view, budget, chosen.neighbours, cfg.synthetic_weight)
                    verified, verifier_skips = verify(current, view, raw, cfg.verifier_z)
                    event: dict[str, Any] = {"round": round_index, "allocated_rollouts": budget.total_budget,
                        "retrieved_rollouts": sum(p.allocated for p in raw), "verified_rollouts": sum(p.allocated for p in verified),
                        "retrieval_skips": retrieval_skips, "verifier_skips": verifier_skips,
                        "allocation": [{"target_sha256": digest(t.id), "count": n, "disagreement": v}
                                       for t, n, v in zip(view.targets, budget.counts, budget.disagreement)],
                        "accepted": False, "audit": None}
                    if not verified:
                        event["reason"] = "no_anchor_consistent_real_rollouts"
                        rejections += 1
                    else:
                        fresh = tuple(p.trajectory for p in verified)
                        proposed_synthetic = fresh if cfg.regime == "replacement" else synthetic + fresh
                        observations = view.real + proposed_synthetic
                        fit_seed = int(rng.integers(0, 2**31 - 1))
                        if self.proposer is None:
                            # Neural proposals fine-tune a private incumbent
                            # snapshot. Linear fit() still solves from scratch.
                            candidate = current.fork(fit_seed)
                            candidate.fit(FitRequest(observations, view.contrasts, chosen.contrast_weight, view.validation))
                        else:
                            context = ProposalContext(view, observations, chosen, round_index, fit_seed)
                            # Never hand a callback the live incumbent object.
                            candidate = self.proposer(context, copy.deepcopy(current))
                        event["visible_training_mae"] = factual_mae(candidate, view.real)
                        event["visible_validation_mae"] = factual_mae(candidate, view.validation)
                        if cfg.audit.mode == "held_out":
                            current, synthetic = candidate, proposed_synthetic
                            event["reason"] = "provisional_only_final_audit_required"
                            rejections = 0
                        else:
                            try:
                                decision = gate.consider(current, candidate)
                            except AuditExhausted:
                                stop_reason = "audit_pool_exhausted"
                                event["reason"] = stop_reason
                                # This attempted proposal is logged but not deployed.
                                history.append(event); completed = round_index
                                persist()
                                if progress:
                                    progress(event)
                                break
                            event["audit"] = decision.to_dict()
                            event["accepted"], event["reason"] = decision.accepted, decision.reason
                            if decision.accepted:
                                current, deployed, synthetic = candidate, candidate, proposed_synthetic
                                accepted += 1; accepted_round = round_index; rejections = 0
                            else:
                                # No candidate parameters, optimiser state or rejected
                                # synthetic rows enter the next accepted training state.
                                rejections += 1
                    completed = round_index
                    event["consecutive_rejections"] = rejections
                    history.append(event)
                    if rejections >= cfg.rejection_patience:
                        stop_reason = "ten_consecutive_rejections"
                    persist()
                    if progress:
                        progress(event)
                    if stop_reason is not None:
                        break
                if stop_reason is None:
                    stop_reason = "round_limit"
                if cfg.audit.mode == "held_out" and current is not base and gate.state_dict()["evaluations"] == 0:
                    decision = gate.consider(base, current)
                    history.append({"round": completed, "phase": "single_final_audit", "accepted": decision.accepted,
                                    "audit": decision.to_dict(), "reason": decision.reason})
                    if decision.accepted:
                        deployed = current; accepted = 1; accepted_round = completed
                    else:
                        current = deployed = base; synthetic = ()
            persist()
            summary = metadata()
            summary["base_visible_mae"] = factual_mae(base, view.real)
            summary["deployed_visible_mae"] = factual_mae(deployed, view.real)
            summary["base_factual_validation_mae"] = factual_mae(base, view.validation) if view.representation == "outcome" else None
            summary["deployed_factual_validation_mae"] = factual_mae(deployed, view.validation) if view.representation == "outcome" else None
            summary["factual_non_degradation_guaranteed"] = False
            atomic_json(store.path / "summary.json", summary)
            return summary
