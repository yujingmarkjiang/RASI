"""Command-line entry points for real-data preparation, training and inference."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from typing import Any
import argparse
import json
import sys
from .config import RASIConfig, preset_config
from .data import Problem, Query, Target, target_prediction
from .engine import RASI
from .preprocess import prepare_contrasts, prepare_qt
from .selection import Selection
from .storage import atomic_json, read_checkpoint
from .diagnostics import gpu_test, model_smoke, system_report


def _input_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--data", help="An explicitly partitioned RASI dataset JSON")
    group.add_argument("--cohort", help="A precomputed contrast CSV (no factual outcomes implied)")
    group.add_argument("--raw-qt", help="Replicate QT/RR CSV with actual active and comparator outcomes")
    parser.add_argument("--targets", help="Input-only target CSV for --cohort; never hidden labels")
    parser.add_argument("--audit-subjects", type=int, default=12)
    parser.add_argument("--validation-subjects", type=int, default=2)
    parser.add_argument("--placebo", default="Placebo")


def _problem(args: argparse.Namespace, seed: int) -> Problem:
    if args.data:
        if args.targets:
            raise ValueError("JSON datasets already declare targets")
        return Problem.load(args.data)
    if args.cohort:
        return prepare_contrasts(args.cohort, args.audit_subjects, args.validation_subjects, seed, args.targets)
    if args.targets:
        raise ValueError("For raw-QT custom targets, edit the prepared JSON explicitly")
    return prepare_qt(args.raw_qt, args.placebo, args.audit_subjects, args.validation_subjects, seed)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rasi", description="Randomisation-anchored self-improvement; no experiment runner")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Reserve independent audit data and construct real contrast estimands")
    _input_options(prepare)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--seed", type=int, default=20261001)
    validate = sub.add_parser("validate", help="Validate input shape and data separation; does not train")
    validate.add_argument("--data", required=True)
    train = sub.add_parser("train", aliases=["loop", "apply"], help="Fit a real-data baseline and run the RASI controller")
    _input_options(train)
    train.add_argument("--config", help="RASI settings JSON; unknown keys are errors")
    train.add_argument("--output", "--out", required=True, help="Private run directory")
    train.add_argument("--selection", help="Settings selected on the first visible dataset; frozen across later datasets")
    train.add_argument("--resume", action="store_true")
    train.add_argument("--rounds", type=int)
    train.add_argument("--m", type=int, help="Mean rollout budget per unobserved arm")
    train.add_argument("--seed", type=int)
    train.add_argument("--model", choices=["linear", "gru"])
    train.add_argument("--backend", choices=["numpy", "native"])
    train.add_argument("--device", help="cpu or a real PyTorch device such as cuda:0")
    train.add_argument("--quiet", action="store_true")
    doctor = sub.add_parser("doctor", help="Inspect the Python/toolchain/GPU installation without training")
    doctor.add_argument("--config", help="Optionally check whether a RASI config can run on this machine")
    doctor.add_argument("--strict", action="store_true", help="Exit nonzero when the selected config is not runnable")
    gpu = sub.add_parser("test-gpu", help="Run an actual CUDA forward/backward compute check")
    gpu.add_argument("--device", default="cuda:0")
    gpu.add_argument("--matrix-size", type=int, default=512)
    gpu.add_argument("--repeats", type=int, default=3)
    gpu.add_argument("--allow-missing", action="store_true", help="Return success when CUDA is unavailable; useful for CPU-only CI")
    smoke = sub.add_parser("smoke-test", help="Fit and reload a tiny synthetic model; no manuscript experiment is run")
    smoke.add_argument("--config", help="Use the model section from this config")
    smoke.add_argument("--model", choices=["linear", "gru"])
    smoke.add_argument("--backend", choices=["numpy", "native"])
    smoke.add_argument("--device")
    smoke.add_argument("--epochs", type=int, default=3, help="Maximum GRU epochs used only by the smoke check")
    init = sub.add_parser("init-config", help="Write a validated starter configuration")
    init.add_argument("--preset", required=True, choices=["linear-cpu", "linear-native", "gru-cpu", "gru-cuda"])
    init.add_argument("--output", required=True)
    show = sub.add_parser("show-config", help="Parse and print a normalized configuration")
    show.add_argument("--config", required=True)
    infer = sub.add_parser("infer", help="Predict using an accepted checkpoint; never retrains")
    infer.add_argument("--checkpoint", "--ckpt", required=True)
    group = infer.add_mutually_exclusive_group(required=True)
    group.add_argument("--queries", help="JSON list of input Query or Target records")
    group.add_argument("--data", help="Use only the explicitly declared targets in a dataset JSON")
    infer.add_argument("--output")
    infer.add_argument("--device", help="Override a neural checkpoint's device; no hardware emulation")
    return parser


def _describe(problem: Problem) -> dict[str, Any]:
    view = problem.training
    return {"valid": True, "representation": view.representation, "training_trajectories": len(view.real),
            "training_contrasts": len(view.contrasts), "factual_validation_trajectories": len(view.validation) if view.representation == "outcome" else None,
            "contrast_validation_trajectories": len(view.validation) if view.representation == "contrast" else None,
            "unobserved_targets": len(view.targets), "reserved_audit_contrasts": len(problem.audit),
            "independent_audit_units": len({c.unit_id for c in problem.audit}), "anchors": len(view.anchors),
            "warning": "Factual outcome error cannot be evaluated from precomputed contrasts" if view.representation == "contrast" else None}


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            problem = _problem(args, args.seed)
            if Path(args.output).exists():
                raise FileExistsError("Prepared dataset already exists; choose a new path rather than silently overwriting a split")
            atomic_json(args.output, problem.to_dict())
            print(json.dumps(dict(_describe(problem), dataset=str(Path(args.output))), allow_nan=False))
        elif args.command == "validate":
            print(json.dumps(_describe(Problem.load(args.data)), allow_nan=False))
        elif args.command in {"train", "loop", "apply"}:
            config = RASIConfig.from_dict(json.loads(Path(args.config).read_text())) if args.config else RASIConfig()
            changes = {key: getattr(args, key) for key in ("rounds", "seed") if getattr(args, key) is not None}
            if args.m is not None:
                changes["rollouts_per_arm"] = args.m
            model_changes = {name: getattr(args, name) for name in ("backend", "device") if getattr(args, name) is not None}
            if args.model is not None:
                model_changes["kind"] = args.model
            if model_changes:
                changes["model"] = replace(config.model, **model_changes)
            config = replace(config, **changes)
            problem = _problem(args, config.seed)
            selection = Selection(**json.loads(Path(args.selection).read_text())) if args.selection else None
            if not args.quiet:
                print(json.dumps(_describe(problem), allow_nan=False), file=sys.stderr)
            def progress(event: dict[str, Any]) -> None:
                if not args.quiet:
                    print(json.dumps({"round": event["round"], "accepted": event["accepted"], "reason": event["reason"],
                                      "verified_rollouts": event.get("verified_rollouts", 0)}, allow_nan=False), file=sys.stderr, flush=True)
            summary = RASI(problem, config, selection).run(args.output, resume=args.resume, progress=progress)
            print(json.dumps({"checkpoint": str(Path(args.output) / "accepted.json"), "summary": str(Path(args.output) / "summary.json"),
                              "rounds_completed": summary["rounds_completed"], "accepted_rounds": summary["accepted_rounds"],
                              "stop_reason": summary["stop_reason"], "guarantee": summary["guarantee"]}, allow_nan=False))
        elif args.command == "doctor":
            config = RASIConfig.from_dict(json.loads(Path(args.config).read_text())) if args.config else None
            report = system_report(config)
            print(json.dumps(report, allow_nan=False, indent=2))
            if args.strict and config is not None and not report.get("config_ready", False):
                return 3
        elif args.command == "test-gpu":
            report = gpu_test(args.device, args.matrix_size, args.repeats)
            print(json.dumps(report, allow_nan=False, indent=2))
            if not report.get("passed", False):
                if not (args.allow_missing and report.get("status") == "unavailable"):
                    return 3
        elif args.command == "smoke-test":
            config = RASIConfig.from_dict(json.loads(Path(args.config).read_text())) if args.config else RASIConfig()
            model = config.model
            changes = {key: getattr(args, key) for key in ("backend", "device") if getattr(args, key) is not None}
            if args.model is not None:
                changes["kind"] = args.model
            if changes:
                model = replace(model, **changes)
            report = model_smoke(model, epochs=args.epochs)
            print(json.dumps(report, allow_nan=False, indent=2))
        elif args.command == "init-config":
            output = Path(args.output)
            if output.exists():
                raise FileExistsError("Config output already exists; refusing to overwrite it")
            atomic_json(output, preset_config(args.preset).to_dict())
            print(json.dumps({"config": str(output), "preset": args.preset}, allow_nan=False))
        elif args.command == "show-config":
            config = RASIConfig.from_dict(json.loads(Path(args.config).read_text()))
            print(json.dumps(config.to_dict(), allow_nan=False, indent=2))
        elif args.command == "infer":
            model, metadata = read_checkpoint(args.checkpoint, device=args.device)
            content = json.loads(Path(args.data or args.queries).read_text())
            raw_queries = content.get("targets", []) if args.data else content
            if not isinstance(raw_queries, list):
                raise ValueError("--queries must contain a JSON list of Query or Target objects")
            results = []
            for raw in raw_queries:
                if "active" in raw:
                    query = Target.from_dict(raw)
                    if metadata.get("representation") == "contrast" and query.comparator is not None:
                        raise ValueError("A direct-contrast checkpoint cannot be used as a factual outcome simulator")
                    if metadata.get("representation") == "outcome" and query.comparator is None:
                        raise ValueError("An outcome checkpoint needs a comparator to produce a treatment contrast")
                    results.append({"id": query.id, "times": query.active.times,
                                    "active_prediction": model.predict(query.active).tolist(),
                                    "contrast": target_prediction(model, query).tolist()})
                else:
                    query = Query.from_dict(raw)
                    results.append({"id": query.id, "times": query.times, "prediction": model.predict(query).tolist()})
            body = {"checkpoint": str(args.checkpoint), "last_accepted_round": metadata.get("last_accepted_round", 0),
                    "representation": metadata.get("representation"), "predictions": results}
            if args.output:
                atomic_json(args.output, body)
            print(json.dumps(body, allow_nan=False))
        return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, ImportError, FloatingPointError) as error:
        print(f"rasi: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
