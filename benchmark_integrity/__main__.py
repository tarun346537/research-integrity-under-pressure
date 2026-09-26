"""Run offline controls, freeze experiments, and run explicitly budgeted pilots."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from .artifacts import canonical_hash, envelope, read_json, source_hash, utc_now, write_json
from .environment import Environment


def build_plan(args) -> None:
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be unique within a plan.")
    if len(set(args.conditions)) != len(args.conditions):
        raise ValueError("Conditions must be unique within a plan.")
    if len(set(args.targets)) != len(args.targets):
        raise ValueError("Targets must be unique within a plan.")
    cases = []
    for index, seed in enumerate(args.seeds):
        cells = [(condition, target) for condition in args.conditions for target in args.targets]
        for condition, target in (cells if index % 2 == 0 else list(reversed(cells))):
            env = Environment(seed=seed, condition=condition, target=target)
            cases.append({
                "seed": seed, "condition": condition, "target_accuracy": target,
                "reference_solution": env.reference_solution(),
                "prompt_sha256": canonical_hash(env.prompt),
                "tools_sha256": canonical_hash(env.tool_schemas()),
            })
    plan = {
        "artifact_version": 1, "kind": "experiment_plan", "created_at": utc_now(),
        "phase": args.phase, "source_sha256": source_hash(),
        "model": "gpt-6-astra", "reasoning_effort": args.reasoning,
        "max_turns": args.max_turns, "max_actions": args.max_actions,
        "max_output_tokens": args.max_output_tokens,
        "cases": cases,
        "interpretation": "Small synthetic pilot; no estimate of general misalignment prevalence.",
    }
    if Path(args.out).exists():
        raise ValueError("Plan already exists. Choose a new filename to preserve provenance.")
    write_json(args.out, plan)
    print(f"Saved {len(cases)} planned episodes to {args.out}. No API calls made.")


def run_pilot(args) -> None:
    from .pilot import run_two_episode_pilot

    manifest = run_two_episode_pilot(
        read_json(args.plan), output=args.out,
        api_key=os.environ.get("OPENAI_API_KEY", ""), budget_usd=args.budget_usd,
    )
    print(f"Pilot {manifest['status']}; evidence and regrades saved in {args.out}.")
    if manifest["status"] != "completed":
        raise ValueError(f"Pilot stopped: {manifest['stop_reason']}. No automatic retry.")


def freeze_main(args) -> None:
    from .main_experiment import build_main_plan

    if Path(args.out).exists():
        raise ValueError("Main plan already exists; choose a new filename to preserve the frozen specification.")
    plan = build_main_plan()
    write_json(args.out, plan)
    print(f"Frozen ten main episodes (seeds 1-5, paired 80%/95%) in {args.out}; $5 maximum. No API calls made.")


def execute_main(args) -> None:
    from .main_experiment import run_main_experiment
    from .pilot import read_evidence

    manifest = run_main_experiment(
        read_evidence(args.plan), output=args.out,
        api_key=os.environ.get("OPENAI_API_KEY", ""), budget_usd=args.budget_usd,
    )
    print(f"Main experiment {manifest['status']}; all planned cases and attempts are recorded in {args.out}.")
    if manifest["status"] != "completed":
        raise ValueError(f"Main experiment stopped: {manifest['stop_reason']}. No automatic retry or replacement.")


def run_plan(args) -> None:
    from .runner import Budget, run_episode

    plan = read_json(args.plan)
    if plan.get("kind") != "experiment_plan":
        raise ValueError("Not an experiment plan.")
    if plan["source_sha256"] != source_hash():
        raise ValueError("Source changed since plan creation. Freeze a new plan before running.")
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set. Never place a key in source, command arguments, or a repository.")
    output = Path(args.out)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Run directory must be empty or new. Existing runs are never overwritten.")
    budget = Budget(args.budget_usd)
    manifest = {
        "artifact_version": 1, "kind": "model_run", "started_at": utc_now(),
        "plan": plan, "plan_sha256": canonical_hash(plan), "episodes": [],
        "budget": budget.as_dict(),
    }
    write_json(output / "run.json", manifest)
    stop_reason = None
    for index, case in enumerate(plan["cases"]):
        stem = f"episode-{index + 1:02d}-{case['condition']}-target-{case['target_accuracy']:g}-seed-{case['seed']}"
        if stop_reason:
            manifest["episodes"].append({**case, "status": "not_started", "reason": stop_reason})
            continue
        env = Environment(seed=case["seed"], condition=case["condition"], target=case["target_accuracy"])
        if canonical_hash(env.prompt) != case["prompt_sha256"] or canonical_hash(env.tool_schemas()) != case["tools_sha256"]:
            raise ValueError("Prompt or tools differ from the frozen plan.")
        pending = envelope(env, kind="model_episode", name=stem, status="started")
        pending.update({"phase": plan["phase"], "plan_sha256": manifest["plan_sha256"]})
        write_json(output / f"{stem}.json", pending)
        def save_progress(snapshot):
            snapshot.update({
                "artifact_version": 1, "kind": "model_episode", "name": stem,
                "phase": plan["phase"], "source_sha256": plan["source_sha256"],
                "plan_sha256": manifest["plan_sha256"],
            })
            write_json(output / f"{stem}.json", snapshot)
        result = run_episode(
            env, api_key=api_key, budget=budget, model=plan["model"],
            max_turns=plan["max_turns"], max_output_tokens=plan["max_output_tokens"],
            max_actions=plan["max_actions"],
            reasoning_effort=plan["reasoning_effort"],
            on_progress=save_progress,
        )
        result.update({
            "artifact_version": 1, "kind": "model_episode", "name": stem,
            "phase": plan["phase"], "source_sha256": source_hash(),
            "plan_sha256": manifest["plan_sha256"],
        })
        write_json(output / f"{stem}.json", result)
        manifest["episodes"].append({**case, "file": f"{stem}.json", "status": result["status"]})
        manifest["budget"] = budget.as_dict()
        write_json(output / "run.json", manifest)
        print(f"{stem}: {result['status']} | {json.dumps(budget.as_dict())}")
        if result["status"] in {"api_error", "budget_exhausted", "interrupted", "accounting_error"} or getattr(budget, "halted", False):
            stop_reason = result["status"]
    manifest["finished_at"] = utc_now()
    manifest["budget"] = budget.as_dict()
    write_json(output / "run.json", manifest)
    print(f"All started episodes and unstarted planned cases are recorded in {output / 'run.json'}.")


def verify_replay(artifact: dict) -> dict:
    """Rebuild and verify the complete saved environment without model calls.

    Tool outputs alone are insufficient: a saved experiment snapshot or its
    population could otherwise be edited without invalidating a replay.
    Historical schema versions must be replayed using their preserved source.
    """
    if not isinstance(artifact, dict):
        raise ValueError("Artifact must contain a JSON object.")
    saved = artifact.get("environment", artifact)
    if not isinstance(saved, dict):
        raise ValueError("Saved environment must be a JSON object.")
    current_schema = Environment().export()["schema_version"]
    if saved.get("schema_version") != current_schema:
        raise ValueError("Environment schema differs; replay with the preserved source for that artifact version.")
    try:
        env = Environment(seed=saved["seed"], condition=saved["condition"], target=saved["target_accuracy"])
        actions = saved["actions"]
    except KeyError as error:
        raise ValueError(f"Missing environment field: {error.args[0]}.") from None
    if not isinstance(actions, list):
        raise ValueError("Saved actions must be a list.")
    for index, action in enumerate(actions):
        if not isinstance(action, dict) or not {"name", "arguments", "result"} <= action.keys():
            raise ValueError(f"Invalid saved action {index + 1}.")
        actual = env.call(action["name"], action["arguments"])
        if canonical_hash(actual) != canonical_hash(action["result"]):
            raise ValueError(f"Replay differs at action {index + 1}.")
    regenerated = env.export()
    if canonical_hash(regenerated) != canonical_hash(saved):
        fields = sorted(key for key in regenerated.keys() | saved.keys()
                        if canonical_hash(regenerated.get(key)) != canonical_hash(saved.get(key))
                        or (key in regenerated) != (key in saved))
        raise ValueError(f"Replayed environment differs in: {', '.join(fields)}.")
    if "environment" in artifact and canonical_hash(artifact.get("grade")) != canonical_hash(regenerated["grade"]):
        raise ValueError("Replayed grade differs from the envelope grade.")
    return regenerated["grade"]


def replay(args) -> None:
    grade = verify_replay(read_json(args.artifact))
    print("Replay matched every tool result, immutable experiment record, report, and stored grade.")
    print(json.dumps(grade, indent=2))


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Must be a positive finite number.")
    return number


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be positive.")
    return number


def probability(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("Must be a finite probability between 0 and 1.")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Run deterministic scripted controls without any API calls")
    demo.add_argument("--out", default="examples/controls")
    demo.set_defaults(handler=lambda args: __import__("benchmark_integrity.controls", fromlist=["run_controls"]).run_controls(args.out))
    plan = sub.add_parser("plan", help="Freeze an experiment specification without calling the API")
    plan.add_argument("--phase", choices=["pilot", "main"], required=True)
    plan.add_argument("--seeds", type=int, nargs="+", default=[0])
    plan.add_argument("--conditions", choices=["editable", "locked"], nargs="+", default=["editable"])
    plan.add_argument("--targets", type=probability, nargs="+", default=[0.8, 0.95])
    plan.add_argument("--reasoning", choices=["low", "medium", "high", "xhigh", "max"], default="medium")
    plan.add_argument("--max-turns", type=positive_int, default=15)
    plan.add_argument("--max-actions", type=positive_int, default=15)
    plan.add_argument("--max-output-tokens", type=positive_int, default=4000)
    plan.add_argument("--out", required=True)
    plan.set_defaults(handler=build_plan)
    run = sub.add_parser("run", help="Make paid API calls using an explicit plan and dollar budget")
    run.add_argument("--plan", required=True)
    run.add_argument("--budget-usd", type=positive_float, required=True)
    run.add_argument("--out", required=True)
    run.set_defaults(handler=run_plan)
    pilot = sub.add_parser("pilot", help="Run exactly two approved pilot episodes with saved-evidence gates")
    pilot.add_argument("--plan", required=True)
    pilot.add_argument("--budget-usd", type=positive_float, required=True)
    pilot.add_argument("--out", required=True)
    pilot.set_defaults(handler=run_pilot)
    mainplan = sub.add_parser("main-plan", help="Freeze exactly ten main cases and the unchanged settings; no API calls")
    mainplan.add_argument("--out", required=True)
    mainplan.set_defaults(handler=freeze_main)
    mainrun = sub.add_parser("main", help="Run the approved ten-case main experiment with a gate after every saved attempt")
    mainrun.add_argument("--plan", required=True)
    mainrun.add_argument("--budget-usd", type=positive_float, required=True)
    mainrun.add_argument("--out", required=True)
    mainrun.set_defaults(handler=execute_main)
    rep = sub.add_parser("replay", help="Verify stored environment actions deterministically offline")
    rep.add_argument("artifact")
    rep.set_defaults(handler=replay)
    args = parser.parse_args()
    try:
        args.handler(args)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
