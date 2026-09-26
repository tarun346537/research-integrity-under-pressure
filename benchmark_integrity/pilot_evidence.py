"""Offline consistency gate for one completed pilot's saved evidence.

This verifies provenance and mechanical consistency, not provider authorship,
intent, subjective morality, or the meaning of free-text explanations. Opaque
response items are compared unchanged, never interpreted or used for grading.
"""
from __future__ import annotations

import copy
import json
import math
from decimal import Decimal
from typing import Any

from .artifacts import canonical_hash
from .environment import Environment


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _same(left: Any, right: Any, message: str) -> None:
    _require(canonical_hash(left) == canonical_hash(right), message)


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _reject_constant(value: str) -> None:
    raise ValueError("Non-finite JSON number.")


def verify_pilot_evidence(artifact: dict, plan: dict, case: dict,
                          current_source_sha256: str) -> dict:
    """Verify a disk-loaded completed artifact or raise ``ValueError``.

The caller must load JSON strictly, preserve failed attempts, and stop on an
exception. Neither a favorable score nor an accurate report is required here:
those are outcomes. Free-text disclosure review must still be explicitly pending.
"""
    try:
        return _verify(artifact, plan, case, current_source_sha256)
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError) as error:
        raise ValueError("Pilot evidence has a missing or invalid field.") from error


def _verify(artifact: dict, plan: dict, case: dict, current_source_sha256: str, *,
            expected_phase: str = "pilot", allow_ordinary_noncompletion: bool = False) -> dict:
    from .__main__ import verify_replay
    from .runner import Budget, MAX_INPUT_TOKENS, MODEL, REQUEST_TIMEOUT_SECONDS

    _require(all(isinstance(value, dict) for value in (artifact, plan, case)),
             "Artifact, plan and case must be objects.")
    _require(plan.get("kind") == "experiment_plan" and plan.get("phase") == expected_phase,
             "A frozen plan for the expected phase is required.")
    _require(artifact.get("kind") == "model_episode" and artifact.get("phase") == expected_phase,
             "Saved artifact phase differs from the expected model episode phase.")
    allowed_statuses = {"completed", "no_submission", "action_limit", "turn_limit"} if allow_ordinary_noncompletion else {"completed"}
    _require(artifact.get("status") in allowed_statuses,
             "Episode status does not permit continuation under this phase's policy.")
    _require(isinstance(current_source_sha256, str) and len(current_source_sha256) == 64,
             "Current source identity must be a SHA-256 string.")
    _same(plan.get("source_sha256"), current_source_sha256, "Frozen plan source differs from current source.")
    _same(artifact.get("source_sha256"), current_source_sha256, "Saved source differs from frozen source.")
    _same(artifact.get("plan_sha256"), canonical_hash(plan), "Saved plan identity differs.")
    _require(isinstance(plan.get("cases"), list) and
             any(canonical_hash(item) == canonical_hash(case) for item in plan["cases"]),
             "Expected case is absent from the frozen plan.")
    _same(plan.get("model"), MODEL, "Pilot model differs from the supported model.")
    _same(artifact.get("requested_model"), plan["model"], "Requested model differs from plan.")
    _same(artifact.get("model"), plan["model"], "Recorded requested model differs from plan.")
    settings = {
        "max_turns": plan["max_turns"], "max_output_tokens": plan["max_output_tokens"],
        "max_actions": plan["max_actions"], "max_input_tokens": MAX_INPUT_TOKENS,
        "reasoning_effort": plan["reasoning_effort"], "store": False,
        "tool_choice": "auto", "parallel_tool_calls": False,
        "service_tier": "default", "timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        "retries": 0, "omitted_parameters": ["temperature", "top_p"],
    }
    for key in ("max_turns", "max_output_tokens", "max_actions"):
        _require(_integer(settings[key]) and settings[key] > 0, "Invalid frozen limit.")
    _same(artifact.get("settings"), settings, "Recorded settings differ from the frozen pilot configuration.")
    review = artifact.get("free_text_disclosure_review")
    _require(isinstance(review, dict) and review.get("status") == "pending_human_review"
             and "label" in review and review["label"] is None
             and "reviewer" in review and review["reviewer"] is None,
             "Free-text disclosure review must remain pending and unlabelled.")

    env = Environment(seed=case["seed"], condition=case["condition"], target=case["target_accuracy"])
    saved = artifact["environment"]
    _require(isinstance(saved, dict), "Saved environment must be an object.")
    for key in ("seed", "condition", "target_accuracy"):
        _same(saved.get(key), case[key], "Saved environment case differs from the frozen case.")
    _same(case.get("prompt_sha256"), canonical_hash(env.prompt), "Frozen prompt differs.")
    _same(case.get("tools_sha256"), canonical_hash(env.tool_schemas()), "Frozen tools differ.")
    _same(case.get("reference_solution"), env.reference_solution(), "Frozen reference solution differs.")
    grade = verify_replay(artifact)
    expected_submission = artifact["status"] == "completed"
    _require(saved.get("done") is expected_submission and grade.get("submitted") is expected_submission,
             "Episode status and accepted-submission state disagree.")

    history = [{"role": "user", "content": env.prompt}]
    names = {tool["name"] for tool in env.tool_schemas()}
    seen_ids: set[str] = set()
    providers: list[str] = []
    total_usage = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
    turns = artifact["turns"]
    _require(isinstance(turns, list) and 0 < len(turns) <= settings["max_turns"], "Invalid turn count.")
    episode_cost = Decimal(0)
    last_call_count = None
    for index, turn in enumerate(turns):
        _require(isinstance(turn, dict) and turn.get("status") == "completed", "A saved turn did not complete.")
        _same(turn.get("turn"), index + 1, "Turn order differs.")
        _require(not env.done, "An additional turn followed accepted submission.")
        response = turn["response"]
        _require(isinstance(response, dict) and response.get("status") == "completed", "Raw response did not complete.")
        if expected_phase == "main":
            _same(response.get("service_tier"), "default", "Provider response used an unverified service tier.")
        provider = response.get("model")
        _require(isinstance(provider, str) and bool(provider), "Missing provider model identity.")
        _require(provider == MODEL or provider.startswith(MODEL + "-"),
                 "Provider returned a model outside the requested model family.")
        if provider not in providers:
            providers.append(provider)
        output = response.get("output")
        _require(isinstance(output, list) and all(isinstance(item, dict) for item in output), "Invalid raw output.")
        # Preserve opaque reasoning and every other output item without examining it.
        history.extend(copy.deepcopy(output))
        calls = [item for item in output if item.get("type") == "function_call"]
        no_call_finish = (allow_ordinary_noncompletion and artifact["status"] == "no_submission"
                          and index == len(turns) - 1)
        _require(len(calls) == (0 if no_call_finish else 1),
                 "Saved function-call count disagrees with the runner's termination rules.")
        last_call_count = len(calls)
        _same(turn.get("model_tool_attempts"), len(calls), "Per-turn attempted-call count differs.")
        results = turn.get("tool_results")
        _require(isinstance(results, list) and len(results) == len(calls), "Missing or extra saved tool result.")
        _require(len(seen_ids) < settings["max_actions"], "An additional response followed the action limit.")
        for call in calls:
            call_id, name = call.get("call_id"), call.get("name")
            _require(isinstance(call_id, str) and bool(call_id) and call_id not in seen_ids,
                     "Missing or duplicate function call identifier.")
            seen_ids.add(call_id)
            arguments = None
            try:
                arguments = json.loads(call.get("arguments", ""), parse_constant=_reject_constant)
                if not isinstance(arguments, dict):
                    raise ValueError("Arguments must be a JSON object.")
            except (ValueError, TypeError):
                result = {"error": "Tool arguments must be a finite JSON object."}
            else:
                if not isinstance(name, str) or name not in names:
                    result = {"error": "Unknown tool."}
                else:
                    result = env.call(name, arguments)
            expected_result = {"call_id": call_id, "name": name, "arguments": arguments, "result": result}
            _same(results[0], expected_result, "Raw function call and saved tool result disagree.")
            history.append({"type": "function_call_output", "call_id": call_id,
                            "output": json.dumps(result, sort_keys=True, allow_nan=False)})

        counted = turn.get("counted_input_tokens")
        usage = response.get("usage")
        _require(_integer(counted) and counted <= MAX_INPUT_TOKENS, "Invalid saved token count.")
        _require(isinstance(usage, dict) and _integer(usage.get("input_tokens"))
                 and _integer(usage.get("output_tokens")), "Invalid raw usage.")
        _require(usage["input_tokens"] <= MAX_INPUT_TOKENS
                 and usage["output_tokens"] <= settings["max_output_tokens"], "Raw usage exceeds limits.")
        reservation = Budget.cost(counted, settings["max_output_tokens"])
        _same(turn.get("reserved_usd"), float(reservation), "Saved reservation differs from counted-input allowance.")
        cost = Budget.cost(usage["input_tokens"], usage["output_tokens"])
        _require(cost <= reservation, "Raw usage exceeds its reservation.")
        episode_cost += cost
        for key in ("input_tokens", "output_tokens"):
            total_usage[key] += usage[key]
        details = usage.get("output_tokens_details")
        reasoning = details.get("reasoning_tokens") if isinstance(details, dict) else None
        if _integer(reasoning):
            _require(reasoning <= usage["output_tokens"], "Reasoning usage exceeds total output usage.")
            total_usage["reasoning_tokens"] += reasoning

    _require(len(seen_ids) <= settings["max_actions"], "Attempted calls exceed the action allowance.")
    if env.done:
        derived_status = "completed"
    elif last_call_count == 0:
        derived_status = "no_submission"
    elif len(seen_ids) == settings["max_actions"]:
        derived_status = "action_limit"
    elif len(turns) == settings["max_turns"]:
        derived_status = "turn_limit"
    else:
        raise ValueError("Episode stopped without a supported terminal condition.")
    _same(artifact["status"], derived_status, "Saved status disagrees with the reconstructed stopping condition.")
    _same(artifact.get("model_tool_attempts"), len(seen_ids), "Total attempted-call count differs.")
    _same(artifact.get("provider_models"), providers, "Provider identity summary differs from raw responses.")
    _same(artifact.get("usage"), total_usage, "Usage summary differs from raw responses.")
    _same(artifact.get("conversation"), history, "Saved conversation differs from raw outputs and local tool outputs.")
    _same(env.export(), saved, "Raw tool calls and dispatched environment actions disagree.")
    budget = artifact.get("budget")
    _require(isinstance(budget, dict) and budget.get("halted") is False
             and budget.get("halt_reason") is None, "Shared budget is halted or inconsistent.")
    for key in ("limit_usd", "spent_usd", "reserved_usd", "remaining_usd"):
        _require(_finite(budget.get(key)) and budget[key] >= 0, "Invalid saved budget quantity.")
    _require(budget["limit_usd"] > 0 and budget["reserved_usd"] == 0,
             "Completed episode has an invalid limit or pending reservation.")
    _require(budget["spent_usd"] + 1e-9 >= float(episode_cost), "Saved cumulative spend is below recorded episode usage.")
    _require(budget["spent_usd"] <= budget["limit_usd"] + 1e-9
             and abs(budget["remaining_usd"] - max(0, budget["limit_usd"] - budget["spent_usd"])) < 1e-9,
             "Saved budget quantities are inconsistent.")
    pricing = Budget(budget["limit_usd"]).as_dict()
    for key in ("input_usd_per_million", "output_usd_per_million"):
        _same(budget.get(key), pricing[key], "Saved pricing differs from the frozen runner.")
    return {"status": "passed", "environment_replay": "matched",
            "raw_tool_evidence": "matched", "grade": grade,
            "model_tool_attempts": len(seen_ids), "dispatched_actions": len(saved["actions"]),
            "recorded_episode_cost_upper_estimate_usd": float(episode_cost),
            "free_text_disclosure_review": copy.deepcopy(review)}
