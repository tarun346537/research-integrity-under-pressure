"""Bounded, auditable Responses API loop. Importing this module makes no requests.

The token-count endpoint is called before each inference request. Reservations
and recorded-usage estimates apply Astra's highest standard short-context input
category ($12.50/M cache-write tokens) to ALL input tokens, plus $50/M output.
This deliberately conservative estimate is not an invoice. Reservations include
the entire output allowance, including reasoning tokens. This local spending
guard is not a provider billing limit. Ambiguous requests halt the shared budget.
"""
from __future__ import annotations

import copy
import json
import math
import urllib.error
import urllib.request
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable


MODEL = "gpt-6-astra"
MAX_INPUT_TOKENS = 272_000
REQUEST_TIMEOUT_SECONDS = 120
INPUT_USD_PER_MILLION = Decimal("12.50")
OUTPUT_USD_PER_MILLION = Decimal("50")
ALLOWED_TOOLS = {"list_files", "read_file", "write_file", "evaluate", "submit"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


class Budget:
    """Shared across episodes; uncertainty permanently prevents further spend."""

    def __init__(self, limit_usd: float):
        if not math.isfinite(float(limit_usd)) or limit_usd <= 0:
            raise ValueError("Budget must be a positive finite dollar amount.")
        self._limit = Decimal(str(limit_usd))
        self._spent = Decimal(0)
        self._reservations: dict[int, Decimal] = {}
        self._next = 0
        self.halted = False
        self.halt_reason: str | None = None

    @staticmethod
    def cost(input_tokens: int, output_tokens: int) -> Decimal:
        return (Decimal(input_tokens) * INPUT_USD_PER_MILLION
                + Decimal(output_tokens) * OUTPUT_USD_PER_MILLION) / 1_000_000

    @property
    def spent_usd(self) -> float:
        return float(self._spent)

    @property
    def reserved_usd(self) -> float:
        return float(sum(self._reservations.values(), Decimal(0)))

    @property
    def remaining_usd(self) -> float:
        return float(max(Decimal(0), self._limit - self._spent - sum(self._reservations.values(), Decimal(0))))

    def halt(self, reason: str) -> None:
        self.halted = True
        self.halt_reason = reason

    def reserve(self, input_tokens: int, max_output_tokens: int) -> int | None:
        amount = self.cost(input_tokens, max_output_tokens)
        committed = self._spent + sum(self._reservations.values(), Decimal(0))
        if self.halted or committed + amount > self._limit:
            return None
        self._next += 1
        self._reservations[self._next] = amount
        return self._next

    def settle(self, reservation: int, input_tokens: int, output_tokens: int) -> bool:
        amount = self.cost(input_tokens, output_tokens)
        if amount > self._reservations[reservation]:
            # Do not discard evidence of a provider/accounting discrepancy.
            self._reservations[reservation] = amount
            self.halt("Recorded-usage upper estimate exceeded the reserved maximum.")
            return False
        del self._reservations[reservation]
        self._spent += amount
        return True

    def as_dict(self) -> dict:
        return {
            "limit_usd": float(self._limit), "spent_usd": self.spent_usd,
            "reserved_usd": self.reserved_usd, "remaining_usd": self.remaining_usd,
            "halted": self.halted, "halt_reason": self.halt_reason,
            "input_usd_per_million": float(INPUT_USD_PER_MILLION),
            "output_usd_per_million": float(OUTPUT_USD_PER_MILLION),
            "pricing_basis": "Conservative upper estimate: all input at the $12.50/M cache-write category; output $50/M; standard short-context service. Not an invoice.",
            "spent_usd_basis": "Upper estimate from recorded usage, without cache-category discounts; not actual invoiced spend.",
        }


class APIError(Exception):
    """Contains only locally generated safe messages, never provider bodies."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward a bearer token to a redirect destination.
        return None


def api_transport(path: str, payload: dict, api_key: str) -> dict:
    """One request, no retries; errors intentionally exclude headers and bodies."""
    if path not in {"/responses", "/responses/input_tokens"}:
        raise APIError("Unrecognized API endpoint.")
    request = urllib.request.Request(
        "https://api.openai.com/v1" + path,
        data=json.dumps(payload, allow_nan=False).encode("utf-8"), method="POST",
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.build_opener(_NoRedirect).open(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            result = json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise APIError(f"OpenAI API returned HTTP {error.code}; response body omitted.") from None
    except Exception:
        raise APIError("OpenAI request failed or returned invalid JSON; details omitted.") from None
    if not isinstance(result, dict):
        raise APIError("OpenAI response was not a JSON object.")
    return result


def _redact(value: Any, secret: str) -> Any:
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]") if secret else value
    if isinstance(value, list):
        return [_redact(item, secret) for item in value]
    if isinstance(value, dict):
        return {_redact(key, secret): _redact(item, secret) for key, item in value.items()}
    return value


def _reject_constant(value: str) -> None:
    raise ValueError("Non-finite JSON number.")


def run_episode(
    env: Any, *, api_key: str, budget: Budget, model: str = MODEL,
    max_turns: int = 15, max_output_tokens: int = 4000, max_actions: int = 15,
    reasoning_effort: str = "medium", transport: Callable = api_transport,
    on_progress: Callable[[dict], None] | None = None,
) -> dict:
    """Run a fresh provider context and return a replayable local artifact.

    ``transport(path, payload, api_key)`` is injectable for fully offline tests.
    ``on_progress`` may atomically persist a snapshot before/after each request.
    Any unexpected persistence failure propagates; no subsequent API call occurs.
    """
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("OPENAI_API_KEY is required; never paste keys into artifacts.")
    if model != MODEL:
        raise ValueError("Only gpt-6-astra is supported by this verified pricing guard.")
    if any(not _integer(value) or value == 0 for value in (max_turns, max_output_tokens, max_actions)):
        raise ValueError("Turn, action, and output-token limits must be positive integers.")
    if reasoning_effort not in {"low", "medium", "high", "xhigh", "max"}:
        raise ValueError("Unsupported reasoning effort.")
    tools = copy.deepcopy(env.tool_schemas())
    if any(tool.get("type") != "function" or tool.get("name") not in ALLOWED_TOOLS for tool in tools):
        raise ValueError("Only the five local benchmark tools are allowed.")
    names = {tool["name"] for tool in tools}
    history: list[dict] = [{"role": "user", "content": env.prompt}]
    artifact = {
        "status": "started", "started_at": _now(), "finished_at": None,
        "requested_model": model, "model": model, "provider_models": [],
        "settings": {"max_turns": max_turns, "max_output_tokens": max_output_tokens,
                     "max_actions": max_actions, "max_input_tokens": MAX_INPUT_TOKENS,
                     "reasoning_effort": reasoning_effort, "store": False,
                     "tool_choice": "auto", "parallel_tool_calls": False,
                     "service_tier": "default", "timeout_seconds": REQUEST_TIMEOUT_SECONDS,
                     "retries": 0, "omitted_parameters": ["temperature", "top_p"]},
        "model_tool_attempts": 0,
        "turns": [], "usage": {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0},
    }

    def snapshot(status: str | None = None) -> dict:
        if status is not None:
            artifact["status"] = status
        artifact.update({"conversation": copy.deepcopy(history), "budget": budget.as_dict(),
                         "environment": env.export(), "grade": env.grade()})
        return _redact(copy.deepcopy(artifact), api_key)

    def progress() -> None:
        if on_progress is not None:
            on_progress(snapshot())

    def finish(status: str) -> dict:
        artifact["finished_at"] = _now()
        result = snapshot(status)
        if on_progress is not None:
            on_progress(result)
        return result

    if budget.halted:
        return finish("budget_exhausted")

    try:
        for turn_index in range(max_turns):
            if artifact["model_tool_attempts"] >= max_actions:
                return finish("action_limit")
            record: dict = {"turn": turn_index + 1, "started_at": _now(), "status": "counting", "tool_results": []}
            artifact["turns"].append(record)
            payload = {
                "model": model, "input": copy.deepcopy(history), "tools": tools,
                "tool_choice": "auto", "parallel_tool_calls": False,
                "reasoning": {"effort": reasoning_effort},
            }
            progress()
            try:
                count = transport("/responses/input_tokens", copy.deepcopy(payload), api_key)
            except Exception:
                record["status"] = "token_count_failed"
                record["error"] = "Token count request failed; no inference request sent. Error details omitted."
                budget.halt("Token counting failed; stopping without retries.")
                return finish("api_error")
            input_tokens = count.get("input_tokens") if isinstance(count, dict) else None
            if not _integer(input_tokens) or input_tokens > MAX_INPUT_TOKENS:
                record["status"] = "invalid_token_count"
                budget.halt("Token count missing, invalid, or outside verified pricing context.")
                return finish("accounting_error")
            record["counted_input_tokens"] = input_tokens
            reservation = budget.reserve(input_tokens, max_output_tokens)
            if reservation is None:
                record["status"] = "budget_rejected"
                return finish("budget_exhausted")
            record["reserved_usd"] = float(Budget.cost(input_tokens, max_output_tokens))
            record["status"] = "request_pending"
            progress()
            payload.update({"store": False, "include": ["reasoning.encrypted_content"],
                            "max_output_tokens": max_output_tokens, "service_tier": "default"})
            try:
                response = transport("/responses", payload, api_key)
            except Exception:
                record["status"] = "api_error"
                record["error"] = "Inference request failed; billing may have occurred. Error details omitted."
                budget.halt("Ambiguous inference failure; maximum reservation retained, no retries.")
                return finish("api_error")
            record["finished_at"] = _now()
            if not isinstance(response, dict):
                record["status"] = "invalid_response"
                budget.halt("Invalid response; maximum reservation retained.")
                return finish("api_error")
            # Preserve model output verbatim, including opaque reasoning and phase.
            record["response"] = copy.deepcopy(response)
            provider_model = response.get("model")
            if isinstance(provider_model, str) and provider_model not in artifact["provider_models"]:
                artifact["provider_models"].append(provider_model)
            output = response.get("output")
            if isinstance(output, list) and all(isinstance(item, dict) for item in output):
                history.extend(copy.deepcopy(output))
            else:
                record["status"] = "invalid_output"
                budget.halt("Invalid output; maximum reservation retained.")
                return finish("api_error")
            calls = [item for item in output if item.get("type") == "function_call"]
            record["model_tool_attempts"] = len(calls)
            artifact["model_tool_attempts"] += len(calls)
            usage = response.get("usage")
            valid_usage = isinstance(usage, dict) and _integer(usage.get("input_tokens")) and _integer(usage.get("output_tokens"))
            if valid_usage:
                for key in ("input_tokens", "output_tokens"):
                    artifact["usage"][key] += usage[key]
                details = usage.get("output_tokens_details")
                reasoning = details.get("reasoning_tokens") if isinstance(details, dict) else None
                if _integer(reasoning):
                    artifact["usage"]["reasoning_tokens"] += reasoning
            if response.get("status") != "completed":
                if len(calls) > 1:
                    record["status"] = "integration_error"
                    record["error"] = "Provider returned multiple function calls despite parallel_tool_calls=false; none executed."
                    budget.halt("Unexpected multiple calls in an unfinished response; maximum reservation retained.")
                    return finish("integration_error")
                record["status"] = "incomplete" if response.get("status") == "incomplete" else "api_error"
                budget.halt("Response did not complete; maximum reservation retained conservatively.")
                return finish(record["status"])
            if not valid_usage or usage["input_tokens"] > MAX_INPUT_TOKENS or usage["output_tokens"] > max_output_tokens:
                record["status"] = "invalid_usage"
                budget.halt("Usage missing, invalid, or outside the reservation assumptions; reservation retained.")
                return finish("accounting_error")
            if not budget.settle(reservation, usage["input_tokens"], usage["output_tokens"]):
                record["status"] = "reservation_exceeded"
                return finish("accounting_error")
            record["status"] = "completed"
            if len(calls) > 1:
                record["status"] = "integration_error"
                record["error"] = "Provider returned multiple function calls despite parallel_tool_calls=false; none executed."
                budget.halt("Unexpected multiple tool calls invalidate continued measurement.")
                return finish("integration_error")
            for call in calls:
                call_id, name = call.get("call_id"), call.get("name")
                if not isinstance(call_id, str) or not call_id:
                    record["status"] = "invalid_tool_call"
                    budget.halt("Provider returned a tool call without a valid call identifier.")
                    return finish("api_error")
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
                    elif env.done:
                        result = {"error": "Episode already submitted."}
                    else:
                        try:
                            result = env.call(name, arguments)
                        except Exception:
                            result = {"error": "Local tool failed; exception details omitted."}
                            record["tool_results"].append({"call_id": call_id, "name": name, "arguments": arguments, "result": result})
                            record["status"] = "local_tool_error"
                            budget.halt("Local tool failure invalidates continued measurement.")
                            return finish("local_tool_error")
                tool_record = {"call_id": call_id, "name": name, "arguments": arguments, "result": result}
                record["tool_results"].append(tool_record)
                history.append({"type": "function_call_output", "call_id": call_id,
                                "output": json.dumps(result, sort_keys=True, allow_nan=False)})
                progress()
            progress()
            if env.done:
                return finish("completed")
            if not calls:
                return finish("no_submission")
            if artifact["model_tool_attempts"] >= max_actions:
                return finish("action_limit")
        return finish("turn_limit")
    except KeyboardInterrupt:
        budget.halt("Interrupted; any pending request reservation retained.")
        if artifact["turns"]:
            artifact["turns"][-1]["status"] = "interrupted"
        return finish("interrupted")
