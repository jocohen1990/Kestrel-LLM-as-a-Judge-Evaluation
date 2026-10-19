"""Thin wrapper around the LLM API.

Keeping every API call in one function means:
  1. Swapping providers or models later only touches this file.
  2. Tests can replace `generate` with a fake (see tests/test_rag.py), so
     most of the suite runs offline and for free.
  3. Latency and token counts are captured on every call, ready for the
     performance and cost metrics in later phases.
"""
import os
import time
from dataclasses import dataclass

# A small, inexpensive model is plenty for Phase 1. Override it with the
# LLM_MODEL environment variable or the --model flag on run_evals.py.
DEFAULT_MODEL = "claude-haiku-4-5-20251001"


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int     # ALL input tokens, cached or not (see _input_usage)
    output_tokens: int
    latency_s: float
    cache_read_input_tokens: int = 0       # part of input_tokens read from the cache
    cache_creation_input_tokens: int = 0   # part of input_tokens written to the cache


def _input_usage(usage) -> dict:
    """Input token counts that stay comparable whether caching is on or off.

    With caching, the API's `input_tokens` counts only the tokens AFTER the
    cache breakpoint; cached tokens are reported separately. Adding them back
    keeps "Tokens in" meaning the same thing in every results file, so a run
    with caching can be compared with Runs 1-7. What caching saves shows up in
    the cache_read field (billed at a fraction of the normal input price).
    """
    read = getattr(usage, "cache_read_input_tokens", 0) or 0
    write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    return {"input_tokens": usage.input_tokens + read + write,
            "cache_read_input_tokens": read,
            "cache_creation_input_tokens": write}


_client = None


def _get_client():
    """Create the API client on first use.

    The import lives inside the function so the offline tests can run even
    if the `anthropic` package or API key isn't available.
    """
    global _client
    if _client is None:
        from anthropic import Anthropic
        _client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    return _client


# Models that answered "temperature is not supported" during this run.
_NO_TEMPERATURE_MODELS: set[str] = set()


def _rejects_temperature(exc: Exception) -> bool:
    """Is this the API saying the model doesn't accept a temperature setting?

    Checked by status code and message rather than by exception class, so
    this file still doesn't need to import `anthropic` at the top.
    """
    status = getattr(exc, "status_code", None)
    return status == 400 and "temperature" in str(exc).lower()


def sends_temperature(model: str) -> bool:
    """Did calls to this model include the temperature setting?

    Only meaningful after at least one call: a model is added to
    _NO_TEMPERATURE_MODELS the first time it rejects the setting. Results
    files record this, because "temperature 0" can't explain consistency for
    a model that never received it.
    """
    return model not in _NO_TEMPERATURE_MODELS


# Models that answered "tool_choice type 'tool' is not supported" this run.
# Found in Run 5, the first API run of the tool-mode judge: the judge model
# (claude-sonnet-5-5) rejects FORCING a tool call, so every judged eval crashed
# on its first call. Those models get tool_choice "auto" instead: the tool is
# still offered (and still strict), the system prompt still says to call it,
# but the model is no longer forced to. If it replies without calling the tool,
# generate_structured raises StructuredOutputError and the judge retries.
_NO_FORCED_TOOL_MODELS: set[str] = set()


def _rejects_forced_tool_choice(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    return status == 400 and "tool_choice" in str(exc).lower()


def forces_tool_choice(model: str) -> bool:
    """Were structured calls to this model sent with a forced tool_choice?

    Like sends_temperature, only meaningful after at least one call. Results
    files record it, because "the API guarantees a tool call" is only true
    when the call was forced.
    """
    return model not in _NO_FORCED_TOOL_MODELS


def warmup(model: str | None = None) -> None:
    """Make one tiny call before measuring anything.

    The first request pays one-time costs (opening the network connection,
    TLS handshake), so in run 1 the first question took 2.4s while the rest
    took under 1.3s. Warming up keeps that setup cost out of the latency stats.
    """
    generate("Reply with OK.", "OK", model=model, max_tokens=5)


def _relax_forced_tool(body: dict) -> None:
    body["tool_choice"] = {"type": "auto"}


def _is_forced(body: dict) -> bool:
    return body.get("tool_choice", {}).get("type") in ("tool", "any")


def _create(request: dict, model: str, temperature: float, extra: dict | None = None):
    """Send one request, handling settings some models reject.

    Two settings get a fallback, each tried at most once per call and then
    remembered per model so later calls don't waste a request:
      * temperature: anthropic 1.x removed it from create(); extra_body still
        sends it. Some newer models reject it entirely, so it's dropped.
      * a forced tool_choice: some models reject forcing a tool call, so it
        becomes "auto" (see _NO_FORCED_TOOL_MODELS).
    Any other error is raised. Everything else in `extra` is always sent.

    Returns (response, latency in seconds).
    """
    body = dict(extra or {})
    if model not in _NO_TEMPERATURE_MODELS:
        body["temperature"] = temperature
    if model in _NO_FORCED_TOOL_MODELS and _is_forced(body):
        _relax_forced_tool(body)
    # One try, plus at most one retry per fallback.
    for _ in range(3):
        start = time.perf_counter()   # a failed attempt's time isn't counted
        try:
            response = _get_client().messages.create(**request, extra_body=body)
            return response, time.perf_counter() - start
        except Exception as exc:
            if "temperature" in body and _rejects_temperature(exc):
                _NO_TEMPERATURE_MODELS.add(model)
                body.pop("temperature")
            elif _is_forced(body) and _rejects_forced_tool_choice(exc):
                _NO_FORCED_TOOL_MODELS.add(model)
                _relax_forced_tool(body)
            else:
                raise
    raise RuntimeError("unreachable: each fallback is applied at most once")


def _base_request(system: str, user: str, model: str, max_tokens: int,
                  cache_system: bool = False) -> dict:
    """cache_system marks the system prompt (and the tools, which come before
    it) as a cacheable prefix. A model only caches a prefix above its minimum
    length (4,096 tokens for Claude Haiku 4.5, 512 for Claude Sonnet 5.5);
    shorter prefixes are sent normally, with no error, so it's always safe."""
    system_field = ([{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
                    if cache_system else system)
    return {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_field,
        "messages": [{"role": "user", "content": user}],
    }


def generate(system: str, user: str, model: str | None = None,
             max_tokens: int = 400, temperature: float = 0.0,
             cache_system: bool = False) -> LLMResponse:
    """Send one prompt to the model and return its reply plus metadata.

    temperature=0.0 makes answers as repeatable as possible, which matters
    when you're comparing eval runs. (Even at 0, LLMs aren't perfectly
    deterministic, which is exactly what the Phase 3 consistency tests probe.)
    """
    model = model or os.getenv("LLM_MODEL", DEFAULT_MODEL)
    response, latency = _create(_base_request(system, user, model, max_tokens, cache_system),
                                model, temperature)
    text = "".join(block.text for block in response.content if block.type == "text")
    return LLMResponse(
        text=text,
        model=model,
        output_tokens=response.usage.output_tokens,
        latency_s=latency,
        **_input_usage(response.usage),
    )


class StructuredOutputError(RuntimeError):
    """The model didn't return the forced tool call (e.g. it hit max_tokens)."""


@dataclass
class StructuredResponse:
    data: dict            # the tool call's input, already parsed from JSON
    model: str
    input_tokens: int     # ALL input tokens, cached or not (see _input_usage)
    output_tokens: int
    latency_s: float
    stop_reason: str | None
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


def generate_structured(system: str, user: str, schema: dict, tool_name: str,
                        tool_description: str, model: str | None = None,
                        max_tokens: int = 1200, temperature: float = 0.0,
                        cache_system: bool = False) -> StructuredResponse:
    """Get a reply that is GUARANTEED to match a JSON schema.

    Asking for "only a JSON object" in the prompt is a request, not a
    guarantee: in Run 4 the judge broke its JSON on 4 of 160 calls. Instead,
    this defines one tool whose input is the schema, sets `strict: true` (the
    API makes the input match the schema exactly) and `tool_choice` (the
    model must call that tool). The reply arrives as a parsed dict, so there
    is no JSON text for us to parse at all.

    Tools are sent through extra_body, like temperature, so this works
    whichever create() parameters the installed SDK version declares.
    """
    model = model or os.getenv("LLM_MODEL", DEFAULT_MODEL)
    tools = {
        "tools": [{
            "name": tool_name,
            "description": tool_description,
            "strict": True,
            "input_schema": schema,
        }],
        "tool_choice": {"type": "tool", "name": tool_name},
    }
    response, latency = _create(_base_request(system, user, model, max_tokens, cache_system),
                                model, temperature, extra=tools)
    for block in response.content:
        if block.type == "tool_use" and block.name == tool_name:
            return StructuredResponse(
                data=dict(block.input),
                model=model,
                output_tokens=response.usage.output_tokens,
                latency_s=latency,
                stop_reason=getattr(response, "stop_reason", None),
                **_input_usage(response.usage),
            )
    # Strict mode guarantees the SHAPE of a tool call, not that one is made:
    # a reply cut off by max_tokens, or a refusal, can still arrive without it.
    raise StructuredOutputError(
        f"no {tool_name!r} tool call in reply (stop_reason={getattr(response, 'stop_reason', None)!r})")
