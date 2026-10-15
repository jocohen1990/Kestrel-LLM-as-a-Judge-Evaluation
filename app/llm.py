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
    input_tokens: int
    output_tokens: int
    latency_s: float


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


def warmup(model: str | None = None) -> None:
    """Make one tiny call before measuring anything.

    The first request pays one-time costs (opening the network connection,
    TLS handshake), so in run 1 the first question took 2.4s while the rest
    took under 1.3s. Warming up keeps that setup cost out of the latency stats.
    """
    generate("Reply with OK.", "OK", model=model, max_tokens=5)


def generate(system: str, user: str, model: str | None = None,
             max_tokens: int = 400, temperature: float = 0.0) -> LLMResponse:
    """Send one prompt to the model and return its reply plus metadata.

    temperature=0.0 makes answers as repeatable as possible, which matters
    when you're comparing eval runs. (Even at 0, LLMs aren't perfectly
    deterministic, which is exactly what the Phase 3 consistency tests probe.)
    """
    model = model or os.getenv("LLM_MODEL", DEFAULT_MODEL)
    request = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    # anthropic 1.x removed `temperature` from create(); extra_body still
    # sends it. Some newer models reject it entirely, so for those we retry
    # once without it and remember the model so later calls skip it.
    if model not in _NO_TEMPERATURE_MODELS:
        request["extra_body"] = {"temperature": temperature}

    start = time.perf_counter()
    try:
        response = _get_client().messages.create(**request)
    except Exception as exc:
        if "extra_body" in request and _rejects_temperature(exc):
            _NO_TEMPERATURE_MODELS.add(model)
            request.pop("extra_body")
            start = time.perf_counter()   # don't count the failed attempt
            response = _get_client().messages.create(**request)
        else:
            raise
    latency = time.perf_counter() - start
    text = "".join(block.text for block in response.content if block.type == "text")
    return LLMResponse(
        text=text,
        model=model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        latency_s=latency,
    )
