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


def generate(system: str, user: str, model: str | None = None,
             max_tokens: int = 400, temperature: float = 0.0) -> LLMResponse:
    """Send one prompt to the model and return its reply plus metadata.

    temperature=0.0 makes answers as repeatable as possible, which matters
    when you're comparing eval runs. (Even at 0, LLMs aren't perfectly
    deterministic, which is exactly what the Phase 3 consistency tests probe.)
    """
    model = model or os.getenv("LLM_MODEL", DEFAULT_MODEL)
    start = time.perf_counter()
    response = _get_client().messages.create(
        model=model,
        max_tokens=max_tokens,
        # anthropic 1.x removed `temperature` from create(); extra_body still
        # sends it. Newer models (Opus 4.7+) reject it, so drop it for those.
        extra_body={"temperature": temperature},
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    latency = time.perf_counter() - start
    text = "".join(block.text for block in response.content if block.type == "text")
    return LLMResponse(
        text=text,
        model=model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        latency_s=latency,
    )
