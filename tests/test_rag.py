"""Test the RAG pipeline's wiring without calling a real model.

`monkeypatch` (a built-in pytest fixture) temporarily swaps a function for
a fake one. We patch `app.rag.generate`, not `app.llm.generate`, because
you patch a name where it's USED, and rag.py imported it into its own
namespace. That's one of the most common mocking mistakes in Python.
"""
from app import rag
from app.llm import LLMResponse


def fake_generate(captured):
    def _fake(system, user, model=None, **kwargs):
        captured["system"] = system
        captured["user"] = user
        return LLMResponse(text="  Revenue was $108.3 million.  ", model="fake-model",
                           input_tokens=100, output_tokens=10, latency_s=0.01)
    return _fake


def test_answer_sends_retrieved_context_and_question(monkeypatch):
    captured = {}
    monkeypatch.setattr(rag, "generate", fake_generate(captured))

    result = rag.RagAssistant(k=3).answer("What was Kestrel's revenue in Q1 2026?")

    assert "108.3" in captured["user"]                       # right passage made it in
    assert "Question: What was Kestrel's revenue" in captured["user"]
    assert rag.REFUSAL in captured["system"]                 # refusal rule is in the prompt
    assert result.answer == "Revenue was $108.3 million."    # whitespace trimmed
    assert "q1_2026_earnings.md" in result.sources
    assert len(result.contexts) == 3


# ---- Prompt versions ------------------------------------------------------

import re  # noqa: E402

import pytest  # noqa: E402


def capture_prompt(monkeypatch, version):
    captured = {}
    monkeypatch.setattr(rag, "generate", fake_generate(captured))
    rag.RagAssistant(k=3, prompt_version=version).answer("What was Kestrel's revenue in Q1 2026?")
    return captured


def test_v1_prompt_is_unchanged(monkeypatch):
    """Old prompt versions are frozen so past runs stay reproducible."""
    captured = capture_prompt(monkeypatch, "v1")
    assert captured["user"].startswith("Context passages:\n[1] (source: ")
    assert "using ONLY the context passages provided" in captured["system"]


def test_v2_never_shows_the_model_words_it_shouldnt_repeat(monkeypatch):
    user = capture_prompt(monkeypatch, "v2")["user"]
    assert user.startswith("Reference material:\n")
    assert not re.search(r"context|passage|\.md\b|\[\d\]", user, re.IGNORECASE)
    assert "Kestrel Freight Analytics Q1 2026 Earnings Release" in user   # titles instead


def test_v2_system_prompt_covers_each_run2_failure():
    system = rag.PROMPTS["v2"]["system"]
    assert rag.REFUSAL in system
    assert "Never put it before an answer" in system         # refusal used as an opener
    assert "Never mention" in system                          # "according to the context"
    assert "premise" in system and "period" in system         # false premise, ambiguity


def test_unknown_prompt_version_is_rejected():
    with pytest.raises(ValueError, match="unknown prompt version"):
        rag.RagAssistant(prompt_version="v99")
