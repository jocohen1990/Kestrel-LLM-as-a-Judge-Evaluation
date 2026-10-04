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
