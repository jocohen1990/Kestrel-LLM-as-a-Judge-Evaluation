"""Offline tests for the judge: prompt building and reply parsing.

The judge's reply is untrusted text from a model, so the parser is tested
against the messy shapes real replies come in.
"""
import pytest

from app.llm import LLMResponse
from evals import judge
from evals.judge import JudgeError, build_judge_prompt, parse_judge_output

GOOD_REPLY = """{"claims": [{"claim": "Revenue was $108.3M", "supported": true},
                            {"claim": "Kestrel split its stock", "supported": false}],
                 "faithful": false, "correct": true, "reasoning": "One claim is unsupported."}"""


def test_parses_clean_json():
    verdict = parse_judge_output(GOOD_REPLY)
    assert verdict.correct is True
    assert verdict.faithful is False
    assert verdict.unsupported_claims == ["Kestrel split its stock"]


def test_parses_json_wrapped_in_code_fences_and_chatter():
    reply = "Here is my evaluation:\n```json\n" + GOOD_REPLY + "\n```"
    assert parse_judge_output(reply).faithful is False


@pytest.mark.parametrize("reply, message", [
    ("I think the answer is fine.", "no JSON"),
    ('{"faithful": true, "correct": }', "invalid JSON"),
    ('{"faithful": "yes", "correct": true}', "faithful"),   # string, not a boolean
    ('{"faithful": true}', "correct"),                      # missing field
])
def test_rejects_malformed_replies(reply, message):
    with pytest.raises(JudgeError, match=message):
        parse_judge_output(reply)


def test_prompt_contains_everything_the_judge_needs():
    prompt = build_judge_prompt("Q?", ["passage one", "passage two"], "REF", "ANS")
    for piece in ("Q?", "[1] passage one", "[2] passage two", "REF", "ANS"):
        assert piece in prompt


def test_judge_answer_uses_judge_model_and_parses_reply(monkeypatch):
    captured = {}

    def fake_generate(system, user, model=None, **kwargs):
        captured["model"] = model
        captured["system"] = system
        return LLMResponse(text=GOOD_REPLY, model=model, input_tokens=1,
                           output_tokens=1, latency_s=0.0)

    monkeypatch.setattr(judge, "generate", fake_generate)
    verdict = judge.judge_answer("Q?", ["ctx"], "REF", "ANS", model="judge-x")
    assert captured["model"] == "judge-x"
    assert "faithful" in captured["system"]
    assert verdict.correct is True


def fake_replies(*replies):
    """A fake generate() that returns the given replies in order."""
    calls = []

    def _fake(system, user, model=None, **kwargs):
        calls.append(user)
        return LLMResponse(text=replies[len(calls) - 1], model=model, input_tokens=1,
                           output_tokens=1, latency_s=0.0)
    return _fake, calls


def test_judge_retries_once_after_a_broken_reply(monkeypatch):
    fake, calls = fake_replies('{"faithful": true, "correct": "unfinished', GOOD_REPLY)
    monkeypatch.setattr(judge, "generate", fake)
    verdict = judge.judge_answer("Q?", ["ctx"], "REF", "ANS")
    assert len(calls) == 2
    assert verdict.correct is True


def test_judge_gives_up_after_two_broken_replies(monkeypatch):
    fake, calls = fake_replies("not json", "still not json")
    monkeypatch.setattr(judge, "generate", fake)
    with pytest.raises(JudgeError, match="after 2 attempts"):
        judge.judge_answer("Q?", ["ctx"], "REF", "ANS")
    assert len(calls) == 2
