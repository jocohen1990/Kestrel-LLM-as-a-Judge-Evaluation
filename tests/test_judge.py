"""Offline tests for the judge: prompt building and reply parsing.

The judge's reply is untrusted text from a model, so the parser is tested
against the messy shapes real replies come in.
"""
import pytest

from app.llm import LLMResponse, StructuredOutputError, StructuredResponse
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
    verdict = judge.judge_answer("Q?", ["ctx"], "REF", "ANS", model="judge-x", output_mode="text")
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
    verdict = judge.judge_answer("Q?", ["ctx"], "REF", "ANS", output_mode="text")
    assert len(calls) == 2
    assert verdict.correct is True


def test_judge_gives_up_after_two_broken_replies(monkeypatch):
    fake, calls = fake_replies("not json", "still not json")
    monkeypatch.setattr(judge, "generate", fake)
    with pytest.raises(JudgeError, match="after 2 attempts"):
        judge.judge_answer("Q?", ["ctx"], "REF", "ANS", output_mode="text")
    assert len(calls) == 2


# ---- Judge prompt versions ----------------------------------------------

def test_judge_v1_is_frozen_and_v2_adds_the_forecast_rule():
    v1, v2 = judge.JUDGE_RULES["judge-v1"], judge.JUDGE_RULES["judge-v2"]
    assert "guidance" not in v1.lower()
    assert "NOT actual results" in v2
    # v2 still allows guidance when the question asks for it (g009, g013).
    assert "asks for guidance" in v2
    assert v2.startswith(v1)   # v2 is v1 plus the new rule


def test_defaults_are_v2_rules_with_tool_output():
    assert judge.DEFAULT_JUDGE_PROMPT == "judge-v2"
    assert judge.DEFAULT_OUTPUT_MODE == "tool"


def test_system_prompt_matches_the_output_mode():
    text = judge.judge_system_prompt("judge-v2", "text")
    tool = judge.judge_system_prompt("judge-v2", "tool")
    assert "Respond with ONLY a JSON object" in text and "submit_verdict" not in text
    assert "submit_verdict" in tool and "Respond with ONLY" not in tool
    # Same grading rules either way: only the reply instructions differ.
    assert text.startswith(judge.JUDGE_RULES["judge-v2"])
    assert tool.startswith(judge.JUDGE_RULES["judge-v2"])


@pytest.mark.parametrize("version", ["judge-v1", "judge-v2"])
def test_judge_answer_sends_the_requested_prompt(monkeypatch, version):
    captured = {}

    def fake_generate(system, user, model=None, **kwargs):
        captured["system"] = system
        return LLMResponse(text=GOOD_REPLY, model=model, input_tokens=1,
                           output_tokens=1, latency_s=0.0)

    monkeypatch.setattr(judge, "generate", fake_generate)
    judge.judge_answer("Q?", ["ctx"], "REF", "ANS", prompt_version=version, output_mode="text")
    assert captured["system"] == judge.judge_system_prompt(version, "text")


@pytest.mark.parametrize("kwargs, message", [
    ({"prompt_version": "judge-v9"}, "judge-v9"),
    ({"output_mode": "yaml"}, "yaml"),
])
def test_unknown_settings_are_rejected_before_any_call(monkeypatch, kwargs, message):
    def exploding(*args, **kw):
        raise AssertionError("no API call should be made")

    monkeypatch.setattr(judge, "generate", exploding)
    monkeypatch.setattr(judge, "generate_structured", exploding)
    with pytest.raises(ValueError, match=message):
        judge.judge_answer("Q?", ["ctx"], "REF", "ANS", **kwargs)


def test_text_mode_error_keeps_the_whole_reply():
    # Run 4 kept only the last 120 characters, too little to diagnose g023/g024.
    reply = '{"faithful": true, "correct": true} ' + "x" * 300 + ' {"second": "object"}'
    with pytest.raises(JudgeError) as info:
        parse_judge_output(reply)
    assert reply in str(info.value)


# ---- Tool output mode ---------------------------------------------------

def structured(data):
    return StructuredResponse(data=data, model="judge-x", input_tokens=1, output_tokens=1,
                              latency_s=0.0, stop_reason="tool_use")


GOOD_DATA = {"claims": [{"claim": "Revenue was $108.3M", "supported": True},
                        {"claim": "Kestrel split its stock", "supported": False}],
             "faithful": False, "correct": True, "reasoning": "One claim is unsupported."}


def test_verdict_schema_is_valid_for_strict_mode():
    """Strict mode needs additionalProperties: false and every field required,
    on every object, or the API rejects the request."""
    def check(node):
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node["properties"])
            for child in node["properties"].values():
                check(child)
        if node.get("type") == "array":
            check(node["items"])
    check(judge.VERDICT_SCHEMA)


def test_tool_mode_sends_schema_and_reads_the_tool_input(monkeypatch):
    captured = {}

    def fake_structured(system, user, schema, tool_name, tool_description, model=None, **kwargs):
        captured.update(system=system, schema=schema, tool_name=tool_name, model=model)
        return structured(GOOD_DATA)

    monkeypatch.setattr(judge, "generate_structured", fake_structured)
    verdict = judge.judge_answer("Q?", ["ctx"], "REF", "ANS", model="judge-x")   # tool is the default
    assert captured["schema"] is judge.VERDICT_SCHEMA
    assert captured["tool_name"] == "submit_verdict"
    assert captured["system"] == judge.judge_system_prompt("judge-v2", "tool")
    assert verdict.correct is True and verdict.faithful is False
    assert verdict.unsupported_claims == ["Kestrel split its stock"]


def test_tool_mode_retries_when_no_tool_call_comes_back(monkeypatch):
    replies = [StructuredOutputError("no tool call (stop_reason='max_tokens')"), structured(GOOD_DATA)]

    def fake_structured(*args, **kwargs):
        r = replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(judge, "generate_structured", fake_structured)
    assert judge.judge_answer("Q?", ["ctx"], "REF", "ANS").correct is True
    assert replies == []


def test_tool_mode_gives_up_with_a_judge_error(monkeypatch):
    def always_fails(*args, **kwargs):
        raise StructuredOutputError("no tool call (stop_reason='refusal')")

    monkeypatch.setattr(judge, "generate_structured", always_fails)
    with pytest.raises(JudgeError, match="refusal"):
        judge.judge_answer("Q?", ["ctx"], "REF", "ANS")


def test_tool_mode_still_validates_the_fields(monkeypatch):
    monkeypatch.setattr(judge, "generate_structured",
                        lambda *a, **k: structured({"claims": [], "faithful": "yes", "correct": True}))
    with pytest.raises(JudgeError, match="faithful"):
        judge.judge_answer("Q?", ["ctx"], "REF", "ANS")
