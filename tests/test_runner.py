"""Test evaluate_case with a fake assistant and fake judges: no API calls."""
from types import SimpleNamespace

from app.llm import LLMResponse
from app.rag import REFUSAL
from evals.judge import JudgeError, Verdict
from evals.runner import evaluate_case


class FakeAssistant:
    def __init__(self, answer):
        self._answer = answer

    def answer(self, question):
        llm = LLMResponse(text=self._answer, model="fake", input_tokens=10,
                          output_tokens=5, latency_s=0.1)
        return SimpleNamespace(answer=self._answer, sources=["q1_2026_earnings.md"],
                               contexts=["ctx"], llm=llm)


CASE = {"id": "g006", "category": "numeric", "question": "Q1 revenue?",
        "expected_answer": "$108.3 million", "expected_numbers": [108.3],
        "must_include": [], "must_include_any": [], "source_docs": ["q1_2026_earnings.md"],
        "evidence": ["108.3"]}


def judge_says(correct, faithful):
    def _judge(*args, **kwargs):
        return Verdict(faithful=faithful, correct=correct, reasoning="r",
                       claims=[{"claim": "made up", "supported": faithful}])
    return _judge


def test_rule_and_judge_both_recorded():
    row = evaluate_case(CASE, FakeAssistant("Revenue was $108.3 million."),
                        judge_fn=judge_says(correct=True, faithful=True))
    assert row["passed"] and row["judge_correct"] and row["judge_faithful"]
    assert row["rule_judge_agree"] is True
    assert row["retrieval_hit"] is True
    assert row["passage_hit"] is False   # the fake assistant's only passage is "ctx"


def test_judge_catches_what_the_rules_miss():
    # Right number (rules pass) plus an invented claim (judge flags it).
    row = evaluate_case(CASE, FakeAssistant("Revenue was $108.3 million, and it split its stock."),
                        judge_fn=judge_says(correct=True, faithful=False))
    assert row["passed"] is True
    assert row["judge_faithful"] is False
    assert row["judge_unsupported_claims"] == ["made up"]


def test_judge_error_is_recorded_not_scored():
    def broken_judge(*args, **kwargs):
        raise JudgeError("garbled reply")

    row = evaluate_case(CASE, FakeAssistant("Revenue was $108.3 million."), judge_fn=broken_judge)
    assert row["judge_error"] == "garbled reply"
    assert row["judge_correct"] is None and row["rule_judge_agree"] is None


def test_no_judge_mode_never_calls_the_judge():
    def exploding_judge(*args, **kwargs):
        raise AssertionError("judge should not be called")

    row = evaluate_case(CASE, FakeAssistant("x 108.3"), use_judge=False, judge_fn=exploding_judge)
    assert row["judge_correct"] is None


def test_strict_refusal_check_only_for_out_of_scope():
    oos = {**CASE, "category": "out_of_scope", "expected_numbers": [], "source_docs": []}
    exact = evaluate_case(oos, FakeAssistant(REFUSAL), use_judge=False)
    chatty = evaluate_case(oos, FakeAssistant(REFUSAL + "\n\nThe passages don't say."), use_judge=False)
    assert exact["passed"] and exact["refusal_exact"] is True
    assert chatty["passed"] and chatty["refusal_exact"] is False   # lenient pass, strict fail
    assert evaluate_case(CASE, FakeAssistant("108.3"), use_judge=False)["refusal_exact"] is None


def test_judge_prompt_version_is_passed_to_the_judge():
    seen = {}

    def recording_judge(*args, **kwargs):
        seen.update(kwargs)
        return Verdict(faithful=True, correct=True, reasoning="r")

    evaluate_case(CASE, FakeAssistant("108.3"), judge_fn=recording_judge, judge_prompt="judge-v1")
    assert seen["prompt_version"] == "judge-v1"


def test_judge_token_usage_is_recorded_and_summed():
    from evals.report import summarize

    def judge_with_usage(*args, **kwargs):
        return Verdict(faithful=True, correct=True, reasoning="r", input_tokens=900,
                       cached_input_tokens=600, output_tokens=50)

    rows = [evaluate_case(CASE, FakeAssistant("Revenue was $108.3 million."),
                          judge_fn=judge_with_usage) for _ in range(2)]
    assert rows[0]["judge_input_tokens"] == 900 and rows[0]["judge_cached_input_tokens"] == 600
    s = summarize(rows)
    assert (s["judge_input_tokens"], s["judge_cached_input_tokens"], s["judge_output_tokens"]) \
        == (1800, 1200, 100)


def test_summary_works_on_rows_saved_before_judge_usage_existed():
    from evals.report import summarize
    row = evaluate_case(CASE, FakeAssistant("Revenue was $108.3 million."), use_judge=False)
    for key in ("judge_input_tokens", "judge_cached_input_tokens", "judge_output_tokens"):
        row.pop(key)
    assert summarize([row])["judge_input_tokens"] == 0
