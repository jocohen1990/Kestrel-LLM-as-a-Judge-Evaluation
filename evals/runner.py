"""Evaluate one golden case: ask the assistant, then apply every grader.

This lives apart from run_evals.py (the command-line part) so tests can call
it with a fake assistant and a fake judge, with no API calls at all.
"""
from app.rag import REFUSAL
from evals.judge import DEFAULT_JUDGE_PROMPT, DEFAULT_OUTPUT_MODE, JudgeError, judge_answer
from evals.metrics import (
    mentions_internals,
    passage_hit,
    refusal_is_exact,
    retrieval_hit,
    score_case,
)


def evaluate_case(case: dict, assistant, use_judge: bool = True,
                  judge_model: str | None = None, judge_fn=judge_answer,
                  judge_prompt: str = DEFAULT_JUDGE_PROMPT,
                  judge_output: str = DEFAULT_OUTPUT_MODE) -> dict:
    """Run one case end to end and return a flat result row (easy to save as JSON)."""
    result = assistant.answer(case["question"])
    rule = score_case(case, result.answer)

    row = {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "expected_answer": case["expected_answer"],
        "expected_numbers": case["expected_numbers"],
        "answer": result.answer,
        # Rule-based grader (Phase 1)
        "passed": rule.passed,
        "reason": rule.reason,
        # Extra rule-based checks (Phase 2)
        "refusal_exact": (refusal_is_exact(result.answer, REFUSAL)
                          if case["category"] == "out_of_scope" else None),
        "mentions_internals": mentions_internals(result.answer),
        # Retrieval
        "retrieved_sources": result.sources,
        "retrieval_hit": retrieval_hit(case, result.sources),
        "passage_hit": passage_hit(case, result.contexts),
        "contexts": result.contexts,   # saved so humans can check faithfulness later
        # Performance
        "model": result.llm.model,
        "latency_s": round(result.llm.latency_s, 3),
        "input_tokens": result.llm.input_tokens,
        "output_tokens": result.llm.output_tokens,
        # LLM judge (filled in below)
        "judge_faithful": None,
        "judge_correct": None,
        "judge_unsupported_claims": [],
        "judge_reasoning": None,
        "judge_error": None,
        "rule_judge_agree": None,
        "judge_input_tokens": None,
        "judge_cached_input_tokens": None,
        "judge_output_tokens": None,
    }

    if use_judge:
        try:
            verdict = judge_fn(case["question"], result.contexts,
                               case["expected_answer"], result.answer, model=judge_model,
                               prompt_version=judge_prompt, output_mode=judge_output)
            row.update({
                "judge_faithful": verdict.faithful,
                "judge_correct": verdict.correct,
                "judge_unsupported_claims": verdict.unsupported_claims,
                "judge_reasoning": verdict.reasoning,
                # Where the two graders disagree is where the interesting
                # findings are: either the rules or the judge got it wrong.
                "rule_judge_agree": rule.passed == verdict.correct,
                "judge_input_tokens": getattr(verdict, "input_tokens", None),
                "judge_cached_input_tokens": getattr(verdict, "cached_input_tokens", None),
                "judge_output_tokens": getattr(verdict, "output_tokens", None),
            })
        except JudgeError as exc:
            # Recorded, never counted as pass or fail.
            row["judge_error"] = str(exc)
    return row
