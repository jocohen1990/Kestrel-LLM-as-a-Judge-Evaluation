"""Repeat-run consistency: ask the same thing several times, measure what changes.

Why: Run 3 found two kinds of drift, both at temperature 0.
  * The assistant's wording changed on 3 of 32 questions between runs.
  * The judge gave OPPOSITE verdicts on an identical answer (g032).
A single eval run can't show either problem. If the score moves between two
runs, you need to know how much it moves on its own before you can blame a
prompt or model change. That "noise floor" is what this module measures.

Two separate measurements, so one kind of noise can't hide inside the other:

1. Assistant consistency: ask each question N times. Count how many different
   answers come back, and whether the rule grader's verdict stays the same.
2. Judge consistency: take ONE fixed answer per question (the first run's) and
   have the judge grade it N times. The answer never changes, so any change
   in verdict is the judge's own noise.

"Stable" means every run gave the same result. A question is reported as
unstable if even one run differs: a test that passes 4 times out of 5 is a
flaky test, and QA treats flaky tests as bugs.
"""
from collections import Counter

from evals.judge import DEFAULT_JUDGE_PROMPT, DEFAULT_OUTPUT_MODE, JudgeError, judge_answer
from evals.metrics import score_case


def normalize_answer(text: str) -> str:
    """Ignore differences in spacing and line breaks; anything else counts."""
    return " ".join((text or "").split())


def answer_stability(answers: list[str]) -> dict:
    """How many different answers came back, and how dominant the most common one is."""
    counts = Counter(normalize_answer(a) for a in answers)
    top = counts.most_common(1)[0][1] if counts else 0
    return {
        "distinct_answers": len(counts),
        "identical": len(counts) == 1,
        "most_common_share": round(top / len(answers), 3) if answers else None,
    }


def verdict_stability(verdicts: list) -> dict:
    """Summarize repeated pass/fail verdicts. None = the grader failed on that run.

    stable is None when fewer than 2 runs produced a verdict, because
    "consistent with itself" means nothing with only one data point.
    """
    known = [v for v in verdicts if v is not None]
    passes = sum(1 for v in known if v)
    return {
        "passes": passes,
        "fails": len(known) - passes,
        "errors": len(verdicts) - len(known),
        "stable": (len(set(known)) == 1) if len(known) >= 2 else None,
    }


def check_case(case: dict, assistant, runs: int, use_judge: bool = True,
               judge_fn=judge_answer, judge_model: str | None = None,
               judge_prompt: str = DEFAULT_JUDGE_PROMPT,
               judge_output: str = DEFAULT_OUTPUT_MODE) -> dict:
    """Run one golden case `runs` times through the assistant, and `runs` times
    through the judge on a fixed answer. Returns one result row."""
    results = [assistant.answer(case["question"]) for _ in range(runs)]
    answers = [r.answer for r in results]
    rule_passes = [score_case(case, a).passed for a in answers]

    row = {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "runs": runs,
        "answers": answers,
        "assistant": answer_stability(answers),
        "rule_verdicts": rule_passes,
        "rule": verdict_stability(rule_passes),
        "judged_answer": None,
        "judge_verdicts": [],
        "judge_correct": None,
        "judge_faithful": None,
        "judge_errors": [],
    }
    if not use_judge:
        return row

    fixed = results[0]   # the same answer and passages, every time
    verdicts, errors = [], []
    for _ in range(runs):
        try:
            v = judge_fn(case["question"], fixed.contexts, case["expected_answer"], fixed.answer,
                         model=judge_model, prompt_version=judge_prompt, output_mode=judge_output)
            verdicts.append({"correct": v.correct, "faithful": v.faithful, "reasoning": v.reasoning})
        except JudgeError as exc:
            verdicts.append(None)
            errors.append(str(exc))

    row.update({
        "judged_answer": fixed.answer,
        "judge_verdicts": verdicts,
        "judge_correct": verdict_stability([v["correct"] if v else None for v in verdicts]),
        "judge_faithful": verdict_stability([v["faithful"] if v else None for v in verdicts]),
        "judge_errors": errors,
    })
    return row


def _rate(flags: list) -> float | None:
    known = [f for f in flags if f is not None]
    return round(sum(known) / len(known), 3) if known else None


def summarize_consistency(rows: list[dict]) -> dict:
    """Roll per-question rows up into the headline numbers."""
    def unstable(key):
        return [r["id"] for r in rows if r[key] and r[key]["stable"] is False]

    return {
        "questions": len(rows),
        "runs_per_question": rows[0]["runs"] if rows else 0,
        # Assistant
        "identical_answer_rate": _rate([r["assistant"]["identical"] for r in rows]),
        "avg_distinct_answers": (round(sum(r["assistant"]["distinct_answers"] for r in rows)
                                       / len(rows), 2) if rows else None),
        "rule_stable_rate": _rate([r["rule"]["stable"] for r in rows]),
        "changed_wording": [r["id"] for r in rows if not r["assistant"]["identical"]],
        "rule_flips": unstable("rule"),
        # Judge (None when the judge was off)
        "judge_correct_stable_rate": _rate([r["judge_correct"]["stable"]
                                            for r in rows if r["judge_correct"]]),
        "judge_faithful_stable_rate": _rate([r["judge_faithful"]["stable"]
                                             for r in rows if r["judge_faithful"]]),
        "judge_correct_flips": unstable("judge_correct"),
        "judge_faithful_flips": unstable("judge_faithful"),
        "judge_errors": sum(len(r["judge_errors"]) for r in rows),
    }
