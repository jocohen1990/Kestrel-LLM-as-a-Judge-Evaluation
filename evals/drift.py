"""Drift over time: compare two results files from different days.

Why: consistency.py asks each question 5 times back to back, and in Run 4 all
32 answers were identical every time. Yet between Run 2 and Run 3, about 4
hours apart, 3 of 32 answers changed wording. So the noise you see depends on
how far apart the runs are. This module compares the answers and verdicts in
any two results files, so the gap can be hours, days or weeks.

It reads both kinds of results file:
  * run_*.json from run_evals.py (one answer per question)
  * consistency_*.json from consistency.py (several answers per question)

A comparison only means something if nothing else changed. If the assistant
model, prompt, k or dataset differ, an answer change could be caused by that
instead of drift, so those differences are reported first. Judge verdicts are
only compared when the judge settings match AND the judged answer is
identical; otherwise a verdict change could just be a different answer.
"""
from datetime import datetime

from evals.consistency import normalize_answer

ASSISTANT_SETTINGS = ("model", "prompt_version", "retriever_version", "k", "golden_file")
JUDGE_SETTINGS = ("judge_model", "judge_prompt_version", "judge_output_mode")


def settings(run: dict) -> dict:
    s = {key: run.get(key) for key in ASSISTANT_SETTINGS + JUDGE_SETTINGS}
    # Files written before output modes existed used free-text judge replies.
    if s["judge_model"] and not s["judge_output_mode"]:
        s["judge_output_mode"] = "text"
    # Files written before retriever versions existed (Runs 1-12) all used r1.
    if not s["retriever_version"]:
        s["retriever_version"] = "r1"
    return s


def outcome(values: list) -> object:
    """Collapse repeated verdicts into one: True, False, "mixed", or None (not measured)."""
    known = {v for v in values if v is not None}
    if not known:
        return None
    return known.pop() if len(known) == 1 else "mixed"


def case_outcomes(run: dict) -> dict[str, dict]:
    """Per question: the answers given, the rule outcome, and the judge outcome
    on the answer the judge saw. Works for both results-file formats."""
    out = {}
    for c in run["cases"]:
        if "answers" in c:      # consistency file
            verdicts = [v for v in c.get("judge_verdicts", []) if v]
            out[c["id"]] = {
                "answers": c["answers"],
                "rule": outcome(c["rule_verdicts"]),
                "judged_answer": c.get("judged_answer"),
                "judge_correct": outcome([v["correct"] for v in verdicts]),
                "judge_faithful": outcome([v["faithful"] for v in verdicts]),
            }
        else:                   # run_evals file
            out[c["id"]] = {
                "answers": [c["answer"]],
                "rule": c["passed"],
                "judged_answer": c["answer"] if c.get("judge_correct") is not None else None,
                "judge_correct": c.get("judge_correct"),
                "judge_faithful": c.get("judge_faithful"),
            }
    return out


def hours_between(old: dict, new: dict) -> float | None:
    try:
        delta = datetime.fromisoformat(new["timestamp"]) - datetime.fromisoformat(old["timestamp"])
    except (KeyError, TypeError, ValueError):
        return None
    return round(delta.total_seconds() / 3600, 1)


def compare_runs(old: dict, new: dict) -> dict:
    """Compare two results files. old/new are the loaded JSON."""
    s_old, s_new = settings(old), settings(new)
    differs = {k: (s_old[k], s_new[k]) for k in s_old if s_old[k] != s_new[k]}
    assistant_comparable = not any(k in differs for k in ASSISTANT_SETTINGS)
    judge_comparable = bool(s_old["judge_model"] and s_new["judge_model"]) and \
        not any(k in differs for k in JUDGE_SETTINGS)

    a, b = case_outcomes(old), case_outcomes(new)
    rows = []
    for case_id in sorted(set(a) & set(b)):
        o, n = a[case_id], b[case_id]
        old_set = {normalize_answer(x) for x in o["answers"]}
        new_set = {normalize_answer(x) for x in n["answers"]}
        same_judged = (o["judged_answer"] is not None and n["judged_answer"] is not None and
                       normalize_answer(o["judged_answer"]) == normalize_answer(n["judged_answer"]))
        row = {
            "id": case_id,
            # Changed = no answer in common. A question that was already
            # unstable within one file isn't counted as drift if it overlaps.
            "answer_changed": not (old_set & new_set),
            "old_answer": o["answers"][0],
            "new_answer": n["answers"][0],
            "rule_old": o["rule"], "rule_new": n["rule"],
            "rule_changed": o["rule"] != n["rule"],
            "judge_compared": judge_comparable and same_judged,
            "judge_changed": [],
        }
        if row["judge_compared"]:
            for crit in ("correct", "faithful"):
                if o[f"judge_{crit}"] != n[f"judge_{crit}"]:
                    row["judge_changed"].append(
                        {"criterion": crit, "old": o[f"judge_{crit}"], "new": n[f"judge_{crit}"]})
        rows.append(row)

    return {
        "hours_apart": hours_between(old, new),
        "settings_differ": differs,
        "assistant_comparable": assistant_comparable,
        "judge_comparable": judge_comparable,
        "only_in_one_file": sorted(set(a) ^ set(b)),
        "summary": {
            "questions": len(rows),
            "answers_changed": [r["id"] for r in rows if r["answer_changed"]],
            "rule_changed": [r["id"] for r in rows if r["rule_changed"]],
            "judge_questions_compared": sum(r["judge_compared"] for r in rows),
            "judge_changed_on_same_answer": [r["id"] for r in rows if r["judge_changed"]],
        },
        "cases": rows,
    }
