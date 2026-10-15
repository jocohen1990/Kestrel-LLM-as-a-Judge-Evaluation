"""Repeat-run consistency check: how much do results change on their own?

Usage:
    python consistency.py                              # 5 runs per question, judge-v2
    python consistency.py --runs 3                     # cheaper
    python consistency.py --judge-prompt judge-v1      # compare judge versions
    python consistency.py --ids g022 g032              # only some questions
    python consistency.py --no-judge                   # assistant only

Cost: about runs x questions assistant calls, plus the same number of judge
calls (5 x 32 = 160 of each by default). It isn't a CI gate; run it when you
need to know the noise floor, e.g. before trusting a 1-question score change.

See evals/consistency.py for what's measured and why.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from app.llm import DEFAULT_MODEL, sends_temperature, warmup
from app.paths import GOLDEN_DIR, RESULTS_DIR
from app.rag import DEFAULT_PROMPT_VERSION, PROMPTS, RagAssistant
from evals.consistency import check_case, summarize_consistency
from evals.dataset import load_golden
from evals.judge import (DEFAULT_JUDGE_MODEL, DEFAULT_JUDGE_PROMPT, DEFAULT_OUTPUT_MODE,
                         JUDGE_RULES, OUTPUT_MODES)


def parse_args():
    parser = argparse.ArgumentParser(description="Measure run-to-run consistency.")
    parser.add_argument("--runs", type=int, default=5, help="Times to repeat each question")
    parser.add_argument("--golden", default=GOLDEN_DIR / "golden_v3.jsonl")
    parser.add_argument("--ids", nargs="*", default=None, help="Only these case ids")
    parser.add_argument("--model", default=None, help="Assistant model")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT_VERSION, choices=sorted(PROMPTS))
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-prompt", default=DEFAULT_JUDGE_PROMPT, choices=sorted(JUDGE_RULES))
    parser.add_argument("--judge-output", default=DEFAULT_OUTPUT_MODE, choices=OUTPUT_MODES,
                        help="How the judge returns its verdict: a strict tool call or free text")
    args = parser.parse_args()
    if args.runs < 2:
        parser.error("--runs must be at least 2; one run can't be compared with anything")
    return args


def fmt(value) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def print_report(run: dict) -> None:
    s = run["summary"]
    print(f"\n=== Consistency: {s['questions']} questions x {s['runs_per_question']} runs ===")
    print(f"Assistant: {run['model']} (prompt {run['prompt_version']}, "
          f"temperature {'0' if run['assistant_temperature_sent'] else 'NOT SENT'})")
    print(f"  Identical answers every run:  {fmt(s['identical_answer_rate'])}")
    print(f"  Avg distinct answers:         {s['avg_distinct_answers']} (1.0 = perfectly repeatable)")
    print(f"  Rule verdict stable:          {fmt(s['rule_stable_rate'])}")
    if s["changed_wording"]:
        print(f"  Wording changed:              {', '.join(s['changed_wording'])}")
    if s["rule_flips"]:
        print(f"  RULE VERDICT FLIPPED:         {', '.join(s['rule_flips'])}")
    if run["judge_model"]:
        print(f"Judge: {run['judge_model']} ({run['judge_prompt_version']}, "
              f"{run['judge_output_mode']} output, "
              f"temperature {'0' if run['judge_temperature_sent'] else 'NOT SENT'}), "
              f"same answer graded {s['runs_per_question']} times")
        print(f"  'correct' verdict stable:     {fmt(s['judge_correct_stable_rate'])}")
        print(f"  'faithful' verdict stable:    {fmt(s['judge_faithful_stable_rate'])}")
        if s["judge_correct_flips"]:
            print(f"  CORRECT FLIPPED:              {', '.join(s['judge_correct_flips'])}")
        if s["judge_faithful_flips"]:
            print(f"  FAITHFUL FLIPPED:             {', '.join(s['judge_faithful_flips'])}")
        if s["judge_errors"]:
            print(f"  Judge errors:                 {s['judge_errors']}")


def main() -> int:
    args = parse_args()
    cases = load_golden(args.golden)
    if args.ids:
        cases = [c for c in cases if c["id"] in set(args.ids)]
        if not cases:
            print(f"No cases match {args.ids}")
            return 1
    assistant = RagAssistant(k=args.k, model=args.model, prompt_version=args.prompt)
    judge_model = None if args.no_judge else (
        args.judge_model or os.getenv("JUDGE_MODEL", DEFAULT_JUDGE_MODEL))

    calls = len(cases) * args.runs
    print(f"{len(cases)} questions x {args.runs} runs = {calls} assistant calls"
          + ("" if args.no_judge else f" + {calls} judge calls"))
    warmup(args.model)

    rows = []
    for case in cases:
        row = check_case(case, assistant, args.runs, use_judge=not args.no_judge,
                         judge_model=judge_model, judge_prompt=args.judge_prompt,
                         judge_output=args.judge_output)
        rows.append(row)
        notes = []
        if not row["assistant"]["identical"]:
            notes.append(f"{row['assistant']['distinct_answers']} different answers")
        for key, label in (("rule", "rule"), ("judge_correct", "judge correct"),
                           ("judge_faithful", "judge faithful")):
            st = row[key]
            if st and st["stable"] is False:
                notes.append(f"{label} flipped ({st['passes']} pass / {st['fails']} fail)")
        print(f"[{'UNSTABLE' if notes else 'stable  '}] {case['id']}  {case['question']}"
              + (f"\n           {'; '.join(notes)}" if notes else ""))

    assistant_model = args.model or os.getenv("LLM_MODEL", DEFAULT_MODEL)
    run = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": assistant_model,
        "prompt_version": args.prompt,
        "assistant_temperature_sent": sends_temperature(assistant_model),
        "judge_model": judge_model,
        "judge_prompt_version": None if args.no_judge else args.judge_prompt,
        "judge_output_mode": None if args.no_judge else args.judge_output,
        "judge_temperature_sent": sends_temperature(judge_model) if judge_model else None,
        "golden_file": Path(args.golden).name,
        "k": args.k,
        "summary": summarize_consistency(rows),
        "cases": rows,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    for name in (f"consistency_{stamp}.json", "consistency_latest.json"):
        (RESULTS_DIR / name).write_text(json.dumps(run, indent=2), encoding="utf-8")

    print_report(run)
    print("\nSaved results/consistency_latest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
