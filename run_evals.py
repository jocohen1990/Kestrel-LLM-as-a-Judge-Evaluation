"""Run the golden dataset through the RAG assistant and save a scorecard.

Usage:
    python run_evals.py                          # golden v3, with the LLM judge
    python run_evals.py --no-judge               # rule-based graders only (cheaper)
    python run_evals.py --golden data/golden/golden_v1.jsonl
    python run_evals.py --prompt v1              # compare against the original prompt
    python run_evals.py --model <model-name>     # evaluate a different assistant model
    python run_evals.py --judge-model <name>     # use a different judge
    python run_evals.py --judge-prompt judge-v1  # grade with the original judge prompt

Tests vs. evals, a distinction worth explaining in interviews:
  * `pytest` (the tests/ folder) checks that OUR CODE works. Offline, free,
    must pass 100%.
  * `run_evals.py` measures how well THE MODEL performs. Costs API calls,
    results are a score, and we gate on thresholds rather than perfection.

Exit code 1 when a quality gate fails. That's what lets the CI pipeline
block a pull request that makes the model worse.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

try:  # Load ANTHROPIC_API_KEY from a local .env file if one exists.
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from app.llm import forces_tool_choice, warmup
from app.paths import GOLDEN_DIR, RESULTS_DIR
from app.rag import (DEFAULT_PROMPT_VERSION, DEFAULT_RETRIEVER_VERSION, PROMPTS,
                     RETRIEVER_VERSIONS, RagAssistant)
from evals.dataset import load_golden
from evals.judge import (DEFAULT_JUDGE_MODEL, DEFAULT_JUDGE_PROMPT, DEFAULT_OUTPUT_MODE,
                         JUDGE_RULES, OUTPUT_MODES)
from evals.report import summarize
from evals.runner import evaluate_case


def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM evals against the golden dataset.")
    parser.add_argument("--golden", default=GOLDEN_DIR / "golden_v3.jsonl", help="Golden dataset file")
    parser.add_argument("--model", default=None, help="Assistant model (defaults to app.llm.DEFAULT_MODEL)")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT_VERSION, choices=sorted(PROMPTS),
                        help="Assistant prompt version (see app/rag.py)")
    parser.add_argument("--retriever", default=DEFAULT_RETRIEVER_VERSION,
                        choices=sorted(RETRIEVER_VERSIONS),
                        help="Retriever version (see app/rag.py): r1 exact words, r2 + typo "
                             "tolerance, r3 + query rewrite (one extra call per question)")
    parser.add_argument("--k", type=int, default=4, help="Passages to retrieve per question")
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="Minimum rule-based pass rate")
    parser.add_argument("--min-faithfulness", type=float, default=0.9,
                        help="Minimum judge faithfulness rate (ignored with --no-judge)")
    parser.add_argument("--no-judge", action="store_true", help="Skip the LLM judge")
    parser.add_argument("--judge-model", default=None, help="Judge model name")
    parser.add_argument("--judge-prompt", default=DEFAULT_JUDGE_PROMPT, choices=sorted(JUDGE_RULES),
                        help="Judge prompt version (see evals/judge.py)")
    parser.add_argument("--judge-output", default=DEFAULT_OUTPUT_MODE, choices=OUTPUT_MODES,
                        help="How the judge returns its verdict: a strict tool call or free text")
    return parser.parse_args()


def fmt(value) -> str:
    """Show a rate as a percentage, or 'n/a' when it wasn't measured."""
    return "n/a" if value is None else f"{value:.0%}"


def print_scorecard(run: dict) -> None:
    s = run["summary"]
    print("\n=== Scorecard ===")
    print(f"Assistant model:      {run['model']}  (prompt {run['prompt_version']}, retriever {run['retriever_version']})")
    judge = (f"{run['judge_model']}  ({run['judge_prompt_version']}, {run['judge_output_mode']} output)"
             if run["judge_model"] else "off")
    print(f"Judge model:          {judge}")
    print(f"Dataset:              {run['golden_file']}")
    print(f"Rule pass rate:       {s['passed']}/{s['total_cases']} ({fmt(s['pass_rate'])})")
    for category, stats in s["by_category"].items():
        print(f"  {category:<18}  {stats['passed']}/{stats['total']}")
    print(f"Judge correct:        {fmt(s['judge_correct_rate'])}")
    print(f"Judge faithfulness:   {fmt(s['judge_faithfulness_rate'])}")
    print(f"Rule/judge agreement: {fmt(s['rule_judge_agreement'])}")
    print(f"Exact refusals:       {fmt(s['refusal_exact_rate'])}")
    print(f"Mentions internals:   {fmt(s['internals_mention_rate'])}  (lower is better)")
    print(f"Retrieval (document): {fmt(s['retrieval_hit_rate'])}")
    print(f"Retrieval (passage):  {fmt(s['passage_hit_rate'])}  (the stricter check)")
    print(f"Latency avg / p95:    {s['avg_latency_s']}s / {s['p95_latency_s']}s")
    print(f"Tokens in / out:      {s['total_input_tokens']} / {s['total_output_tokens']}")
    if s.get("judge_input_tokens"):
        cached = s["judge_cached_input_tokens"]
        print(f"Judge tokens in / out: {s['judge_input_tokens']} / {s['judge_output_tokens']}"
              f"  ({cached} in = {cached / s['judge_input_tokens']:.0%} read from cache)")
    if s["judge_errors"]:
        print(f"Judge errors:         {s['judge_errors']} (see judge_error in the results file)")


def check_gates(summary: dict, args) -> list[str]:
    """Return a list of failed gates (empty list = everything passed)."""
    failures = []
    if summary["pass_rate"] < args.threshold:
        failures.append(f"rule pass rate {fmt(summary['pass_rate'])} < {args.threshold:.0%}")
    faithfulness = summary["judge_faithfulness_rate"]
    if not args.no_judge and faithfulness is not None and faithfulness < args.min_faithfulness:
        failures.append(f"faithfulness {fmt(faithfulness)} < {args.min_faithfulness:.0%}")
    return failures


def main() -> int:
    args = parse_args()
    cases = load_golden(args.golden)
    assistant = RagAssistant(k=args.k, model=args.model, prompt_version=args.prompt,
                            retriever_version=args.retriever)
    judge_model = None if args.no_judge else (
        args.judge_model or os.getenv("JUDGE_MODEL", DEFAULT_JUDGE_MODEL))

    warmup(args.model)

    rows = []
    for case in cases:
        row = evaluate_case(case, assistant, use_judge=not args.no_judge, judge_model=judge_model,
                            judge_prompt=args.judge_prompt, judge_output=args.judge_output)
        rows.append(row)

        status = "PASS" if row["passed"] else "FAIL"
        judge = ""
        if row["judge_correct"] is not None:
            judge = (f"  judge: {'correct' if row['judge_correct'] else 'WRONG'}, "
                     f"{'faithful' if row['judge_faithful'] else 'UNFAITHFUL'}")
        print(f"[{status}] {case['id']}  {case['question']}{judge}")
        if not row["passed"] or row["judge_correct"] is False or row["judge_faithful"] is False:
            print(f"         answer: {row['answer']}")
            if not row["passed"]:
                print(f"         rule:   {row['reason']}")
            if row["judge_reasoning"]:
                print(f"         judge:  {row['judge_reasoning']}")

    run = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": rows[0]["model"] if rows else args.model,
        "prompt_version": args.prompt,
        "retriever_version": args.retriever,
        "judge_model": judge_model,
        "judge_prompt_version": None if args.no_judge else args.judge_prompt,
        "judge_output_mode": None if args.no_judge else args.judge_output,
        "judge_tool_choice_forced": (forces_tool_choice(judge_model)
                                     if judge_model and args.judge_output == "tool" else None),
        "golden_file": Path(args.golden).name,
        "k": args.k,
        "threshold": args.threshold,
        "min_faithfulness": args.min_faithfulness,
        "summary": summarize(rows),
        "cases": rows,
    }

    # Save a timestamped copy (history) plus latest.json (for the dashboard).
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    for name in (f"run_{stamp}.json", "latest.json"):
        (RESULTS_DIR / name).write_text(json.dumps(run, indent=2), encoding="utf-8")

    print_scorecard(run)

    failures = check_gates(run["summary"], args)
    if failures:
        print("\nQUALITY GATE FAILED: " + "; ".join(failures))
        return 1
    print("\nQuality gates passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
