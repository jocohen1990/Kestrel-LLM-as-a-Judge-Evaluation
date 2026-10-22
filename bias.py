"""Bias probes: the same question asked by different people, in different tones,
and with different opinions attached. The answers should not change.

Usage:
    python bias.py                       # 8 groups, 28 questions, default prompt
    python bias.py --groups bias04 bias06
    python bias.py --min-consistency 1.0 # exit 1 if any group is inconsistent
    python bias.py --max-verdict-differences 0
                                         # exit 1 if any variant's verdict differs from its group

Cost: one assistant call per row (28 by default). No judge calls.
See evals/bias.py for what's measured and data/bias/README.md for the dataset.
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
from app.paths import BIAS_DIR, RESULTS_DIR
from app.rag import (DEFAULT_PROMPT_VERSION, DEFAULT_RETRIEVER_VERSION, PROMPTS,
                     RETRIEVER_VERSIONS, RagAssistant)
from evals.bias import load_bias, run_row, summarize_bias


def parse_args():
    parser = argparse.ArgumentParser(description="Run the bias probes.")
    parser.add_argument("--cases", default=BIAS_DIR / "bias_v1.jsonl")
    parser.add_argument("--groups", nargs="*", default=None, help="Only these groups")
    parser.add_argument("--model", default=None, help="Assistant model")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT_VERSION, choices=sorted(PROMPTS))
    parser.add_argument("--retriever", default=DEFAULT_RETRIEVER_VERSION,
                        choices=sorted(RETRIEVER_VERSIONS),
                        help="Retriever version (see app/rag.py): r1 exact words, r2 + typo "
                             "tolerance, r3 + query rewrite (one extra call per question)")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--min-consistency", type=float, default=None,
                        help="Fail (exit 1) if the share of consistent groups is below this")
    # The CI gate uses this one. Verdicts are reliable; word flags aren't (a model
    # that pushes back with "it wasn't disappointing" gets flagged for the word),
    # so flags are reported for review but don't fail the build.
    parser.add_argument("--max-verdict-differences", type=int, default=None,
                        help="Fail (exit 1) if more variants than this get a different verdict")
    return parser.parse_args()


def fmt(value) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def print_report(run: dict) -> None:
    s = run["summary"]
    print(f"\n=== Bias probes: {s['groups']} groups, {s['rows']} questions ===")
    print(f"Assistant: {run['model']} (prompt {run['prompt_version']}, retriever {run['retriever_version']}, k={run['k']})")
    print(f"  Consistent groups:  {s['consistent_groups']}/{s['groups']} ({fmt(s['consistency_rate'])})")
    print(f"  Rows passed:        {fmt(s['pass_rate'])}")
    for key, label in (("verdict_differs_ids", "VERDICT DIFFERS FROM THE PLAIN VERSION"),
                       ("flagged_ids", "Echoed the asker's framing or identity"),
                       ("failed_ids", "Failed")):
        if s[key]:
            print(f"  {label}: {', '.join(s[key])}")
    print("  By group:")
    for name, g in s["by_group"].items():
        status = "consistent  " if g["consistent"] else "INCONSISTENT"
        print(f"    {name} {g['dimension']:<15} [{status}] "
              f"{'all passed' if g['all_passed'] else 'some failed':<11} "
              f"length max/min {g['length_ratio']}")


def main() -> int:
    args = parse_args()
    rows = load_bias(args.cases)
    if args.groups:
        rows = [r for r in rows if r["group"] in set(args.groups)]
    if not rows:
        print("No rows match those filters.")
        return 1

    assistant = RagAssistant(k=args.k, model=args.model, prompt_version=args.prompt,
                            retriever_version=args.retriever)
    print(f"{len(rows)} assistant calls")
    warmup(args.model)

    results = []
    for row in rows:
        r = run_row(row, assistant)
        results.append(r)
        flag = f"  flagged: {', '.join(r['flag_hits'])}" if r["flagged"] else ""
        print(f"[{'PASS' if r['passed'] else 'FAIL'}] {r['id']:<22} {r['question'][:70]}{flag}")

    model = args.model or os.getenv("LLM_MODEL", DEFAULT_MODEL)
    run = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "prompt_version": args.prompt,
        "retriever_version": args.retriever,
        "assistant_temperature_sent": sends_temperature(model),
        "cases_file": Path(args.cases).name,
        "k": args.k,
        "summary": summarize_bias(results),
        "cases": results,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    for name in (f"bias_{stamp}.json", "bias_latest.json"):
        (RESULTS_DIR / name).write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")

    print_report(run)
    print("\nSaved results/bias_latest.json")

    s = run["summary"]
    failed = False
    rate = s["consistency_rate"]
    if args.min_consistency is not None and rate is not None and rate < args.min_consistency:
        print(f"\nFAIL: {rate:.0%} of groups consistent, below the {args.min_consistency:.0%} minimum")
        failed = True
    differs = len(s["verdict_differs_ids"])
    if args.max_verdict_differences is not None and differs > args.max_verdict_differences:
        print(f"\nFAIL: {differs} variants got a different verdict from their group "
              f"(limit {args.max_verdict_differences}): {', '.join(s['verdict_differs_ids'])}")
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
