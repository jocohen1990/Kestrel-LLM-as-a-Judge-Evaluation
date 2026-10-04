"""Run the golden dataset through the RAG assistant and save a scorecard.

Usage:
    python run_evals.py                      # default model, 85% pass threshold
    python run_evals.py --threshold 0.9
    python run_evals.py --model <model-name>   # compare a different model

Tests vs. evals, a distinction worth explaining in interviews:
  * `pytest` (the tests/ folder) checks that OUR CODE works. Offline, free,
    must pass 100%.
  * `run_evals.py` measures how well THE MODEL performs. Costs API calls,
    results are a score, and we gate on a threshold rather than perfection.

Exit code 1 when the pass rate falls below the threshold. That's what lets
the CI pipeline block a pull request that makes the model worse.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

try:  # Load ANTHROPIC_API_KEY from a local .env file if one exists.
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from app.paths import GOLDEN_DIR, RESULTS_DIR
from app.rag import RagAssistant
from evals.dataset import load_golden
from evals.metrics import retrieval_hit, score_case
from evals.report import summarize


def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM evals against the golden dataset.")
    parser.add_argument("--golden", default=GOLDEN_DIR / "golden_v1.jsonl", help="Golden dataset file")
    parser.add_argument("--model", default=None, help="Model name (defaults to app.llm.DEFAULT_MODEL)")
    parser.add_argument("--k", type=int, default=4, help="Passages to retrieve per question")
    parser.add_argument("--threshold", type=float, default=0.85, help="Minimum pass rate to succeed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = load_golden(args.golden)
    assistant = RagAssistant(k=args.k, model=args.model)

    rows = []
    for case in cases:
        result = assistant.answer(case["question"])
        score = score_case(case, result.answer)
        rows.append({
            "id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "expected_answer": case["expected_answer"],
            "answer": result.answer,
            "passed": score.passed,
            "reason": score.reason,
            "retrieved_sources": result.sources,
            "retrieval_hit": retrieval_hit(case, result.sources),
            "model": result.llm.model,
            "latency_s": round(result.llm.latency_s, 3),
            "input_tokens": result.llm.input_tokens,
            "output_tokens": result.llm.output_tokens,
        })
        status = "PASS" if score.passed else "FAIL"
        print(f"[{status}] {case['id']}  {case['question']}")
        if not score.passed:
            print(f"         reason: {score.reason}")
            print(f"         answer: {result.answer}")

    summary = summarize(rows)
    run = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": rows[0]["model"] if rows else args.model,
        "golden_file": Path(args.golden).name,
        "k": args.k,
        "threshold": args.threshold,
        "summary": summary,
        "cases": rows,
    }

    # Save a timestamped copy (history) plus latest.json (for the dashboard).
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    for name in (f"run_{stamp}.json", "latest.json"):
        (RESULTS_DIR / name).write_text(json.dumps(run, indent=2), encoding="utf-8")

    print("\n=== Scorecard ===")
    print(f"Model:              {run['model']}")
    print(f"Pass rate:          {summary['passed']}/{summary['total_cases']} ({summary['pass_rate']:.0%})")
    for category, stats in summary["by_category"].items():
        print(f"  {category:<16}  {stats['passed']}/{stats['total']}")
    print(f"Retrieval hit rate: {summary['retrieval_hit_rate']:.0%}")
    print(f"Latency avg / p95:  {summary['avg_latency_s']}s / {summary['p95_latency_s']}s")
    print(f"Tokens in / out:    {summary['total_input_tokens']} / {summary['total_output_tokens']}")

    if summary["pass_rate"] < args.threshold:
        print(f"\nQUALITY GATE FAILED: {summary['pass_rate']:.0%} is below {args.threshold:.0%}")
        return 1
    print(f"\nQuality gate passed (threshold {args.threshold:.0%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
