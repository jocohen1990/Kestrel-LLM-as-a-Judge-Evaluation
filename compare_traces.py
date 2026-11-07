"""Compare two trace files of the same traffic (Phase 6 step 3). Free: no API calls
unless --judge-sample is set.

Usage:
    python compare_traces.py A.jsonl B.jsonl
        # the RAG assistant vs the agent, or one app before (A) and after (B) a change
    python compare_traces.py A.jsonl B.jsonl --reviews data/traffic/reviews_v2.jsonl
        # score the comparison against a person's review (across apps: one file
        # covers both; across releases it's taken as A's review)
    python compare_traces.py A.jsonl B.jsonl --reviews REV_A --reviews-b REV_B
        # across releases, with a review of each
    python compare_traces.py A.jsonl B.jsonl --all     # also list answers that were only reworded
    python compare_traces.py A.jsonl B.jsonl --strict  # exit 1 if B has an alert A didn't

Each file is monitored first with the same checks (evals/monitors.py), then
the two traces of each request are put side by side. The comparison queue
holds every request where:

    one side answered and the other declined   (at most one of them is right)
    the outcome changed any other way          (asked back, an error)
    a review or alert check fires on one side only
    the request has a trace on one side only

Writes results/comparison_latest.json (and a timestamped copy), which the
dashboard shows. See evals/compare.py.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from app.paths import RESULTS_DIR
from evals.compare import compare, format_comparison, score_against_reviews
from evals.monitors import load_reviews
from evals.traces import load_traces


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Compare two trace files of the same traffic.")
    p.add_argument("a", help="Trace file A (the baseline, across releases)")
    p.add_argument("b", help="Trace file B (the candidate, across releases)")
    p.add_argument("--reviews", metavar="FILE",
                   help="Review of the replies (across apps: both sides; across releases: A)")
    p.add_argument("--reviews-b", metavar="FILE", help="Review of B's replies (across releases)")
    p.add_argument("--judge-sample", type=float, default=0.0, metavar="RATE",
                   help="Run the monitors' sampled judge on each side too (costs a call per answer)")
    p.add_argument("--all", action="store_true", help="Also list answers that were only reworded")
    p.add_argument("--strict", action="store_true", help="Exit 1 if B raises an alert A didn't")
    p.add_argument("--no-save", action="store_true", help="Don't write results/comparison_*.json")
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not 0 <= args.judge_sample <= 1:
        print("--judge-sample must be between 0 and 1")
        return 2
    traces = {}
    for name in ("a", "b"):
        path = Path(getattr(args, name))
        if not path.exists():
            print(f"No such file: {path}")
            return 2
        traces[name] = load_traces(path)
        if not traces[name]:
            print(f"No traces in {path}.")
            return 2
    print(f"A: {len(traces['a'])} trace(s) from {args.a}\nB: {len(traces['b'])} trace(s) from {args.b}\n")

    result = compare(traces["a"], traces["b"], judge_sample=args.judge_sample)
    result["files"] = {"a": Path(args.a).name, "b": Path(args.b).name}
    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    if result["paired"] == 0:
        print("The two files share no traffic case ids: nothing to compare.")
        return 2

    scores = None
    if args.reviews or args.reviews_b:
        rev_a = load_reviews(args.reviews) if args.reviews else None
        if result["mode"] == "apps":
            rev_b = load_reviews(args.reviews_b) if args.reviews_b else rev_a
        else:
            rev_b = load_reviews(args.reviews_b) if args.reviews_b else None
        scores = score_against_reviews(result, rev_a, rev_b)
        result["review_scores"] = scores
        result["reviews_files"] = {"a": args.reviews and Path(args.reviews).name,
                                   "b": args.reviews_b and Path(args.reviews_b).name}

    print(format_comparison(result, scores, show_all=args.all))

    if not args.no_save:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        args.results_dir.mkdir(parents=True, exist_ok=True)
        for name in (f"comparison_{stamp}.json", "comparison_latest.json"):
            (args.results_dir / name).write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                                 encoding="utf-8")
        print("\nSaved results/comparison_latest.json")

    new_alerts = result["b"]["summary"]["alerts"] > result["a"]["summary"]["alerts"]
    return 1 if args.strict and new_alerts else 0


if __name__ == "__main__":
    sys.exit(main())
