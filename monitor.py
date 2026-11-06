"""Monitor production traces without answer keys (Phase 6 step 2).

Usage:
    python monitor.py                          # newest file in traces/: every rule check (free)
    python monitor.py traces/traces_<time>.jsonl
    python monitor.py --queue                  # ... and print the review queue
    python monitor.py --judge-sample 0.25      # + the judge's faithfulness check on 25% of answers
    python monitor.py --judge-sample 1         # ... on every answer worth judging
    python monitor.py --reviews data/traffic/reviews_v1.jsonl   # score the checks against a person's review
    python monitor.py --strict                 # exit 1 on warnings, not just alerts

Cost: the rule checks are free. The judge (claude-sonnet-5-5) is about a cent
per judged answer; exact refusals aren't judged (no claims to check).

Writes results/monitor_<app>_latest.json (and a timestamped copy), which the
dashboard shows. Exit code 1 if any alert fired, so it can run on a schedule
and fail loudly.

A flag means "a person should look", not "wrong". See evals/monitors.py.
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

from app.paths import RESULTS_DIR, TRACES_DIR
from evals.monitors import (format_queue, format_report, load_reviews, monitor,
                            score_against_reviews)
from evals.traces import load_traces, trace_files


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run the production monitors on recorded traces.")
    p.add_argument("file", nargs="?", help="Trace file (default: newest in traces/)")
    p.add_argument("--judge-sample", type=float, default=0.0, metavar="RATE",
                   help="Share of answers the judge checks for faithfulness, 0 to 1 (default 0)")
    p.add_argument("--queue", action="store_true", help="Print the review queue")
    p.add_argument("--reviews", metavar="FILE", help="Score the checks against a human review file")
    p.add_argument("--strict", action="store_true", help="Exit 1 on warnings too")
    p.add_argument("--no-save", action="store_true", help="Don't write results/monitor_*.json")
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not 0 <= args.judge_sample <= 1:
        print("--judge-sample must be between 0 and 1")
        return 2
    source = Path(args.file) if args.file else (trace_files(TRACES_DIR) or [None])[-1]
    if source is None or not source.exists():
        print("No trace files yet. Record some with: python traffic.py --limit 5")
        return 2
    traces = load_traces(source)
    if not traces:
        print(f"No traces in {source}.")
        return 2
    print(f"Read {len(traces)} trace(s) from {source}\n")

    result = monitor(traces, judge_sample=args.judge_sample)
    result["trace_file"] = source.name
    result["timestamp"] = datetime.now(timezone.utc).isoformat()

    scores = None
    if args.reviews:
        reviews = load_reviews(args.reviews)
        ids = [t["attributes"].get("traffic.case_id") for t in traces]
        scores = score_against_reviews(result, reviews, [i for i in ids if i])
        result["review_scores"] = scores
        result["reviews_file"] = Path(args.reviews).name

    print(format_report(result, scores))
    if args.queue:
        print("\n" + format_queue(result))

    if not args.no_save:
        app = result["apps"][0] if len(result["apps"]) == 1 else "mixed"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        args.results_dir.mkdir(parents=True, exist_ok=True)
        for name in (f"monitor_{app}_{stamp}.json", f"monitor_{app}_latest.json"):
            (args.results_dir / name).write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                                 encoding="utf-8")
        print(f"\nSaved results/monitor_{app}_latest.json")

    levels = {a["level"] for a in result["alerts"]}
    return 1 if "alert" in levels or (args.strict and "warn" in levels) else 0


if __name__ == "__main__":
    sys.exit(main())
