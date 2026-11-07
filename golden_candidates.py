"""Turn flagged production requests into golden-set candidates (Phase 6 step 3). Free.

Usage:
    python golden_candidates.py                   # every trace file in traces/, both reviews
    python golden_candidates.py traces/A.jsonl traces/B.jsonl
    python golden_candidates.py --reviews data/traffic/reviews_v2.jsonl
    python golden_candidates.py --false-alarms    # also list monitor flags a review found fine

A candidate is a request that a monitor queued, that two trace files of the
same traffic disagreed on (one answered, one declined), or that a person's
review found a problem with. The monitors run on every file; the comparison
runs on every two files that replayed the same traffic file.

Candidates have no answer key. To promote one, read the documents, write the
case into a new dataset version (golden_v4.jsonl, agent_v2.jsonl) with
"origin": {"traffic_id": ...}, and run this again: it shows as promoted.

Writes results/golden_candidates_latest.json (and a timestamped copy).
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from app.paths import AGENT_DIR, GOLDEN_DIR, RESULTS_DIR, TRACES_DIR, TRAFFIC_DIR
from evals.candidates import (build_candidates, format_candidates, load_traffic_files,
                              promoted_cases)
from evals.monitors import load_reviews
from evals.traces import load_traces, trace_files

DATASETS = [GOLDEN_DIR / "golden_v4.jsonl", AGENT_DIR / "agent_v2.jsonl"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Golden-set candidates from flagged production requests.")
    p.add_argument("files", nargs="*", help="Trace files (default: every file in traces/)")
    p.add_argument("--reviews", nargs="*", metavar="FILE",
                   help="Review files (default: every data/traffic/reviews_v*.jsonl)")
    p.add_argument("--datasets", nargs="*", metavar="FILE", default=[str(d) for d in DATASETS],
                   help="Datasets whose `origin` fields mark candidates as promoted")
    p.add_argument("--false-alarms", action="store_true", help="Also list priority-4 false alarms")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    paths = [Path(f) for f in args.files] or trace_files(TRACES_DIR)
    if not paths:
        print("No trace files yet. Record some with: python traffic.py")
        return 2
    files = {}
    for path in paths:
        if not path.exists():
            print(f"No such file: {path}")
            return 2
        files[path.name] = load_traces(path)
    review_paths = ([Path(r) for r in args.reviews] if args.reviews is not None
                    else sorted(TRAFFIC_DIR.glob("reviews_v*.jsonl")))
    reviews = [row for p in review_paths for row in load_reviews(p)]

    result = build_candidates(files, reviews, load_traffic_files(TRAFFIC_DIR),
                              promoted_cases(*args.datasets))
    result["reviews_files"] = [p.name for p in review_paths]
    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    print(format_candidates(result, include_false_alarms=args.false_alarms))

    if not args.no_save:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        args.results_dir.mkdir(parents=True, exist_ok=True)
        for name in (f"golden_candidates_{stamp}.json", "golden_candidates_latest.json"):
            (args.results_dir / name).write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                                 encoding="utf-8")
        print("\nSaved results/golden_candidates_latest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
