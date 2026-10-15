"""Compare two results files from different times: has anything drifted?

Usage:
    python compare_runs.py results/consistency_<old>.json results/consistency_latest.json
    python compare_runs.py results/run_<old>.json results/latest.json

No API calls: it only reads the two files. Works with run_*.json (from
run_evals.py) and consistency_*.json (from consistency.py), in any mix.

Typical use: run consistency.py today, again tomorrow (or next week), then
compare the two files. See evals/drift.py for what counts as a change.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from app.paths import RESULTS_DIR
from evals.drift import compare_runs


def short(text: str, width: int = 110) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 3] + "..."


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two results files for drift.")
    parser.add_argument("old", help="The earlier results file")
    parser.add_argument("new", help="The later results file")
    args = parser.parse_args()

    old = json.loads(Path(args.old).read_text(encoding="utf-8"))
    new = json.loads(Path(args.new).read_text(encoding="utf-8"))
    result = compare_runs(old, new)
    s = result["summary"]

    gap = result["hours_apart"]
    print(f"=== Drift: {Path(args.old).name} -> {Path(args.new).name}"
          + (f" ({gap} hours apart)" if gap is not None else "") + " ===")

    if result["settings_differ"]:
        print("\nSettings that differ (changes may come from these, not drift):")
        for key, (a, b) in result["settings_differ"].items():
            print(f"  {key}: {a} -> {b}")
    if not result["assistant_comparable"]:
        print("\nWARNING: assistant settings differ, so answer changes are NOT pure drift.")
    if not result["judge_comparable"]:
        print("Judge verdicts not compared (judge off or judge settings differ).")
    if result["only_in_one_file"]:
        print(f"In only one file (skipped): {', '.join(result['only_in_one_file'])}")

    q = s["questions"]
    print(f"\nQuestions compared:        {q}")
    print(f"Answer changed:            {len(s['answers_changed'])}/{q}"
          + (f"  ({', '.join(s['answers_changed'])})" if s["answers_changed"] else ""))
    print(f"Rule verdict changed:      {len(s['rule_changed'])}/{q}"
          + (f"  ({', '.join(s['rule_changed'])})" if s["rule_changed"] else ""))
    if result["judge_comparable"]:
        print(f"Judge changed, same answer: {len(s['judge_changed_on_same_answer'])}/"
              f"{s['judge_questions_compared']}"
              + (f"  ({', '.join(s['judge_changed_on_same_answer'])})"
                 if s["judge_changed_on_same_answer"] else ""))

    for row in result["cases"]:
        if row["answer_changed"] or row["rule_changed"] or row["judge_changed"]:
            print(f"\n[{row['id']}]")
            if row["answer_changed"]:
                print(f"  old: {short(row['old_answer'])}")
                print(f"  new: {short(row['new_answer'])}")
            if row["rule_changed"]:
                print(f"  rule: {row['rule_old']} -> {row['rule_new']}")
            for c in row["judge_changed"]:
                print(f"  judge {c['criterion']}: {c['old']} -> {c['new']} (same answer)")

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    result["files"] = [Path(args.old).name, Path(args.new).name]
    (RESULTS_DIR / f"drift_{stamp}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved results/drift_{stamp}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
