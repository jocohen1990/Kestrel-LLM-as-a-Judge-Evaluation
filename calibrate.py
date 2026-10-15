"""Check the judge before trusting it. Three commands, used in this order:

1. Mutation check (automatic, no labeling needed):
       python calibrate.py mutate
   Takes results/latest.json, plants known errors in the answers, and reports
   how many the judge catches vs. how many the rule-based grader catches.

2. Export a blind labeling sheet:
       python calibrate.py export results/latest.json results/mutations_latest.json
   Writes data/labels/labels_<timestamp>.csv. Open it in Excel or Google
   Sheets and fill in human_correct and human_faithful with y or n, following
   data/labels/LABELING_GUIDE.md.
   Don't open the matching .key.json file until you're done; it holds the
   judge's verdicts. Aim for about 50 labeled rows.

3. Score agreement:
       python calibrate.py score data/labels/labels_<timestamp>.csv

After changing the judge prompt, re-grade the rows you already labeled. Your
labels stay the same; only the judge changes, so the comparison is fair:
       python calibrate.py rejudge data/labels/labels_<timestamp>.csv --judge-prompt judge-v2
       python calibrate.py mutate --judge-prompt judge-v2     # still catches planted errors?
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

from app.paths import GOLDEN_DIR, RESULTS_DIR, ROOT
from evals.calibration import changed_verdicts, export_for_labeling, rejudge_label_rows, score_labels
from evals.dataset import load_golden
from evals.judge import DEFAULT_JUDGE_PROMPT, JUDGE_PROMPTS, JudgeError, judge_answer
from evals.metrics import score_case
from evals.mutations import caught, make_mutations, summarize_mutations


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def save(name: str, data: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    for filename in (f"{name}_{stamp()}.json", f"{name}_latest.json"):
        (RESULTS_DIR / filename).write_text(json.dumps(data, indent=2), encoding="utf-8")


def cmd_mutate(args) -> int:
    run = json.loads(Path(args.results).read_text(encoding="utf-8"))
    cases = {c["id"]: c for c in load_golden(GOLDEN_DIR / run["golden_file"])}

    rows = []
    for original in run["cases"]:
        for m in make_mutations(original):
            # The rule grader "catches" a mutation if it fails the answer.
            m["rule_caught"] = not score_case(cases[m["id"]], m["answer"]).passed
            try:
                verdict = judge_answer(m["question"], m["contexts"], m["expected_answer"],
                                       m["answer"], model=args.judge_model,
                                       prompt_version=args.judge_prompt)
            except JudgeError as exc:
                # Saved, not skipped, so the results file shows every judge failure.
                m.update({"judge_correct": None, "judge_faithful": None, "judge_reasoning": None,
                          "judge_caught": None, "judge_error": str(exc)})
                rows.append(m)
                print(f"[ERROR ] {m['id']} {m['mutation']:<12} judge failed: {str(exc)[:120]}")
                continue
            m.update({
                "judge_correct": verdict.correct,
                "judge_faithful": verdict.faithful,
                "judge_reasoning": verdict.reasoning,
                "judge_caught": caught(m["mutation"], verdict.correct, verdict.faithful),
                "judge_error": None,
            })
            rows.append(m)
            mark = "caught" if m["judge_caught"] else "MISSED"
            print(f"[{mark}] {m['id']} {m['mutation']:<12} {m['answer'][:90]}")

    by_type = summarize_mutations(rows)
    print("\n=== Mutation check: planted errors caught ===")
    print(f"{'mutation':<14}{'judge':>12}{'rule grader':>16}{'judge errors':>15}")
    for name, s in by_type.items():
        print(f"{name:<14}{s['judge_caught']:>6}/{s['total']:<5}"
              f"{s['rule_caught']:>10}/{s['total']:<5}{s['judge_errors']:>10}")

    save("mutations", {"source_results": Path(args.results).name,
                       "judge_model": args.judge_model, "judge_prompt_version": args.judge_prompt,
                       "by_type": dict(by_type), "cases": rows})
    print("\nSaved results/mutations_latest.json")
    return 0


def cmd_export(args) -> int:
    rows = []
    for path in args.results:
        rows.extend(json.loads(Path(path).read_text(encoding="utf-8"))["cases"])
    out = Path(args.out) if args.out else ROOT / "data" / "labels" / f"labels_{stamp()}.csv"
    key_path = export_for_labeling(rows, out)
    print(f"Labeling sheet: {out}")
    print(f"Answer key:     {key_path}  (don't open until you've finished labeling)")
    print("Read data/labels/LABELING_GUIDE.md first; it has the rules and worked examples.")
    print("Fill human_correct and human_faithful with y or n, then run:")
    print(f"  python calibrate.py score {out}")
    return 0


def print_agreement(results: dict) -> dict:
    """Print the agreement report; return it as plain dicts for saving."""
    summary = {}
    for criterion, a in results.items():
        if a.n == 0:
            print(f"\n{criterion}: no labels yet")
            continue
        kappa = "undefined (every label identical)" if a.kappa is None else a.kappa
        print(f"\n{criterion}  (n={a.n})")
        print(f"  accuracy:            {a.accuracy:.0%}")
        print(f"  Cohen's kappa:       {kappa}")
        print(f"  both pass / fail:    {a.both_pass} / {a.both_fail}")
        print(f"  judge too lenient:   {a.judge_pass_human_fail}  (passed something you failed)")
        print(f"  judge too strict:    {a.judge_fail_human_pass}  (failed something you passed)")
        if a.disagreements:
            print(f"  review these rows:   {', '.join(a.disagreements)}")
        summary[criterion] = a.__dict__
    return summary


def cmd_score(args) -> int:
    results = score_labels(args.labels, args.key)
    print("=== Judge vs. human agreement ===")
    summary = print_agreement(results)
    save("calibration", {"labels_file": Path(args.labels).name,
                         "key_file": Path(args.key).name if args.key else None,
                         "agreement": summary})
    return 0


def cmd_rejudge(args) -> int:
    labels = Path(args.labels)
    old_key_path = labels.with_suffix(".key.json")
    new_key_path = labels.with_suffix(f".{args.judge_prompt}.key.json")
    print(f"Re-judging {labels.name} with {args.judge_prompt}...")
    rejudge_label_rows(labels, judge_answer, out_key_path=new_key_path,
                       model=args.judge_model, prompt_version=args.judge_prompt)

    old_key = json.loads(old_key_path.read_text(encoding="utf-8"))
    new_key = json.loads(new_key_path.read_text(encoding="utf-8"))
    errors = [i for i, e in new_key.items() if e.get("judge_error")]

    print("\n=== Original judge (key from export) vs. human ===")
    before = print_agreement(score_labels(labels, old_key_path))
    print(f"\n=== {args.judge_prompt} vs. human ===")
    after = print_agreement(score_labels(labels, new_key_path))

    changes = changed_verdicts(old_key, new_key)
    print(f"\n=== Verdicts that changed ({len(changes)}) ===")
    for c in changes:
        print(f"  {c['label_id']} ({c['case_id']}) {c['criterion']}: {c['before']} -> {c['after']}")
    if errors:
        print(f"\nJudge errors on {len(errors)} rows (left out of the scores): {', '.join(errors)}")

    save("calibration", {"labels_file": labels.name, "judge_prompt_version": args.judge_prompt,
                         "key_file": new_key_path.name, "agreement_before": before,
                         "agreement": after, "changed_verdicts": changes, "judge_errors": errors})
    print(f"\nNew answer key: {new_key_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the LLM judge.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("mutate", help="Plant known errors and check the judge catches them")
    p.add_argument("--results", default=RESULTS_DIR / "latest.json")
    p.add_argument("--judge-model", default=None)
    p.add_argument("--judge-prompt", default=DEFAULT_JUDGE_PROMPT, choices=sorted(JUDGE_PROMPTS))
    p.set_defaults(func=cmd_mutate)

    p = sub.add_parser("export", help="Write a blind CSV for human labeling")
    p.add_argument("results", nargs="+", help="One or more results JSON files")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("score", help="Compare human labels with judge verdicts")
    p.add_argument("labels", help="The filled-in labeling CSV")
    p.add_argument("--key", default=None,
                   help="Answer key to score (default: the .key.json written by export)")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("rejudge", help="Re-grade already-labeled rows with another judge prompt")
    p.add_argument("labels", help="The filled-in labeling CSV")
    p.add_argument("--judge-prompt", default=DEFAULT_JUDGE_PROMPT, choices=sorted(JUDGE_PROMPTS))
    p.add_argument("--judge-model", default=None)
    p.set_defaults(func=cmd_rejudge)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
