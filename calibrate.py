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
from evals.calibration import export_for_labeling, score_labels
from evals.dataset import load_golden
from evals.judge import JudgeError, judge_answer
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
                                       m["answer"], model=args.judge_model)
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
                       "judge_model": args.judge_model, "by_type": dict(by_type), "cases": rows})
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


def cmd_score(args) -> int:
    results = score_labels(args.labels)
    print("=== Judge vs. human agreement ===")
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
    save("calibration", {"labels_file": Path(args.labels).name, "agreement": summary})
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the LLM judge.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("mutate", help="Plant known errors and check the judge catches them")
    p.add_argument("--results", default=RESULTS_DIR / "latest.json")
    p.add_argument("--judge-model", default=None)
    p.set_defaults(func=cmd_mutate)

    p = sub.add_parser("export", help="Write a blind CSV for human labeling")
    p.add_argument("results", nargs="+", help="One or more results JSON files")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("score", help="Compare human labels with judge verdicts")
    p.add_argument("labels", help="The filled-in labeling CSV")
    p.set_defaults(func=cmd_score)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
