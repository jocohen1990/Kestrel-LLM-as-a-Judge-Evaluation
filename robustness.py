"""Robustness eval: ask every golden question reworded, casually, and with typos.

Usage:
    python robustness.py --retrieval-only      # free: retrieval checks only, no API calls
    python robustness.py                       # all styles, default prompt (186 assistant calls)
    python robustness.py --styles original typo typos3
    python robustness.py --ids g001 g022
    python robustness.py --min-robustness 0.9  # exit 1 if fewer than 90% of passing
                                               # questions still pass when varied

No judge calls: every check is rule-based. See evals/robustness.py for what's
measured and why, and data/robustness/README.md for the dataset.
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
from app.paths import CORPUS_DIR, GOLDEN_DIR, RESULTS_DIR, ROBUSTNESS_DIR
from app.rag import DEFAULT_PROMPT_VERSION, PROMPTS, RagAssistant
from app.retriever import Retriever, load_chunks
from evals.dataset import load_golden
from evals.robustness import (STYLES, build_variants, load_hand_variants, retrieval_row,
                              run_variant, summarize_robustness)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the robustness evals.")
    parser.add_argument("--golden", default=GOLDEN_DIR / "golden_v3.jsonl")
    parser.add_argument("--variants", default=ROBUSTNESS_DIR / "paraphrases_v1.jsonl")
    parser.add_argument("--styles", nargs="*", default=list(STYLES), choices=STYLES)
    parser.add_argument("--ids", nargs="*", default=None, help="Only these golden case ids")
    parser.add_argument("--model", default=None, help="Assistant model")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT_VERSION, choices=sorted(PROMPTS))
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Only check retrieval (free, no API key needed)")
    parser.add_argument("--min-robustness", type=float, default=None,
                        help="Fail (exit 1) if overall robustness is below this, e.g. 0.9")
    return parser.parse_args()


def fmt(value) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def print_report(run: dict) -> None:
    s = run["summary"]
    print(f"\n=== Robustness: {s['questions']} questions, {s['variants']} variants ===")
    if run["retrieval_only"]:
        print("Retrieval only (no model calls)")
    else:
        print(f"Assistant: {run['model']} (prompt {run['prompt_version']}, k={run['k']})")
        print(f"  Overall robustness: {fmt(s['overall_robustness'])} "
              f"({s['flips']} flips: {s['flips_retrieval']} retrieval, {s['flips_model']} model)")
    print(f"  {'style':<10} {'passages':>9} {'pass':>6} {'robust':>7}  flips")
    for style, e in s["by_style"].items():
        flips = ""
        if e.get("flips"):
            flips = ", ".join(f"{i} ({'retrieval' if i in e['flips_retrieval'] else 'model'})"
                              for i in e["flips"])
        print(f"  {style:<10} {fmt(e['passage_hit_rate']):>9} {fmt(e.get('pass_rate')):>6} "
              f"{fmt(e.get('robustness')):>7}  {flips}")
    for style, e in s["by_style"].items():
        if e.get("passages_lost"):
            print(f"  Answer's paragraph lost ({style}): {', '.join(e['passages_lost'])}")
        if e.get("gained"):
            print(f"  Failed as written, passed as {style}: {', '.join(e['gained'])}")


def main() -> int:
    args = parse_args()
    golden = load_golden(args.golden)
    hand = load_hand_variants(args.variants, golden)
    if args.ids:
        golden = [c for c in golden if c["id"] in set(args.ids)]
    styles = list(args.styles)
    # The original is the baseline every flip is measured against, so always ask it.
    if "original" not in styles:
        styles.insert(0, "original")
    variants = build_variants(golden, hand, styles)
    if not variants:
        print("No cases match those filters.")
        return 1
    cases = {c["id"]: c for c in golden}

    rows = []
    if args.retrieval_only:
        retriever = Retriever(load_chunks(CORPUS_DIR))
        for v in variants:
            rows.append(retrieval_row(v, cases[v["source_id"]], retriever, k=args.k))
    else:
        assistant = RagAssistant(k=args.k, model=args.model, prompt_version=args.prompt)
        print(f"{len(variants)} assistant calls ({len(golden)} questions x up to {len(styles)} styles)")
        warmup(args.model)
        for v in variants:
            row = run_variant(v, cases[v["source_id"]], assistant)
            rows.append(row)
            hit = {True: "", False: "  [paragraph NOT retrieved]", None: ""}[row["passage_hit"]]
            print(f"[{'PASS' if row['passed'] else 'FAIL'}] {row['id']:<14} {row['question'][:70]}{hit}")

    model = args.model or os.getenv("LLM_MODEL", DEFAULT_MODEL)
    run = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "retrieval_only": args.retrieval_only,
        "model": None if args.retrieval_only else model,
        "prompt_version": None if args.retrieval_only else args.prompt,
        "assistant_temperature_sent": None if args.retrieval_only else sends_temperature(model),
        "golden_file": Path(args.golden).name,
        "variants_file": Path(args.variants).name,
        "styles": styles,
        "k": args.k,
        "summary": summarize_robustness(rows),
        "cases": rows,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    prefix = "robustness_retrieval" if args.retrieval_only else "robustness"
    for name in (f"{prefix}_{stamp}.json", f"{prefix}_latest.json"):
        (RESULTS_DIR / name).write_text(json.dumps(run, indent=2), encoding="utf-8")

    print_report(run)
    print(f"\nSaved results/{prefix}_latest.json")

    score = run["summary"].get("overall_robustness")
    if args.min_robustness is not None and score is not None and score < args.min_robustness:
        print(f"\nFAIL: robustness {score:.0%} is below the {args.min_robustness:.0%} minimum")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
