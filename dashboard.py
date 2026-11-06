"""Phase 5: build the results dashboard (one HTML page) from saved results.

Usage:
    python dashboard.py                          # writes results/dashboard.html
    python dashboard.py --open                   # ... and opens it in your browser
    python dashboard.py --out my_dashboard.html  # somewhere else
    python dashboard.py --results-dir path/to/results

No API calls: it only reads the results files the eval scripts already saved,
so it's free to run as often as you like. Run it after any eval run (or after
compare_models.py) to refresh the page.

The page has five parts:
  1. Would the build pass?  every CI gate against each suite's *_latest.json
  2. Which model?           the model comparison (model_comparison_latest.json)
  3. How did we get here?   each suite's gated number over every saved run
  4. What needs a look?     cases that failed or were flagged in the latest runs
  5. Production             the latest monitor.py run per app (monitor_<app>_latest.json)

Exit code is always 0: this reports on the gates, it doesn't enforce them (CI does).
"""
import argparse
import webbrowser
from pathlib import Path

from app.paths import RESULTS_DIR
from evals.dashboard import build_dashboard, render_html


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build the results dashboard (no API calls).")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help="folder of results files (default: results/)")
    parser.add_argument("--out", type=Path, default=None,
                        help="where to write the page (default: <results-dir>/dashboard.html)")
    parser.add_argument("--open", action="store_true", help="open the page in your browser when done")
    args = parser.parse_args(argv)

    out = args.out or args.results_dir / "dashboard.html"
    data = build_dashboard(args.results_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(data), encoding="utf-8")

    t = data["gate_totals"]
    print(f"Gates: {t['passed']} passed, {t['failed']} failed, {t['not_run']} not run (of {t['total']})")
    rec = (data["comparison"] or {}).get("recommendation") or {}
    if rec:
        print(f"Model comparison pick: {rec.get('model') or 'none'}")
    fails = sum(1 for r in data["issues"] if r["severity"] == "fail")
    print(f"Open issues: {fails} counting toward a gate, {len(data['issues']) - fails} worth a look")
    for app, m in data["monitoring"].items():
        levels = [a["level"] for a in m.get("alerts", [])]
        print(f"Monitoring ({app}): {levels.count('alert')} alert(s), {levels.count('warn')} warning(s), "
              f"{len(m.get('review_queue', []))} in the review queue")
    print(f"Wrote {out}")
    if args.open:
        webbrowser.open(out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
