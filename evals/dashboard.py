"""Results dashboard: one HTML page built from the results files already saved.

It answers five questions, in this order:

  1. Would the build pass right now?   Every CI gate, from the *_latest.json files.
  2. Which model should we use?         The model comparison (model_comparison_latest.json).
  3. How did we get here?               Each suite's headline number over every saved run.
  4. What still needs a look?           The cases that failed or were flagged in the latest runs.
  5. How is production doing?           The latest monitor run per app (monitor_<app>_latest.json,
                                        Phase 6): alerts, what each check flagged, the review queue.

Like evals/scorecard.py, nothing here calls an API: it only reads files, so it
is free to rebuild and can be tested offline. The page is one self-contained
HTML file (no internet, no libraries), so it opens anywhere and can be shared
as-is.

The gates come from evals/scorecard.py, which copies .github/workflows/ci.yml.
So "passes" on this page means the same thing it means in CI.
"""
import html
import json
from datetime import datetime
from pathlib import Path

from app.agent import DEFAULT_AGENT_VERSION
from app.llm import DEFAULT_MODEL
from app.rag import DEFAULT_PROMPT_VERSION, DEFAULT_RETRIEVER_VERSION
from evals.judge import DEFAULT_JUDGE_MODEL, DEFAULT_JUDGE_PROMPT
from evals.pricing import base_name
from evals.scorecard import SUITES, Suite, run_matches, suite_result

COMPARISON_FILE = "model_comparison_latest.json"
MONITOR_APPS = ("rag", "agent")

# The one number per suite that the history charts follow. Each is a gated metric,
# so the chart can draw the CI line it has to stay on the right side of.
HEADLINE = {
    "golden": "pass_rate",
    "injection": "attack_success_rate",
    "jailbreak": "attack_success_rate",
    "bias": "verdict_differences",
    "agent": "fully_correct_rate",
    "robustness": "overall_robustness",
}

# Which setting is the dataset for each suite. History only compares runs on the
# same dataset: golden_v1 had 20 questions, golden_v3 has 32, so they don't line up.
DATASET_KEY = {
    "golden": "golden_file",
    "injection": "attacks_file",
    "jailbreak": "cases_file",
    "bias": "cases_file",
    "agent": "cases_file",
    "robustness": "variants_file",
}


# ---- Reading files ---------------------------------------------------------------

def load_json(path: Path) -> dict | None:
    """The file's contents, or None if it's missing or isn't valid JSON."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def metric_value(suite: Suite, key: str, run: dict):
    metric = next(m for m in suite.metrics if m.key == key)
    return metric.get(run.get("summary", {}))


# ---- 1. Gates --------------------------------------------------------------------

def gate_status(results_dir: Path) -> list[dict]:
    """One entry per suite: its latest run and how each of its CI gates went.

    `default_setup` is False when the latest file wasn't made with the default
    model and settings (for example, a quick test of an old prompt). The gates
    are still shown, but the page says the run isn't the setup CI checks.
    """
    out = []
    for suite in SUITES.values():
        run = load_json(Path(results_dir) / suite.latest)
        if run is None or not run.get("model"):
            out.append({"suite": suite.name, "label": suite.label, "run": None})
            continue
        result = suite_result(suite, run, source=suite.latest)
        out.append({
            "suite": suite.name,
            "label": suite.label,
            "run": result,
            "default_setup": run_matches(run, suite, DEFAULT_MODEL),
            "cases": len(run.get("cases", [])),
        })
    return out


def gate_totals(gates: list[dict]) -> dict:
    checked = [g for s in gates if s["run"] for g in s["run"]["gates"]]
    expected = sum(len(SUITES[s["suite"]].gates) for s in gates)
    return {
        "passed": sum(1 for g in checked if g["passed"]),
        "failed": sum(1 for g in checked if g["passed"] is False),
        "not_run": expected - len([g for g in checked if g["passed"] is not None]),
        "total": expected,
    }


# ---- 3. History ------------------------------------------------------------------

def version_label(suite_name: str, run: dict) -> str:
    """A short name for the setup a run used, for the x-axis: 'v6 r3', 'a3'."""
    if suite_name == "agent":
        return run.get("agent_version") or "?"
    label = run.get("prompt_version") or "v1"
    retriever = run.get("retriever_version")
    if retriever and retriever != "r1":
        label += f" {retriever}"
    return label


def history(results_dir: Path, model: str = DEFAULT_MODEL) -> dict[str, dict]:
    """Every saved full run of each suite with one model, oldest first.

    Kept: runs with this model on the dataset the gates use. Dropped: other
    models (those belong to the comparison), other datasets, and partial runs
    (a re-run of 2 cases to check a fix isn't the suite's score). A partial run
    is one with fewer cases than the biggest run of that suite.

    The timestamped files are in .gitignore, so this is the history saved on the
    machine that builds the page. On a fresh clone or in CI it may be one point.
    """
    out = {}
    for name, suite in SUITES.items():
        key = DATASET_KEY[name]
        runs = []
        for path in sorted(Path(results_dir).glob(f"{suite.prefix}*.json")):
            if path.name == suite.latest:
                continue
            run = load_json(path)
            if not run or not run.get("model") or not run.get("timestamp"):
                continue
            if base_name(run["model"]) != base_name(model):
                continue
            if run.get(key) != suite.match.get(key):
                continue
            # "robustness_*" also matches the robustness_retrieval_* files, which
            # check retrieval only and have no answers to score.
            if name == "robustness" and run.get("retrieval_only"):
                continue
            runs.append((path.name, run))
        if not runs:
            out[name] = {"suite": name, "label": suite.label, "points": []}
            continue
        full = max(len(r.get("cases", [])) for _, r in runs)
        runs = [(p, r) for p, r in runs if len(r.get("cases", [])) == full]
        runs.sort(key=lambda pr: pr[1]["timestamp"])
        metric_key = HEADLINE[name]
        metric = next(m for m in suite.metrics if m.key == metric_key)
        gate = next(g for g in suite.gates if g.metric == metric_key)
        points = [{
            "file": p,
            "timestamp": r["timestamp"],
            "label": version_label(name, r),
            "value": metric_value(suite, metric_key, r),
        } for p, r in runs]
        out[name] = {
            "suite": name,
            "label": suite.label,
            "metric": metric_key,
            "metric_label": metric.label,
            "kind": metric.kind,
            "higher_is_better": metric.higher_is_better,
            "gate": {"op": gate.op, "limit": gate.limit},
            "cases": full,
            "points": [pt for pt in points if pt["value"] is not None],
        }
    return out


# ---- 4. Open issues ----------------------------------------------------------------

def _ids(summary: dict, key: str) -> list[str]:
    return list(summary.get(key) or [])


def open_issues(results_dir: Path) -> list[dict]:
    """What failed or was flagged in each suite's latest run, one row per finding.

    `severity` is "fail" when the case counts against a gated number (it may
    still be inside the gate's allowance: 1 of 21 injections is under 5%) and
    "flag" when it's worth a look but no gate counts it (word flags, internal
    names, a judge disagreeing with a rule check that passed).
    """
    rows = []

    def add(suite, ids, what, severity):
        if ids:
            rows.append({"suite": suite, "ids": ids, "what": what, "severity": severity})

    golden = load_json(Path(results_dir) / SUITES["golden"].latest)
    if golden:
        for case in golden.get("cases", []):
            if not case.get("passed"):
                add("golden", [case["id"]], f"Rule check failed: {case.get('reason') or 'no reason saved'}",
                    "fail")
            elif case.get("judge_correct") is False:
                add("golden", [case["id"]], "Judge marked the answer not correct", "flag")
            if case.get("judge_faithful") is False:
                add("golden", [case["id"]], "Judge marked the answer not faithful to the documents",
                    "fail")
        add("golden", [c["id"] for c in golden.get("cases", []) if c.get("mentions_internals")],
            "Answer mentions internal names", "flag")

    inj = load_json(Path(results_dir) / SUITES["injection"].latest)
    if inj:
        s = inj.get("summary", {})
        add("injection", _ids(s, "succeeded_ids"), "Attack worked", "fail")
        add("injection", _ids(s, "task_broken_ids"), "Attack stopped the real question being answered",
            "flag")
        add("injection", _ids(s, "prompt_leak_ids"), "System prompt leaked", "flag")
        add("injection", _ids(s, "mentions_internals_ids"), "Answer mentions internal names", "flag")

    jb = load_json(Path(results_dir) / SUITES["jailbreak"].latest)
    if jb:
        s = jb.get("summary", {})
        add("jailbreak", _ids(s, "succeeded_ids"), "Jailbreak worked", "fail")
        add("jailbreak", _ids(s, "over_refused_ids"), "Safe question refused", "fail")
        add("jailbreak", _ids(s, "inexact_refusal_ids"), "Refused, but not with the exact sentence", "flag")
        add("jailbreak", _ids(s, "prompt_leak_ids"), "System prompt leaked", "flag")
        add("jailbreak", _ids(s, "mentions_internals_ids"), "Answer mentions internal names", "flag")

    bias = load_json(Path(results_dir) / SUITES["bias"].latest)
    if bias:
        s = bias.get("summary", {})
        add("bias", _ids(s, "verdict_differs_ids"), "Different verdict depending on who asked", "fail")
        add("bias", _ids(s, "flagged_ids"), "Answer used a flagged word (verdict unchanged)", "flag")

    agent = load_json(Path(results_dir) / SUITES["agent"].latest)
    if agent:
        s = agent.get("summary", {})
        add("agent", _ids(s, "wrong_answer_ids"), "Wrong answer", "fail")
        add("agent", _ids(s, "right_answer_wrong_path_ids"),
            "Right answer, wrong path (tools, arguments or order)", "fail")
        add("agent", _ids(s, "unsupported_number_ids"), "Answer has a number no tool returned", "flag")
        add("agent", _ids(s, "empty_answer_ids"), "No answer", "fail")
        add("agent", _ids(s, "mentions_internals_ids"), "Answer mentions internal names", "flag")

    rob = load_json(Path(results_dir) / SUITES["robustness"].latest)
    if rob:
        by_style = rob.get("summary", {}).get("by_style", {})
        flips = [i for style in by_style.values() for i in style.get("flips", [])]
        add("robustness", flips, "Passed as written, failed when reworded or misspelled", "fail")
    return rows


# ---- Everything the page needs ------------------------------------------------------

def build_dashboard(results_dir: Path) -> dict:
    results_dir = Path(results_dir)
    gates = gate_status(results_dir)
    return {
        "default_setup": {
            "model": DEFAULT_MODEL,
            "prompt": DEFAULT_PROMPT_VERSION,
            "retriever": DEFAULT_RETRIEVER_VERSION,
            "agent": DEFAULT_AGENT_VERSION,
            "judge": f"{DEFAULT_JUDGE_MODEL} ({DEFAULT_JUDGE_PROMPT})",
        },
        "gates": gates,
        "gate_totals": gate_totals(gates),
        "comparison": load_json(results_dir / COMPARISON_FILE),
        "history": history(results_dir),
        "issues": open_issues(results_dir),
        "monitoring": {app: m for app in MONITOR_APPS
                       if (m := load_json(results_dir / f"monitor_{app}_latest.json"))},
    }


# ---- HTML -------------------------------------------------------------------------

def esc(value) -> str:
    return html.escape(str(value), quote=True)


def fmt(value, kind: str = "rate") -> str:
    if value is None:
        return "n/a"
    if kind == "count":
        return str(value)
    pct = value * 100
    return f"{pct:.0f}%" if abs(pct - round(pct)) < 0.05 else f"{pct:.1f}%"


def fmt_time(ts: str | None) -> str:
    if not ts:
        return "n/a"
    try:
        return datetime.fromisoformat(ts).strftime("%b %d, %Y %H:%M UTC").replace(" 0", " ", 1)
    except ValueError:
        return ts


def short_model(model: str) -> str:
    """'claude-haiku-4-5-20251001' -> 'Haiku 4.5'."""
    parts = base_name(model).split("-")
    if len(parts) >= 3 and parts[0] == "claude":
        return f"{parts[1].capitalize()} {'.'.join(parts[2:])}"
    return model


def status_badge(passed) -> str:
    if passed is True:
        return '<span class="badge good"><span aria-hidden="true">✓</span> Pass</span>'
    if passed is False:
        return '<span class="badge bad"><span aria-hidden="true">✕</span> Fail</span>'
    return '<span class="badge none"><span aria-hidden="true">–</span> Not run</span>'


def _gate_text(gate: dict, kind: str) -> str:
    op = "≥" if gate["op"] == ">=" else "≤"
    return f"{op} {fmt(gate['limit'], kind)}"


def _metric_meta(suite_name: str, key: str):
    return next(m for m in SUITES[suite_name].metrics if m.key == key)


def render_gates(data: dict) -> str:
    t = data["gate_totals"]
    if t["failed"]:
        headline = f"{t['failed']} of {t['total']} gates failing"
        cls = "bad"
    elif t["not_run"]:
        headline = f"{t['passed']} of {t['total']} gates passing, {t['not_run']} not run"
        cls = "none"
    else:
        headline = f"All {t['total']} gates passing"
        cls = "good"
    rows = []
    for s in data["gates"]:
        run = s["run"]
        if run is None:
            for gate in SUITES[s["suite"]].gates:
                meta = _metric_meta(s["suite"], gate.metric)
                rows.append(f"<tr><td>{esc(s['label'])}</td><td>{esc(meta.label)}</td>"
                            f"<td class='num'>n/a</td><td class='num'>"
                            f"{esc(_gate_text({'op': gate.op, 'limit': gate.limit}, meta.kind))}</td>"
                            f"<td>{status_badge(None)}</td><td class='muted'>no results file</td></tr>")
            continue
        note = f"{esc(short_model(run['model']))} · {esc(fmt_time(run['timestamp']))}"
        if not s["default_setup"]:
            note += " · <strong>not the default setup</strong>"
        for gate in run["gates"]:
            meta = _metric_meta(s["suite"], gate["metric"])
            rows.append(
                f"<tr><td>{esc(s['label'])}</td><td>{esc(meta.label)}</td>"
                f"<td class='num'>{esc(fmt(gate['value'], meta.kind))}</td>"
                f"<td class='num'>{esc(_gate_text(gate, meta.kind))}</td>"
                f"<td>{status_badge(gate['passed'])}</td><td class='muted nowrap'>{note}</td></tr>")
    return f"""
<section>
  <h2>Would the build pass?</h2>
  <p class="lede">The same gates as <code>.github/workflows/ci.yml</code>, checked against each suite's
  latest results file.</p>
  <div class="hero {cls}"><span class="hero-num">{t['passed']}/{t['total']}</span>
    <span class="hero-label">{esc(headline)}</span></div>
  <div class="table-wrap"><table>
    <thead><tr><th>Suite</th><th>Gate</th><th class="num">Latest</th><th class="num">Needs</th>
    <th>Status</th><th>Run</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table></div>
</section>"""


def hbar_chart(title: str, items: list[tuple[str, float, str]], unit_fmt, highlight: str | None = None) -> str:
    """Horizontal bars, one per model: (label, value, tooltip). One series, so no legend."""
    items = [i for i in items if i[1] is not None]
    if not items:
        return ""
    width, label_w, value_w, bar_h, gap = 440, 82, 64, 22, 14
    plot_w = width - label_w - value_w
    top = max(v for _, v, _ in items) or 1
    height = len(items) * (bar_h + gap) + 4
    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">']
    parts.append(f'<line class="axis" x1="{label_w}" y1="0" x2="{label_w}" y2="{height}"/>')
    for i, (label, value, tip) in enumerate(items):
        y = i * (bar_h + gap) + 2
        w = max(2.0, plot_w * value / top)
        r = min(4, w / 2)
        # Rounded at the data end only, square at the baseline.
        path = (f"M{label_w},{y} H{label_w + w - r} Q{label_w + w},{y} {label_w + w},{y + r} "
                f"V{y + bar_h - r} Q{label_w + w},{y + bar_h} {label_w + w - r},{y + bar_h} H{label_w} Z")
        cls = "bar hl" if label == highlight else "bar"
        parts.append(f'<g class="hit" data-tip="{esc(tip)}" tabindex="0">'
                     f'<rect class="hit-area" x="0" y="{y - gap / 2}" width="{width}" height="{bar_h + gap}"/>'
                     f'<path class="{cls}" d="{path}"/>'
                     f'<text class="tick" x="{label_w - 8}" y="{y + bar_h / 2}" text-anchor="end" '
                     f'dominant-baseline="central">{esc(label)}</text>'
                     f'<text class="val" x="{label_w + w + 8}" y="{y + bar_h / 2}" '
                     f'dominant-baseline="central">{esc(unit_fmt(value))}</text></g>')
    parts.append("</svg>")
    return f'<figure><figcaption>{esc(title)}</figcaption>{"".join(parts)}</figure>'


def render_comparison(data: dict) -> str:
    card = data["comparison"]
    if not card or not card.get("models"):
        return """
<section>
  <h2>Which model?</h2>
  <p class="empty">No model comparison saved yet. Run <code>python compare_models.py</code>.</p>
</section>"""
    models = card["models"]
    names = [m["model"] for m in models]
    rec = card.get("recommendation") or {}
    rec_model = rec.get("model")

    head = "".join(f'<th class="num">{esc(short_model(n))}'
                   f'{"<br><span class=muted>recommended</span>" if n == rec_model else ""}</th>' for n in names)
    body = []
    for suite_name in card["suites"]:
        body.append(f'<tr class="group"><th colspan="{len(names) + 1}">'
                    f'{esc(card["suite_labels"][suite_name])}</th></tr>')
        for key, meta in card["metric_labels"][suite_name].items():
            cells = []
            for n in names:
                result = card["results"].get(n, {}).get(suite_name)
                if result is None:
                    cells.append('<td class="num muted">not run</td>')
                    continue
                gate = next((g for g in result["gates"] if g["metric"] == key), None)
                cell = esc(fmt(result["metrics"].get(key), meta["kind"]))
                if gate and gate["passed"] is False:
                    cell += ' <span class="badge bad">✕ Fail</span>'
                cells.append(f'<td class="num">{cell}</td>')
            hint = "" if meta["higher_is_better"] else ' <span class="muted">(lower is better)</span>'
            body.append(f'<tr><td>{esc(meta["label"])}{hint}</td>{"".join(cells)}</tr>')
    by = {m["model"]: m for m in models}
    body.append(f'<tr class="group"><th colspan="{len(names) + 1}">Overall</th></tr>')
    body.append("<tr><td>CI gates passed</td>" + "".join(
        f'<td class="num">{by[n]["gates_passed"]}/{by[n]["gates_total"]}</td>' for n in names) + "</tr>")
    body.append("<tr><td>temperature=0 accepted</td>" + "".join(
        f'<td class="num">{"yes" if by[n]["temperature_sent"] is True else "no" if by[n]["temperature_sent"] is False else "n/a"}</td>'
        for n in names) + "</tr>")
    body.append("<tr><td>Graded by itself (judge is the same model)</td>" + "".join(
        f'<td class="num">{"yes" if by[n]["graded_by_itself"] else "no"}</td>' for n in names) + "</tr>")

    cost = hbar_chart(
        "Cost per 1,000 golden-set questions (assistant only)",
        [(short_model(n), by[n]["cost_per_1k_questions_usd"],
          f"{short_model(n)}: ${by[n]['cost_per_1k_questions_usd']:.2f} per 1,000 questions"
          if by[n]["cost_per_1k_questions_usd"] is not None else "") for n in names],
        lambda v: f"${v:.2f}", highlight=short_model(rec_model) if rec_model else None)
    latency = hbar_chart(
        "Average answer time on the golden set",
        [(short_model(n), by[n]["avg_latency_s"],
          f"{short_model(n)}: {by[n]['avg_latency_s']:.2f}s average, {by[n]['p95_latency_s']:.2f}s p95"
          if by[n]["avg_latency_s"] is not None else "") for n in names],
        lambda v: f"{v:.2f}s")

    caveats = []
    for m in models:
        if m["temperature_sent"] is False:
            caveats.append(f"{esc(short_model(m['model']))} doesn't accept temperature=0, so its answers vary "
                           "more between runs and one run says less about it.")
        if m["graded_by_itself"]:
            caveats.append(f"{esc(short_model(m['model']))} is also the judge, so its judge scores may be "
                           "generous. Its rule-based scores aren't affected.")
    caveats.append(f"Each model ran each suite once ({esc(fmt_time(card.get('timestamp')))}). With 32 golden "
                   "questions one question is about 3 points, so small differences are ties: "
                   "the gates decide, not the averages.")
    caveats.append(f"Prices checked {esc(card.get('prices_checked', 'n/a'))} (<code>evals/pricing.py</code>).")
    rec_text = (f"<strong>Scorecard pick: {esc(short_model(rec_model))}.</strong> {esc(rec.get('reason', ''))}"
                if rec_model else f"<strong>No pick.</strong> {esc(rec.get('reason', ''))}")
    default_note = ""
    if rec_model and base_name(rec_model) != base_name(data["default_setup"]["model"]):
        default_note = (f'<p class="note">The scorecard pick isn\'t the default model '
                        f'({esc(short_model(data["default_setup"]["model"]))}). The pick is the cheapest model '
                        f'that passes every gate; switching is a separate decision (see the README\'s findings).</p>')
    return f"""
<section>
  <h2>Which model?</h2>
  <p class="lede">The same evals, settings and judge on each model; only the assistant model changes.</p>
  <div class="callout">{rec_text}</div>
  {default_note}
  <div class="charts two">{cost}{latency}</div>
  <div class="table-wrap"><table>
    <thead><tr><th></th>{head}</tr></thead>
    <tbody>{''.join(body)}</tbody>
  </table></div>
  <ul class="caveats">{''.join(f'<li>{c}</li>' for c in caveats)}</ul>
</section>"""


def _nice_domain(h: dict) -> tuple[float, float, float]:
    """(low, high, tick step) for a history chart, on round numbers.

    Rates that should be high get a zoomed axis ending at 100% (the gate and every
    run are in the top quarter, so a 0-100% axis would flatten them into one line).
    Rates that should be low start at 0.
    """
    values = [p["value"] for p in h["points"]] + [h["gate"]["limit"]]
    if h["kind"] == "count":
        return 0, max(max(values) + 1, 2), 1
    lo, hi = min(values), max(values)
    if h["higher_is_better"]:
        lo, hi = max(0.0, lo - 0.05), 1.0
    else:
        lo, hi = 0.0, max(0.2, hi + 0.05)
    step = 0.05 if hi - lo <= 0.3 else 0.1 if hi - lo <= 0.6 else 0.25
    lo = int(lo / step + 1e-9) * step
    hi = -int(-hi / step - 1e-9) * step
    return round(lo, 4), round(min(hi, 1.0) if h["higher_is_better"] else hi, 4), step


def line_chart(h: dict) -> str:
    """One suite's headline number across runs: a line, a dot per run, the CI line."""
    pts = h["points"]
    width, height = 420, 220
    left, right, top, bottom = 46, 16, 16, 40
    pw, ph = width - left - right, height - top - bottom
    lo, hi, step = _nice_domain(h)

    def y(v):
        return top + ph - (v - lo) / (hi - lo) * ph

    n = len(pts)
    def x(i):
        return left + (pw / 2 if n == 1 else i * pw / (n - 1))

    out = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
           f'aria-label="{esc(h["label"])}: {esc(h["metric_label"])} by run">']
    for k in range(int(round((hi - lo) / step)) + 1):
        v = lo + k * step
        out.append(f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>')
        out.append(f'<text class="tick" x="{left - 6}" y="{y(v):.1f}" text-anchor="end" '
                   f'dominant-baseline="central">{esc(fmt(v if h["kind"] != "count" else int(round(v)), h["kind"]))}</text>')
    g = h["gate"]["limit"]
    gate_word = "min" if h["gate"]["op"] == ">=" else "max"
    out.append(f'<line class="gate" x1="{left}" x2="{width - right}" y1="{y(g):.1f}" y2="{y(g):.1f}"/>')
    # On the left, so it never sits under the latest value's label on the right.
    out.append(f'<text class="gate-label" x="{left + 14}" y="{y(g) - 6:.1f}" text-anchor="start">'
               f'CI gate ({gate_word} {esc(fmt(g, h["kind"]))})</text>')
    if n > 1:
        d = " ".join(f"{'M' if i == 0 else 'L'}{x(i):.1f},{y(p['value']):.1f}" for i, p in enumerate(pts))
        out.append(f'<path class="line" d="{d}"/>')
    for i, p in enumerate(pts):
        tip = (f"{p['label']} · {fmt(p['value'], h['kind'])} · {fmt_time(p['timestamp'])} · {p['file']}")
        out.append(f'<g class="hit" data-tip="{esc(tip)}" tabindex="0">'
                   f'<circle class="hit-area" cx="{x(i):.1f}" cy="{y(p["value"]):.1f}" r="12"/>'
                   f'<circle class="dot" cx="{x(i):.1f}" cy="{y(p["value"]):.1f}" r="4.5"/></g>')
        out.append(f'<text class="tick" x="{x(i):.1f}" y="{height - bottom + 16}" text-anchor="middle">'
                   f'{esc(p["label"])}</text>')
    last = pts[-1]
    lx, ly = x(n - 1), y(last["value"])
    anchor = "end" if n > 1 else "start"
    dx = -8 if n > 1 else 8
    out.append(f'<text class="val" x="{lx + dx:.1f}" y="{ly - 10:.1f}" text-anchor="{anchor}">'
               f'{esc(fmt(last["value"], h["kind"]))}</text>')
    out.append(f'<text class="axis-title" x="{left + pw / 2}" y="{height - 4}" text-anchor="middle">'
               f'setup used, oldest run → newest</text>')
    out.append("</svg>")
    return "".join(out)


def render_history(data: dict) -> str:
    figures, table_rows = [], []
    for name, h in data["history"].items():
        if not h["points"]:
            figures.append(f'<figure><figcaption>{esc(h["label"])}</figcaption>'
                           f'<p class="empty">No saved full runs with the default model.</p></figure>')
            continue
        sub = f'{esc(h["metric_label"])}{"" if h["higher_is_better"] else " (lower is better)"}'
        one = ('<p class="muted small">Only one saved run here: the timestamped results files are '
               'kept on the machine that ran them, not in git.</p>' if len(h["points"]) == 1 else "")
        figures.append(f'<figure><figcaption>{esc(h["label"])} <span class="muted">· {sub} · '
                       f'{h["cases"]} cases</span></figcaption>{line_chart(h)}{one}</figure>')
        for p in h["points"]:
            table_rows.append(f"<tr><td>{esc(h['label'])}</td><td>{esc(p['label'])}</td>"
                              f"<td class='num'>{esc(fmt(p['value'], h['kind']))}</td>"
                              f"<td>{esc(fmt_time(p['timestamp']))}</td><td><code>{esc(p['file'])}</code></td></tr>")
    table = ""
    if table_rows:
        table = f"""<details><summary>Show the numbers as a table</summary>
  <div class="table-wrap"><table><thead><tr><th>Suite</th><th>Setup</th><th class="num">Value</th>
  <th>When</th><th>File</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></div></details>"""
    return f"""
<section>
  <h2>How did we get here?</h2>
  <p class="lede">Each suite's gated number on every saved full run of the default model
  ({esc(short_model(data["default_setup"]["model"]))}), labelled by the prompt, retriever or agent version
  it used. Partial re-runs and other datasets are left out. The README's Findings section explains each change.</p>
  <div class="charts">{''.join(figures)}</div>
  {table}
</section>"""


def render_issues(data: dict) -> str:
    rows = data["issues"]
    if not rows:
        body = '<p class="empty">Nothing failed or was flagged in the latest runs.</p>'
    else:
        trs = []
        order = list(SUITES)
        for r in sorted(rows, key=lambda r: (r["severity"] != "fail", order.index(r["suite"]))):
            badge = ('<span class="badge bad">✕ Counts toward a gate</span>' if r["severity"] == "fail"
                     else '<span class="badge warn">! Worth a look</span>')
            ids = ", ".join(f"<code>{esc(i)}</code>" for i in r["ids"])
            trs.append(f"<tr><td>{badge}</td><td>{esc(SUITES[r['suite']].label)}</td>"
                       f"<td>{esc(r['what'])}</td><td>{ids}</td></tr>")
        body = f"""<div class="table-wrap"><table><thead><tr><th>Type</th><th>Suite</th><th>What happened</th>
  <th>Cases</th></tr></thead><tbody>{''.join(trs)}</tbody></table></div>"""
    return f"""
<section>
  <h2>What still needs a look?</h2>
  <p class="lede">From each suite's latest run. "Counts toward a gate" means the case lowers a gated
  number; the gate can still pass, because most gates allow a little room (up to 5% of injection
  attacks, for example). "Worth a look" isn't counted by any gate.</p>
  {body}
</section>"""


ALERT_BADGES = {
    "alert": '<span class="badge bad"><span aria-hidden="true">✕</span> Alert</span>',
    "warn": '<span class="badge warn"><span aria-hidden="true">!</span> Warning</span>',
    "info": '<span class="badge none"><span aria-hidden="true">–</span> Note</span>',
}
APP_LABELS = {"rag": "RAG assistant", "agent": "Agent"}


def _monitor_card(app: str, m: dict) -> str:
    stats = m.get("stats", {})
    alerts = m.get("alerts", [])
    worst = ("alert" if any(a["level"] == "alert" for a in alerts)
             else "warn" if any(a["level"] == "warn" for a in alerts) else None)
    headline = {"alert": "Alerts firing", "warn": "Warnings"}.get(worst, "No alerts")
    cls = {"alert": "bad", "warn": "warn"}.get(worst, "good")
    p95, cost = stats.get("p95_latency_ms"), stats.get("cost_per_1000_requests")
    facts = [f"{m.get('requests', 0)} requests", f"from <code>{esc(m.get('trace_file', '?'))}</code>",
             f"p95 {p95 / 1000:.2f}s" if p95 is not None else "p95 n/a",
             f"${cost:.2f} per 1,000" if cost is not None else "cost n/a",
             f"{len(m.get('review_queue', []))} in the review queue"]
    alert_rows = "".join(
        f"<tr><td>{ALERT_BADGES.get(a['level'], esc(a['level']))}</td><td>{esc(a['message'])}</td>"
        f"<td class=\"small\">{', '.join(f'<code>{esc(c)}</code>' for c in a['cases'][:8])}"
        f"{' …' if len(a['cases']) > 8 else ''}</td></tr>" for a in alerts)
    check_rows = "".join(
        f"<tr><td><code>{esc(name)}</code></td><td class=\"small muted\">{esc(c['severity'])}</td>"
        f"<td>{esc(c['what'])}</td><td class=\"num\">{c['count']}</td>"
        f"<td class=\"small\">{', '.join(f'<code>{esc(x)}</code>' for x in c['cases'][:6])}"
        f"{' …' if len(c['cases']) > 6 else ''}</td></tr>"
        for name, c in m.get("checks", {}).items() if c["count"])
    judge = m.get("judge", {})
    judge_line = (f"<p class=\"note\">Judge (faithfulness, {fmt(judge['sample_rate'])} sample): "
                  f"{judge['judged']} judged, {judge['unfaithful']} with an unsupported claim, "
                  f"{judge['errors']} error(s).</p>" if judge.get("sample_rate") else
                  '<p class="note">The judge wasn\'t run (<code>--judge-sample</code> 0).</p>')
    score = (m.get("review_scores") or {}).get(app)
    score_line = ""
    if score:
        score_line = (f"<p class=\"note\">Against a person's review "
                      f"(<code>{esc(m.get('reviews_file', ''))}</code>): {score['queued_with_problem']} "
                      f"of {score['queued']} queued requests had a problem (precision "
                      f"{fmt(score['precision'])}); {score['queued_with_problem']} of "
                      f"{score['problems']} problems were queued (recall {fmt(score['recall'])}).</p>")
    return f"""
  <figure>
    <figcaption>{esc(APP_LABELS.get(app, app))} <span class="badge {cls}">{esc(headline)}</span></figcaption>
    <p class="note small">{' · '.join(facts)}</p>
    {f'<div class="table-wrap"><table><tbody>{alert_rows}</tbody></table></div>' if alert_rows else ''}
    {judge_line}{score_line}
    <details><summary>What each check flagged</summary>
    <div class="table-wrap"><table><thead><tr><th>Check</th><th>Severity</th><th>What it looks for</th>
    <th class="num">Requests</th><th>Cases</th></tr></thead><tbody>{check_rows or
    '<tr><td colspan="5" class="muted">Nothing flagged.</td></tr>'}</tbody></table></div></details>
  </figure>"""


def render_monitoring(data: dict) -> str:
    mon = data.get("monitoring") or {}
    if not mon:
        body = ('<p class="empty">No monitor results yet. Record traces with <code>python traffic.py</code>, '
                'then run <code>python monitor.py</code>.</p>')
    else:
        body = "".join(_monitor_card(app, m) for app, m in mon.items())
    return f"""
<section>
  <h2>How is production doing?</h2>
  <p class="lede">The latest <code>monitor.py</code> run for each app, on recorded traces (Phase 6). Production
  has no answer keys, so these checks look at what a trace alone can show: the app's own rules, claims about the
  documents, each step doing its job, and (on a sample) whether the judge finds every claim supported. A flag means
  a person should look, not that the answer is wrong.</p>
  {body}
</section>"""


CSS = """
.viz-root{color-scheme:light;--surface-0:#f6f5f2;--surface-1:#fcfcfb;--border:#e4e2dc;--grid:#ebe9e4;
--text-primary:#0b0b0b;--text-secondary:#52514e;--text-muted:#6f6e69;--series-1:#2a78d6;--series-1-wash:#2a78d61a;
--good:#0ca30c;--good-text:#006300;--critical:#d03b3b;--critical-text:#b02a2a;--warning:#fab219;--warning-text:#8a5a00;
--hl:#1c5cab}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;
--surface-0:#121211;--surface-1:#1a1a19;--border:#2e2e2b;--grid:#2a2a27;--text-primary:#f0efec;--text-secondary:#c3c2b7;
--text-muted:#9a998f;--series-1:#3987e5;--series-1-wash:#3987e526;--good-text:#4cc94c;--critical-text:#f08080;
--warning-text:#fab219;--hl:#86b6ef}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--surface-0:#121211;--surface-1:#1a1a19;--border:#2e2e2b;
--grid:#2a2a27;--text-primary:#f0efec;--text-secondary:#c3c2b7;--text-muted:#9a998f;--series-1:#3987e5;
--series-1-wash:#3987e526;--good-text:#4cc94c;--critical-text:#f08080;--warning-text:#fab219;--hl:#86b6ef}
*{box-sizing:border-box}
body{margin:0;background:var(--surface-0)}
.viz-root{font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;color:var(--text-primary);
background:var(--surface-0);min-height:100vh}
main{max-width:1080px;margin:0 auto;padding:32px 20px 64px}
header h1{font-size:28px;margin:0 0 4px;letter-spacing:-.01em}
header p{margin:0;color:var(--text-secondary)}
.setup{display:flex;flex-wrap:wrap;gap:8px;margin-top:14px}
.chip{background:var(--surface-1);border:1px solid var(--border);border-radius:999px;padding:3px 12px;font-size:13px;
color:var(--text-secondary)}
.chip b{color:var(--text-primary);font-weight:600}
section{background:var(--surface-1);border:1px solid var(--border);border-radius:14px;padding:24px;margin-top:20px}
h2{font-size:19px;margin:0 0 4px}
.lede{margin:0 0 16px;color:var(--text-secondary);max-width:75ch}
.hero{display:flex;align-items:baseline;gap:14px;margin:0 0 18px}
.hero-num{font-size:44px;font-weight:650;line-height:1;font-variant-numeric:tabular-nums}
.hero-label{font-size:17px;font-weight:600}
.hero.good .hero-label{color:var(--good-text)}.hero.bad .hero-label{color:var(--critical-text)}
.table-wrap{overflow-x:auto}
.table-wrap table{min-width:640px}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--grid);vertical-align:top}
thead th{font-weight:600;color:var(--text-secondary);font-size:13px;border-bottom:1px solid var(--border)}
tr.group th{padding-top:16px;font-size:13px;color:var(--text-secondary);text-transform:uppercase;letter-spacing:.04em}
.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.muted{color:var(--text-muted)}.nowrap{white-space:nowrap}.small{font-size:13px}
code{font:12.5px ui-monospace,SFMono-Regular,Consolas,monospace;background:var(--surface-0);padding:1px 5px;
border-radius:5px;border:1px solid var(--grid)}
.badge{display:inline-flex;gap:4px;align-items:center;font-size:12.5px;font-weight:600;padding:1px 8px;border-radius:999px;
border:1px solid currentColor;white-space:nowrap}
.badge.good{color:var(--good-text)}.badge.bad{color:var(--critical-text)}.badge.warn{color:var(--warning-text)}
.badge.none{color:var(--text-muted)}
figure + figure{margin-top:14px}
.callout{border-left:3px solid var(--series-1);background:var(--series-1-wash);padding:12px 16px;border-radius:0 8px 8px 0;
margin-bottom:12px}
.note{color:var(--text-secondary);margin:0 0 12px}
.caveats{color:var(--text-secondary);font-size:13.5px;margin:16px 0 0;padding-left:20px}
.caveats li{margin:3px 0}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:16px;margin:8px 0 16px}
figure{margin:0;border:1px solid var(--grid);border-radius:10px;padding:14px}
figcaption{font-weight:600;font-size:14px;margin-bottom:8px}
figcaption .muted{font-weight:400}
.chart{width:100%;height:auto;display:block;overflow:visible}
.chart .grid,.chart .axis{stroke:var(--grid);stroke-width:1}
.chart .tick{fill:var(--text-muted);font-size:11.5px}
.chart .val{fill:var(--text-primary);font-size:12.5px;font-weight:600}
.chart .axis-title{fill:var(--text-muted);font-size:11px}
.chart .gate{stroke:var(--critical);stroke-width:1.25;opacity:.75}
.chart .gate-label{fill:var(--text-secondary);font-size:11px;paint-order:stroke;stroke:var(--surface-1);
stroke-width:4px;stroke-linejoin:round}
.chart .line{fill:none;stroke:var(--series-1);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.chart .dot{fill:var(--series-1);stroke:var(--surface-1);stroke-width:2}
.chart .bar{fill:var(--series-1)}.chart .bar.hl{fill:var(--hl)}
.chart .hit-area{fill:transparent}
.chart .hit:hover .dot,.chart .hit:focus .dot{r:6}
.chart .hit:hover .bar,.chart .hit:focus .bar{opacity:.85}
.chart .hit:focus{outline:none}
.empty{color:var(--text-muted);margin:0}
details{margin-top:4px}summary{cursor:pointer;color:var(--text-secondary);font-size:14px}
#tip{position:fixed;pointer-events:none;background:var(--text-primary);color:var(--surface-1);font-size:12.5px;
padding:6px 9px;border-radius:6px;max-width:320px;opacity:0;transition:opacity .08s;z-index:10}
footer{color:var(--text-muted);font-size:13px;margin-top:24px}
@media (max-width:520px){.charts{grid-template-columns:1fr}section{padding:16px}}
"""

JS = """
(function(){var tip=document.getElementById('tip');
function show(e,t){tip.textContent=t;tip.style.opacity=1;var x=e.clientX||0,y=e.clientY||0;
if(!e.clientX){var r=e.target.getBoundingClientRect();x=r.left+r.width/2;y=r.top;}
var w=tip.offsetWidth;tip.style.left=Math.min(window.innerWidth-w-8,x+12)+'px';tip.style.top=(y-36)+'px';}
document.querySelectorAll('[data-tip]').forEach(function(el){var t=el.getAttribute('data-tip');if(!t)return;
el.addEventListener('mousemove',function(e){show(e,t)});el.addEventListener('focus',function(e){show(e,t)});
el.addEventListener('mouseleave',function(){tip.style.opacity=0});el.addEventListener('blur',function(){tip.style.opacity=0});});
})();
"""


def render_html(data: dict, generated_at: str | None = None) -> str:
    setup = data["default_setup"]
    generated_at = generated_at or datetime.now().astimezone().strftime("%b %d, %Y %H:%M %Z")
    chips = "".join(f'<span class="chip">{esc(k)} <b>{esc(v)}</b></span>' for k, v in (
        ("Model", short_model(setup["model"])), ("Prompt", setup["prompt"]), ("Retriever", setup["retriever"]),
        ("Agent", setup["agent"]), ("Judge", setup["judge"])))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM Evaluation Dashboard</title>
<style>{CSS}</style>
</head>
<body>
<div class="viz-root">
<main>
<header>
  <h1>LLM Evaluation Dashboard</h1>
  <p>A RAG assistant and a tool-using agent over a fictional company's financial documents (Kestrel Freight
  Analytics), graded by rules, an LLM judge, and safety, robustness and bias probes.</p>
  <div class="setup">{chips}</div>
</header>
{render_gates(data)}
{render_comparison(data)}
{render_history(data)}
{render_issues(data)}
{render_monitoring(data)}
<footer>Built {esc(generated_at)} by <code>python dashboard.py</code> from the files in <code>results/</code>.
No API calls; rebuild any time.</footer>
</main>
<div id="tip" role="tooltip"></div>
</div>
<script>{JS}</script>
</body>
</html>
"""
