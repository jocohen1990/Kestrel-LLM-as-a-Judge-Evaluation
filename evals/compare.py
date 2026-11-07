"""Comparing two trace files of the same traffic (Phase 6 step 3).

The monitors (evals/monitors.py) read one trace at a time, and Run 22 showed
what that can't see: a wrong refusal is the exact refusal sentence, which is
also what a right refusal looks like. 0 of 2 wrong RAG answers were queued.

Two traces of the SAME request can see it. If one app (or one release)
answered and the other declined, at most one of them can be right, so a person
should read both. In Runs 20-22 that happened 7 times, and all 7 were real
problems, including every wrong refusal. That was noticed while reading Run 22,
so Run 23 counts it properly and says what it would and wouldn't catch.

A comparison is one of two kinds, decided by the traces themselves:

  across apps    the RAG assistant and the agent on the same traffic. Neither
                 is the baseline; a disagreement says "one of these is wrong".
  across releases the same app before and after a change (a new prompt,
                 retriever or model). A is the baseline, B the candidate, and
                 the question is whether B is better in production, not just
                 in the evals: which flags went away, which appeared, which
                 answers changed and need reading again.

Like the monitors, nothing here needs an answer key, and a difference means
"a person should look", not "B is wrong".
"""
from evals.monitors import ASK_BACK_RE, declined, monitor, view

# What a reply did. Only the first three are about the answer itself.
OUTCOMES = ("answered", "declined", "asked_back", "error", "no_content")

# Differences between the two traces of one request, most important first.
#   answer_vs_decline  one answered, the other declined: at most one is right
#   outcome_changed    any other change of outcome (asked back vs answered, an error)
#   flags_changed      same outcome, but a review or alert check fires on one side only
#   answer_changed     same outcome and flags, different words (releases only: two
#                      apps always word things differently)
#   missing            the request has a trace on one side only (a lost trace, Run 22)
KINDS = ("answer_vs_decline", "outcome_changed", "flags_changed", "answer_changed", "missing")

# Kinds that go into the comparison's review queue: the ones only a comparison
# can see. flags_changed doesn't: a flag on either side already put that request
# in that side's monitor queue, so queueing it again would only inflate the
# count (across apps it's mostly checks that exist for one app only, such as
# rewrite_not_a_query). It's still reported, because across releases "which
# flags went away, which appeared" is the point. answer_changed doesn't either:
# rewording is expected after a prompt change, and the old review has to be
# read again for those requests anyway (see stale_reviews).
QUEUED_KINDS = ("answer_vs_decline", "outcome_changed", "missing")

SERIOUS = ("alert", "review")


def outcome(v) -> str:
    """answered, declined, asked_back, error or no_content, for one TraceView."""
    if v.status == "error":
        return "error"
    if not v.content:
        return "no_content"
    if declined(v.answer):
        return "declined"
    if v.answer.rstrip().endswith("?") or ASK_BACK_RE.search(v.answer):
        return "asked_back"
    return "answered"


def setup_of(traces: list[dict]) -> dict:
    """The app and versions a trace file was recorded with. A file with more
    than one setup is reported as mixed."""
    keys = ("app.kind", "kestrel.prompt_version", "kestrel.retriever_version",
            "kestrel.agent_version", "kestrel.tools_version", "service.version")
    seen = {tuple((k, t["attributes"].get(k)) for k in keys) for t in traces}
    if len(seen) != 1:
        return {"app": "mixed", "label": "mixed setups", "versions": {}}
    attrs = dict(next(iter(seen)))
    models = sorted({m for t in traces for m in t["attributes"].get("llm.models", [])})
    versions = {k.split(".")[-1]: v for k, v in attrs.items()
                if v and k not in ("app.kind", "service.version")}
    if models:
        versions["model"] = ", ".join(m.removeprefix("claude-") for m in models)
    label = f"{attrs['app.kind']} " + " ".join(
        f"{k}={v}" for k, v in versions.items() if k != "model")
    return {"app": attrs["app.kind"], "label": label.strip(), "versions": versions}


def _flag_names(flags: list[dict]) -> list[str]:
    return sorted({f["check"] for f in flags if f["severity"] in SERIOUS})


def _norm(text: str) -> str:
    return " ".join((text or "").split()).lower()


def _side(v, result, cost) -> dict:
    flags = next((q["flags"] for q in result["review_queue"] if q["trace_id"] == v.trace_id), [])
    return {"trace_id": v.trace_id, "outcome": outcome(v), "answer": v.answer,
            "flags": _flag_names(flags), "duration_ms": v.duration_ms, "cost_usd": cost}


def classify(a: dict | None, b: dict | None, same_app: bool) -> str | None:
    """The most important difference between the two sides of one request, or None."""
    if a is None or b is None:
        return "missing"
    pair = {a["outcome"], b["outcome"]}
    if pair == {"answered", "declined"}:
        return "answer_vs_decline"
    if a["outcome"] != b["outcome"]:
        return "outcome_changed"
    if a["flags"] != b["flags"]:
        return "flags_changed"
    if same_app and _norm(a["answer"]) != _norm(b["answer"]):
        return "answer_changed"
    return None


def compare(a_traces: list[dict], b_traces: list[dict], judge_sample: float = 0.0,
            judge=None) -> dict:
    """Pair two trace files by traffic case id and report every difference.

    Both files are monitored first (same checks, same thresholds), so the
    comparison and the monitors can't disagree about what a flag is.
    """
    from evals.traces import trace_cost

    sides = {}
    for name, traces in (("a", a_traces), ("b", b_traces)):
        result = monitor(traces, judge_sample=judge_sample, judge=judge)
        views = {v.case_id: v for v in map(view, traces)}
        costs = {t["trace_id"]: trace_cost(t) for t in traces}
        sides[name] = {"setup": setup_of(traces), "monitor": result, "views": views,
                       "costs": costs}

    a, b = sides["a"], sides["b"]
    same_app = a["setup"]["app"] == b["setup"]["app"] != "mixed"
    cases = sorted(set(a["views"]) | set(b["views"]))
    rows = []
    for case in cases:
        va, vb = a["views"].get(case), b["views"].get(case)
        sa = _side(va, a["monitor"], a["costs"].get(va.trace_id)) if va else None
        sb = _side(vb, b["monitor"], b["costs"].get(vb.trace_id)) if vb else None
        kind = classify(sa, sb, same_app)
        rows.append({"case": case, "question": (va or vb).question, "kind": kind,
                     "a": sa, "b": sb,
                     "new_flags": sorted(set(sb["flags"]) - set(sa["flags"])) if sa and sb else [],
                     "resolved_flags": (sorted(set(sa["flags"]) - set(sb["flags"]))
                                        if sa and sb else [])})

    def queued_on(side):
        return {q["case"] for q in sides[side]["monitor"]["review_queue"]}

    differences = [r for r in rows if r["kind"]]
    queue = [r for r in differences if r["kind"] in QUEUED_KINDS]
    already = queued_on("a") | queued_on("b")
    return {
        "mode": "releases" if same_app else "apps",
        "a": {"setup": a["setup"], "requests": len(a["views"]),
              "summary": side_summary(a["monitor"])},
        "b": {"setup": b["setup"], "requests": len(b["views"]),
              "summary": side_summary(b["monitor"])},
        "paired": sum(1 for r in rows if r["a"] and r["b"]),
        "only_in_a": [r["case"] for r in rows if r["a"] and not r["b"]],
        "only_in_b": [r["case"] for r in rows if r["b"] and not r["a"]],
        "counts": {k: sum(1 for r in rows if r["kind"] == k) for k in KINDS},
        "rows": rows,
        "queue": [r["case"] for r in queue],
        # What the comparison adds: queued here, and in neither side's monitor queue.
        "new_to_review": [r["case"] for r in queue if r["case"] not in already],
        "monitor_queues": {"a": sorted(queued_on("a")), "b": sorted(queued_on("b"))},
        "monitors_version": a["monitor"]["monitors_version"],
    }


def side_summary(result: dict) -> dict:
    """The window numbers worth putting side by side."""
    checks = result["checks"]

    def rate(name):
        return checks.get(name, {}).get("rate")

    return {
        "error_rate": result["stats"]["error_rate"],
        "p95_latency_ms": result["stats"]["p95_latency_ms"],
        "cost_per_1000_requests": result["stats"]["cost_per_1000_requests"],
        "refusal_rate": rate("refusal"),
        "asked_back_rate": rate("asked_back"),
        "review_queue": len(result["review_queue"]),
        "alerts": sum(1 for x in result["alerts"] if x["level"] == "alert"),
        "warnings": sum(1 for x in result["alerts"] if x["level"] == "warn"),
        "check_counts": {name: c["count"] for name, c in checks.items()
                         if c["severity"] in SERIOUS and c["count"]},
    }


# ---- Checking the comparison against a person's review -------------------------------

def _problems(reviews: list[dict] | None, app: str) -> dict:
    return {r["case_id"]: r for r in (reviews or []) if r["app"] == app and r["verdict"] != "ok"}


def stale_reviews(result: dict) -> list[str]:
    """Across releases, a review made of A's replies says nothing about a B reply
    that changed. These requests need reading again before B's numbers mean anything."""
    if result["mode"] != "releases":
        return []
    return [r["case"] for r in result["rows"]
            if r["a"] and r["b"] and _norm(r["a"]["answer"]) != _norm(r["b"]["answer"])]


def score_against_reviews(result: dict, reviews_a: list[dict] | None,
                          reviews_b: list[dict] | None) -> dict:
    """How well the comparison's queue finds the problems a person found.

    reviews_a / reviews_b list the problems in each side's replies (the
    monitor review format, data/traffic/reviews_v*.jsonl), filtered here by
    each side's app. Across apps one file usually covers both. Across releases
    pass B's own review: A's review is only true of A's replies.

    A difference counts as a hit if either side of it had a problem: it
    asks a person to read both replies, and either can be the wrong one.
    """
    pa = _problems(reviews_a, result["a"]["setup"]["app"])
    pb = _problems(reviews_b, result["b"]["setup"]["app"])
    paired = {r["case"] for r in result["rows"] if r["a"] and r["b"]}
    pa = {c: p for c, p in pa.items() if c in paired}
    pb = {c: p for c, p in pb.items() if c in paired}
    with_problem = set(pa) | set(pb)
    queue = set(result["queue"]) & paired
    monitors = (set(result["monitor_queues"]["a"]) | set(result["monitor_queues"]["b"])) & paired
    wrong = ({c for c, p in pa.items() if p["verdict"] == "wrong"}
             | {c for c, p in pb.items() if p["verdict"] == "wrong"})

    def recall(found, of):
        return round(len(found & of) / len(of), 3) if of else None

    per_kind = {}
    for kind in KINDS:
        cases = {r["case"] for r in result["rows"] if r["kind"] == kind} & paired
        if cases:
            per_kind[kind] = {"count": len(cases), "with_problem": len(cases & with_problem),
                              "cases": sorted(cases),
                              "false_alarms": sorted(cases - with_problem)}
    side_of = {}
    for case in sorted(queue & with_problem):
        side_of[case] = "both" if case in pa and case in pb else ("a" if case in pa else "b")
    return {
        "requests_with_problem": len(with_problem),
        "wrong": sorted(wrong),
        "queued": len(queue),
        "queued_with_problem": len(queue & with_problem),
        "precision": round(len(queue & with_problem) / len(queue), 3) if queue else None,
        "recall": recall(queue, with_problem),
        "recall_wrong": recall(queue, wrong),
        "monitors_alone": {"queued": len(monitors), "recall": recall(monitors, with_problem),
                           "recall_wrong": recall(monitors, wrong)},
        "monitors_plus_comparison": {
            "queued": len(monitors | queue),
            "recall": recall(monitors | queue, with_problem),
            "recall_wrong": recall(monitors | queue, wrong),
            "missed": sorted(with_problem - monitors - queue),
            "missed_wrong": sorted(wrong - monitors - queue)},
        "by_kind": per_kind,
        "problem_side": side_of,
        "stale_reviews": stale_reviews(result) if reviews_b is None else [],
    }


# ---- Text output ------------------------------------------------------------------------

def _pct(x) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def _ms(x) -> str:
    return "n/a" if x is None else f"{x / 1000:.2f}s"


def _money(x) -> str:
    return "n/a" if x is None else f"${x:.2f}"


def _short(text: str, width: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= width else text[: width - 3] + "..."


def format_comparison(result: dict, scores: dict | None = None, width: int = 100,
                      show_all: bool = False) -> str:
    a, b = result["a"], result["b"]
    head = ("Across releases (A is the baseline, B the candidate)" if result["mode"] == "releases"
            else "Across apps (neither is the baseline)")
    lines = [head,
             f"  A: {a['setup']['label']}  ({a['requests']} traces)",
             f"  B: {b['setup']['label']}  ({b['requests']} traces)",
             f"  {result['paired']} requests in both; monitors {result['monitors_version']} on each",
             ""]
    for side, cases in (("A", result["only_in_a"]), ("B", result["only_in_b"])):
        if cases:
            lines.append(f"WARNING: only in {side}: {', '.join(cases)} (a lost trace, or "
                         "different traffic)")
    sa, sb = a["summary"], b["summary"]
    lines.append(f"{'':<26}{'A':>10}{'B':>10}")
    for label, key, fmt in [("Error rate", "error_rate", _pct), ("p95 latency", "p95_latency_ms", _ms),
                            ("Cost per 1,000 requests", "cost_per_1000_requests", _money),
                            ("Refusal rate", "refusal_rate", _pct),
                            ("Asked back", "asked_back_rate", _pct),
                            ("Review queue", "review_queue", str), ("Alerts", "alerts", str)]:
        lines.append(f"{label:<26}{fmt(sa[key]):>10}{fmt(sb[key]):>10}")
    checks = sorted(set(sa["check_counts"]) | set(sb["check_counts"]))
    if checks:
        lines.append("Review and alert checks:")
        for name in checks:
            lines.append(f"  {name:<24}{sa['check_counts'].get(name, 0):>10}"
                         f"{sb['check_counts'].get(name, 0):>10}")
    lines.append("")
    c = result["counts"]
    lines.append("Differences: " + ", ".join(f"{k} {c[k]}" for k in KINDS if c[k]) if any(c.values())
                 else "No differences.")
    lines.append(f"Comparison queue: {len(result['queue'])} request(s); "
                 f"{len(result['new_to_review'])} not already in either monitor queue"
                 + (f" ({', '.join(result['new_to_review'])})" if result["new_to_review"] else ""))
    lines.append("")
    listed = QUEUED_KINDS + (("flags_changed",) if result["mode"] == "releases" else ())
    for r in result["rows"]:
        if not r["kind"] or (r["kind"] not in listed and not show_all):
            continue
        lines.append(f"{r['case']}  [{r['kind']}]  Q: {_short(r['question'], width - 30)}")
        for side in ("a", "b"):
            s = r[side]
            if s is None:
                lines.append(f"   {side.upper()}: (no trace)")
                continue
            flags = f"  [{', '.join(s['flags'])}]" if s["flags"] else ""
            lines.append(f"   {side.upper()}: {s['outcome']:<10} {_short(s['answer'], width - 20)}{flags}")
        if result["mode"] == "releases" and (r["new_flags"] or r["resolved_flags"]):
            parts = []
            if r["resolved_flags"]:
                parts.append("resolved " + ", ".join(r["resolved_flags"]))
            if r["new_flags"]:
                parts.append("new " + ", ".join(r["new_flags"]))
            lines.append("   " + "; ".join(parts))
    hidden = [k for k in KINDS if k not in listed and c[k]]
    if hidden and not show_all:
        lines.append("\nNot listed (--all to list): " + ", ".join(f"{k} {c[k]}" for k in hidden)
                     + ("; a flag on one side is already in that side's monitor queue"
                        if "flags_changed" in hidden else ""))
    if scores:
        lines.append("")
        lines.append(f"Against the human review: {scores['requests_with_problem']} paired requests had "
                     f"a problem ({len(scores['wrong'])} wrong: {', '.join(scores['wrong']) or 'none'})")
        lines.append(f"  Comparison queue: {scores['queued_with_problem']} of {scores['queued']} had a "
                     f"problem (precision {_pct(scores['precision'])}); recall {_pct(scores['recall'])}, "
                     f"wrong answers {_pct(scores['recall_wrong'])}")
        for kind, k in scores["by_kind"].items():
            if kind not in QUEUED_KINDS:
                continue
            fa = f"; no problem found: {', '.join(k['false_alarms'])}" if k["false_alarms"] else ""
            lines.append(f"    {kind}: {k['with_problem']} of {k['count']}{fa}")
        mo, mp = scores["monitors_alone"], scores["monitors_plus_comparison"]
        lines.append(f"  Monitors alone (both queues): {mo['queued']} queued, recall {_pct(mo['recall'])}, "
                     f"wrong answers {_pct(mo['recall_wrong'])}")
        lines.append(f"  Monitors + comparison: {mp['queued']} queued, recall {_pct(mp['recall'])}, "
                     f"wrong answers {_pct(mp['recall_wrong'])}")
        if mp["missed"]:
            lines.append(f"    still missed: {', '.join(mp['missed'])}")
        if scores["stale_reviews"]:
            lines.append(f"  NOTE: {len(scores['stale_reviews'])} of B's replies differ from A's, so the "
                         f"review (made of A's replies) doesn't cover them: "
                         f"{', '.join(scores['stale_reviews'])}. Read them, or pass --reviews-b.")
    return "\n".join(lines)
