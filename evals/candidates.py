"""From flagged production requests to golden-set candidates (Phase 6 step 3).

The loop this closes: production shows a failure the evals never did (Run 20's
multi-fact question, Run 21's refusals without a tool call, Run 22's French
refusal). If that request never becomes a test case, the next prompt change
can bring the failure back and every CI gate stays green.

So every request that anything flagged becomes a candidate:

  * a monitor check queued it (evals/monitors.py)
  * two traces of it disagreed: one answered, one declined (evals/compare.py)
  * a person's review found a problem with it (data/traffic/reviews_v*.jsonl)

Each candidate carries every reason, the answers each app gave, and a priority.
What it doesn't carry is an answer key: production has none, and writing one
means reading the documents. A person does that, adds the case to a new
golden-set version with an `origin` saying which request it came from, and the
next run of this script shows the candidate as promoted. Candidates are never
written into a dataset automatically.

Priority, highest first:
  1  a reviewer found the reply wrong
  2  the two traces disagreed, or a reviewer found a rule break
  3  a reviewer found a pipeline problem (the reply was fine), or a monitor
     flagged it and nobody has reviewed that traffic yet
  4  a monitor flagged it, but a review of that traffic found nothing wrong:
     a false alarm, listed so the monitors' noise stays visible
"""
import json
from itertools import combinations
from pathlib import Path

from evals.compare import compare
from evals.monitors import monitor, view

PRIORITY_LABELS = {1: "wrong (reviewed)", 2: "disagreement or rule break",
                   3: "unreviewed flag or pipeline", 4: "false alarm"}


def load_traffic_files(folder) -> dict:
    """{case id: {"file": name, **row}} for every traffic_v*.jsonl in a folder."""
    out = {}
    for path in sorted(Path(folder).glob("traffic_v*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                out[row["id"]] = {"file": path.name, **row}
    return out


def _traffic_file(traces: list[dict], traffic: dict) -> str | None:
    """Which traffic file a trace file replayed: the traces say so since step 3;
    older traces are matched by their case ids."""
    named = {t["attributes"].get("traffic.file") for t in traces} - {None}
    if len(named) == 1:
        return named.pop()
    files = {traffic[c]["file"] for c in (t["attributes"].get("traffic.case_id") for t in traces)
             if c in traffic}
    return files.pop() if len(files) == 1 else None


def promoted_cases(*datasets) -> dict:
    """{traffic case id: [dataset case ids]} for every case with an `origin`."""
    out = {}
    for path in datasets:
        path = Path(path)
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                case = json.loads(line)
                origin = case.get("origin") or {}
                if origin.get("traffic_id"):
                    out.setdefault(origin["traffic_id"], []).append(case["id"])
    return out


def build_candidates(files: dict, reviews: list[dict], traffic: dict,
                     promoted: dict | None = None) -> dict:
    """files: {trace file name: traces}. Monitors run on each file, the comparison
    on every two files that replayed the same traffic file."""
    promoted = promoted or {}
    runs = {}
    for name, traces in files.items():
        result = monitor(traces)
        runs[name] = {"traces": traces, "monitor": result, "app": result["apps"][0]
                      if len(result["apps"]) == 1 else "mixed",
                      "traffic_file": _traffic_file(traces, traffic),
                      "views": {v.case_id: v for v in map(view, traces)}}

    comparisons = []
    for x, y in combinations(sorted(runs), 2):
        if runs[x]["traffic_file"] and runs[x]["traffic_file"] == runs[y]["traffic_file"]:
            comparisons.append((x, y, compare(runs[x]["traces"], runs[y]["traces"])))

    # A traffic file counts as reviewed for an app if the review lists any problem
    # for that app on that file: unlisted requests were read and found fine.
    reviewed = {(r["app"], r.get("traffic")) for r in reviews}
    reviewed |= {(r["app"], traffic[r["case_id"]]["file"]) for r in reviews if r["case_id"] in traffic}

    cands = {}

    def cand(case_id, question=""):
        if case_id not in cands:
            row = traffic.get(case_id, {})
            cands[case_id] = {"traffic_id": case_id, "traffic_file": row.get("file"),
                              "question": row.get("question") or question,
                              "kind": row.get("kind"), "reasons": [], "answers": {}}
        return cands[case_id]

    for name, run in runs.items():
        for q in run["monitor"]["review_queue"]:
            c = cand(q["case"], q["question"])
            c["reasons"].append({"source": "monitor", "app": q["app"], "trace_file": name,
                                 "trace_id": q["trace_id"],
                                 "checks": sorted({f["check"] for f in q["flags"]})})
    for x, y, result in comparisons:
        for r in result["rows"]:
            if r["kind"] in ("answer_vs_decline", "outcome_changed"):
                c = cand(r["case"], r["question"])
                c["reasons"].append({
                    "source": "comparison", "kind": r["kind"], "trace_files": [x, y],
                    "outcomes": {result["a"]["setup"]["label"]: r["a"]["outcome"],
                                 result["b"]["setup"]["label"]: r["b"]["outcome"]}})
    for r in reviews:
        if r["verdict"] != "ok":
            c = cand(r["case_id"])
            c["reasons"].append({"source": "review", "app": r["app"], "verdict": r["verdict"],
                                 "run": r.get("run"), "note": r.get("note", "")})

    # What each app answered, from every trace file that has the request.
    for name, run in runs.items():
        for case_id, c in cands.items():
            v = run["views"].get(case_id)
            if v is not None:
                c["answers"][f"{run['app']} ({name})"] = v.answer

    for case_id, c in cands.items():
        c["priority"] = _priority(c, reviewed)
        c["why"] = _why(c)
        c["promoted_to"] = promoted.get(case_id, [])
        c["status"] = "promoted" if c["promoted_to"] else (
            "false_alarm" if c["priority"] == 4 else "open")

    ordered = sorted(cands.values(), key=lambda c: (c["priority"], c["traffic_id"]))
    return {
        "trace_files": {n: {"app": r["app"], "traffic_file": r["traffic_file"],
                            "requests": len(r["traces"])} for n, r in runs.items()},
        "comparisons": [[x, y] for x, y, _ in comparisons],
        "candidates": ordered,
        "counts": {
            "candidates": len(ordered),
            "by_priority": {p: sum(1 for c in ordered if c["priority"] == p) for p in PRIORITY_LABELS},
            "promoted": sum(1 for c in ordered if c["status"] == "promoted"),
            "open": sum(1 for c in ordered if c["status"] == "open"),
            "false_alarms": sum(1 for c in ordered if c["status"] == "false_alarm"),
        },
    }


def _priority(c: dict, reviewed: set) -> int:
    verdicts = {r["verdict"] for r in c["reasons"] if r["source"] == "review"}
    if "wrong" in verdicts:
        return 1
    if "rule_break" in verdicts or any(r["source"] == "comparison" for r in c["reasons"]):
        return 2
    if "pipeline" in verdicts:
        return 3
    # Only monitor flags. Was that app's traffic reviewed? Then nothing was wrong.
    apps = {r["app"] for r in c["reasons"] if r["source"] == "monitor"}
    if apps and all((app, c["traffic_file"]) in reviewed for app in apps):
        return 4
    return 3


def _why(c: dict) -> str:
    parts = []
    for r in c["reasons"]:
        if r["source"] == "review":
            parts.append(f"review: {r['app']} {r['verdict']}")
        elif r["source"] == "comparison":
            # Across apps the app name is enough ("rag answered vs agent declined");
            # across releases both sides are the same app, so show the versions.
            apps = [k.split()[0] for k in r["outcomes"]]
            names = apps if len(set(apps)) == len(apps) else list(r["outcomes"])
            parts.append("comparison: " + " vs ".join(
                f"{n} {v}" for n, v in zip(names, r["outcomes"].values())))
        else:
            parts.append(f"{r['app']}: " + ", ".join(r["checks"]))
    seen, out = set(), []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return "; ".join(out)


def format_candidates(result: dict, width: int = 100, include_false_alarms: bool = False) -> str:
    n = result["counts"]
    lines = [f"{n['candidates']} candidate(s) from {len(result['trace_files'])} trace file(s) and "
             f"{len(result['comparisons'])} comparison(s): {n['promoted']} promoted, {n['open']} open, "
             f"{n['false_alarms']} false alarm(s)", ""]
    for p, label in PRIORITY_LABELS.items():
        rows = [c for c in result["candidates"] if c["priority"] == p]
        if not rows or (p == 4 and not include_false_alarms):
            continue
        lines.append(f"Priority {p}: {label} ({len(rows)})")
        for c in rows:
            status = (f"promoted to {', '.join(c['promoted_to'])}" if c["promoted_to"]
                      else c["status"])
            q = " ".join((c["question"] or "").split())
            q = q if len(q) <= width - 40 else q[: width - 43] + "..."
            lines.append(f"  {c['traffic_id']}  [{status}]  {q}")
            lines.append(f"      {c['why'][:width]}")
        lines.append("")
    if n["false_alarms"] and not include_false_alarms:
        lines.append(f"{n['false_alarms']} false alarm(s) not listed (--false-alarms to list)")
    return "\n".join(lines).rstrip()
