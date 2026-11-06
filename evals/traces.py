"""Reading production traces (Phase 6 step 1).

app/tracing.py writes one JSON object per request to a .jsonl file. This module
turns a file of them into what an on-call engineer looks at first:

  * a summary: how many requests, how many failed, how slow (p50 and p95),
    which step the time went to, how many model calls and tokens per request,
    and what it all cost
  * one trace as a tree, to see exactly what happened in one request

Cost is worked out here from the token counts (evals/pricing.py), not stored in
the trace, for the same reason as the results files: prices change, and a trace
should stay true after they do.

Nothing here grades a reply. Judging whether production answers were any good,
without knowing the right answers, is the next step (monitoring).
"""
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from app.tracing import TRACE_SCHEMA
from evals.pricing import cost_usd


# ---- Loading -------------------------------------------------------------------

def trace_files(folder) -> list[Path]:
    """Trace files in a folder, oldest first (the names start with a timestamp)."""
    return sorted(Path(folder).glob("traces_*.jsonl"))


def load_traces(path) -> list[dict]:
    """Every trace in a .jsonl file, or in every traces_*.jsonl file of a folder.

    A line that isn't valid JSON, or was written by a different schema, is
    skipped with a warning instead of stopping the read: a request that died
    halfway through writing its trace shouldn't hide all the others.
    """
    path = Path(path)
    files = trace_files(path) if path.is_dir() else [path]
    traces, skipped = [], 0
    for file in files:
        for line in file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if record.get("schema") != TRACE_SCHEMA:
                skipped += 1
                continue
            traces.append(record)
    if skipped:
        print(f"warning: skipped {skipped} unreadable or other-schema line(s)")
    return traces


def find_trace(traces: list[dict], key: str) -> dict | None:
    """A trace by the start of its trace id, its request id, or its traffic case id."""
    for t in traces:
        attrs = t["attributes"]
        if (t["trace_id"].startswith(key) or attrs.get("request.id") == key
                or attrs.get("traffic.case_id") == key):
            return t
    return None


# ---- Numbers -------------------------------------------------------------------

def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile: q% of values are at most this. Same method as
    the p95 in evals/report.py, so trace and eval latencies compare."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(math.ceil(q / 100 * len(ordered)) - 1, 0)]


def latency_stats(values: list[float]) -> dict:
    return {
        "count": len(values),
        "avg_ms": round(sum(values) / len(values), 1) if values else None,
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "max_ms": max(values) if values else None,
    }


def llm_spans(trace: dict) -> list[dict]:
    return [s for s in trace["spans"] if s["kind"] == "llm"]


def trace_cost(trace: dict) -> float | None:
    """Dollar cost of one request's model calls. None if a model isn't priced."""
    total = 0.0
    for s in llm_spans(trace):
        a = s["attributes"]
        model = a.get("gen_ai.response.model") or a.get("gen_ai.request.model") or ""
        cost = cost_usd(model, a.get("gen_ai.usage.input_tokens") or 0,
                        a.get("gen_ai.usage.output_tokens") or 0,
                        a.get("gen_ai.usage.cache_read_input_tokens") or 0,
                        a.get("gen_ai.usage.cache_creation_input_tokens") or 0)
        if cost is None:
            return None
        total += cost
    return total


def summarize(traces: list[dict]) -> dict:
    """Everything the summary report shows, as plain numbers (easy to test, easy
    to save as JSON for the monitoring step to compare against)."""
    n = len(traces)
    roots = [t["duration_ms"] for t in traces]
    errors = [t for t in traces if t["status"] == "error"]

    by_span = defaultdict(list)
    for t in traces:
        for s in t["spans"]:
            if s["kind"] != "request":
                by_span[s["name"]].append(s["duration_ms"])

    costs = [trace_cost(t) for t in traces]
    priced = [c for c in costs if c is not None]
    attrs = [t["attributes"] for t in traces]

    def avg(key):
        values = [a.get(key) or 0 for a in attrs]
        return round(sum(values) / n, 1) if n else None

    agent_runs = [s["attributes"] for t in traces for s in t["spans"] if s["name"] == "agent.run"]
    tools = Counter(s["name"].removeprefix("tool.") for t in traces for s in t["spans"]
                    if s["kind"] == "tool")
    tool_errors = sum(1 for t in traces for s in t["spans"]
                      if s["kind"] == "tool" and s["attributes"].get("tool.is_error"))
    top_scores = [s["attributes"]["retrieval.top_score"] for t in traces for s in t["spans"]
                  if s["name"] == "bm25.search" and "retrieval.top_score" in s["attributes"]]

    versions = Counter()
    for a in attrs:
        keys = ("app.kind", "service.version", "kestrel.prompt_version",
                "kestrel.retriever_version", "kestrel.agent_version")
        versions[" ".join(f"{k.split('.')[-1]}={a[k]}" for k in keys if a.get(k))] += 1

    return {
        "requests": n,
        "errors": len(errors),
        "error_rate": round(len(errors) / n, 3) if n else None,
        "error_messages": Counter(
            next((s["error"] for s in t["spans"] if s["status"] == "error"), "?") for t in errors),
        "first_start": min((t["start_time"] for t in traces), default=None),
        "last_start": max((t["start_time"] for t in traces), default=None),
        "setups": dict(versions),
        "models": dict(Counter(m for a in attrs for m in a.get("llm.models", []))),
        "content_captured": sum(1 for t in traces if t.get("content_captured")),
        "latency": latency_stats(roots),
        # in the order steps first appear, which is pipeline order
        "span_latency": {name: latency_stats(v) for name, v in by_span.items()},
        "llm_calls_per_request": avg("llm.calls"),
        "input_tokens_per_request": avg("usage.input_tokens"),
        "output_tokens_per_request": avg("usage.output_tokens"),
        "cost_usd": round(sum(priced), 6) if priced else None,
        "cost_per_1000_requests": round(1000 * sum(priced) / len(priced), 4) if priced else None,
        "unpriced_requests": len(costs) - len(priced),
        "retrieval_top_score": {"p05": percentile(top_scores, 5), "p50": percentile(top_scores, 50),
                                "min": min(top_scores) if top_scores else None,
                                "count": len(top_scores)},
        "agent": {
            "runs": len(agent_runs),
            "stopped": dict(Counter(a.get("agent.stopped") for a in agent_runs)),
            "avg_turns": (round(sum(a.get("agent.turns", 0) for a in agent_runs)
                                / len(agent_runs), 2) if agent_runs else None),
            "tool_calls": dict(tools),
            "tool_errors": tool_errors,
        },
    }


def slowest(traces: list[dict], n: int = 5) -> list[dict]:
    return sorted(traces, key=lambda t: t["duration_ms"], reverse=True)[:n]


# ---- Text output ---------------------------------------------------------------

def _ms(value) -> str:
    if value is None:
        return "n/a"
    return f"{value / 1000:.2f}s" if value >= 1000 else f"{value:.0f}ms"


def short_model(model: str) -> str:
    return model.removeprefix("claude-")


def describe_span(span: dict) -> str:
    """The few attributes worth showing for each kind of span, on one line."""
    a = span["attributes"]
    kind, parts = span["kind"], []
    if kind == "llm":
        model = a.get("gen_ai.response.model") or a.get("gen_ai.request.model") or "?"
        parts.append(short_model(model))
        parts.append(f"{a.get('gen_ai.usage.input_tokens', '?')} in / "
                     f"{a.get('gen_ai.usage.output_tokens', '?')} out")
        reasons = [r for r in a.get("gen_ai.response.finish_reasons") or [] if r]
        if reasons:
            parts.append(f"stop={reasons[0]}")
        if a.get("llm.tool_calls"):
            parts.append(f"{a['llm.tool_calls']} tool call(s)")
    elif span["name"] == "bm25.search":
        if "retrieval.docs" in a:
            docs = Counter(d.removesuffix(".md") for d in a["retrieval.docs"])
            parts.append(", ".join(f"{d} x{c}" if c > 1 else d for d, c in docs.items()))
            parts.append(f"top score {a.get('retrieval.top_score')}")
        if a.get("retrieval.corrections"):
            parts.append("corrected " + ", ".join(f"{k}->{v}"
                                                  for k, v in a["retrieval.corrections"].items()))
    elif span["name"] == "query_rewrite" and "rewrite.output" in a:
        parts.append(f'"{shorten(a["rewrite.output"], 60)}"')
    elif kind == "tool":
        if "tool.arguments" in a:
            parts.append(shorten(a["tool.arguments"], 60))
        if a.get("tool.is_error"):
            parts.append("TOOL ERROR")
    elif span["name"] == "agent.turn":
        if a.get("agent.tool_calls"):
            parts.append("asks for " + ", ".join(a["agent.tool_calls"]))
        else:
            parts.append("answers")
    elif span["name"] == "agent.run":
        parts.append(f"{a.get('agent.stopped')} after {a.get('agent.turns')} turn(s)")
    elif kind == "request":
        parts.append(f"{a.get('app.kind')} {a.get('request.channel')}")
        if a.get("traffic.case_id"):
            parts.append(a["traffic.case_id"])
        parts.append(f"{a.get('llm.calls', 0)} model call(s)")
    if span["status"] == "error":
        parts.append(f"ERROR {span['error']}")
    return "  ".join(parts)


def shorten(text: str, width: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 3] + "..."


def render_tree(trace: dict) -> str:
    """One trace as an indented tree, children in the order they started."""
    children = defaultdict(list)
    for s in trace["spans"]:
        children[s["parent_id"]].append(s)
    lines = [f"trace {trace['trace_id']}  {trace['start_time']}  {_ms(trace['duration_ms'])}  "
             f"{trace['status']}"]
    question = trace["attributes"].get("input.question")
    if question:
        lines.append(f"  Q: {shorten(question, 100)}")

    def walk(parent_id, depth):
        for s in children.get(parent_id, []):
            label = f"{'  ' * depth}{s['name']}"
            lines.append(f"  {label:<34} {_ms(s['duration_ms']):>8}  {describe_span(s)}")
            walk(s["span_id"], depth + 1)

    walk(None, 0)
    answer = trace["attributes"].get("output.answer")
    if answer:
        lines.append(f"  A: {shorten(answer, 100)}")
    cost = trace_cost(trace)
    if cost is not None:
        lines.append(f"  cost ${cost:.5f}")
    return "\n".join(lines)


def format_summary(s: dict) -> str:
    if not s["requests"]:
        return "No traces."
    lat = s["latency"]
    lines = [
        f"Requests: {s['requests']}  ({s['first_start']} to {s['last_start']})",
        "Setup: " + "; ".join(f"{k} ({v})" for k, v in s["setups"].items()),
        "Models: " + ", ".join(f"{short_model(m)} ({c} requests)" for m, c in s["models"].items()),
        f"Errors: {s['errors']} ({(s['error_rate'] or 0):.1%})",
    ]
    for message, count in s["error_messages"].items():
        lines.append(f"  {count} x {shorten(message, 100)}")
    lines.append(f"Latency per request: avg {_ms(lat['avg_ms'])}, p50 {_ms(lat['p50_ms'])}, "
                 f"p95 {_ms(lat['p95_ms'])}, max {_ms(lat['max_ms'])}")
    lines.append(f"Model calls per request: {s['llm_calls_per_request']}; tokens per request: "
                 f"{s['input_tokens_per_request']} in, {s['output_tokens_per_request']} out")
    if s["cost_usd"] is not None:
        extra = f" ({s['unpriced_requests']} unpriced)" if s["unpriced_requests"] else ""
        lines.append(f"Cost: ${s['cost_usd']:.4f} in all, ${s['cost_per_1000_requests']:.2f} "
                     f"per 1,000 requests{extra}")
    rs = s["retrieval_top_score"]
    if rs["count"]:
        lines.append(f"Retrieval top score: median {rs['p50']}, 5th percentile {rs['p05']}, "
                     f"lowest {rs['min']}")
    ag = s["agent"]
    if ag["runs"]:
        lines.append(f"Agent: {ag['runs']} runs, avg {ag['avg_turns']} turns, stopped "
                     + ", ".join(f"{k} x{v}" for k, v in ag["stopped"].items())
                     + "; tools " + ", ".join(f"{k} x{v}" for k, v in ag["tool_calls"].items())
                     + f"; {ag['tool_errors']} tool error(s)")
    lines.append("")
    lines.append(f"{'Where the time goes':<22} {'calls':>6} {'avg':>8} {'p50':>8} {'p95':>8}"
                 "   (a step's time includes the steps inside it)")
    for name, st in s["span_latency"].items():
        lines.append(f"{name:<22} {st['count']:>6} {_ms(st['avg_ms']):>8} "
                     f"{_ms(st['p50_ms']):>8} {_ms(st['p95_ms']):>8}")
    if s["content_captured"] < s["requests"]:
        lines.append(f"\n{s['requests'] - s['content_captured']} trace(s) were recorded without "
                     "content (no questions or answers stored).")
    return "\n".join(lines)
