"""Turn per-question results into a summary scorecard."""
import math
from collections import defaultdict


def p95(values: list[float]) -> float:
    """95th percentile: 95% of calls were at least this fast."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def summarize(rows: list[dict]) -> dict:
    total = len(rows)
    passed = sum(r["passed"] for r in rows)

    by_category = defaultdict(lambda: {"passed": 0, "total": 0})
    for r in rows:
        by_category[r["category"]]["total"] += 1
        by_category[r["category"]]["passed"] += r["passed"]
    for stats in by_category.values():
        stats["pass_rate"] = round(stats["passed"] / stats["total"], 3)

    hits = [r["retrieval_hit"] for r in rows if r["retrieval_hit"] is not None]
    latencies = [r["latency_s"] for r in rows]

    return {
        "total_cases": total,
        "passed": passed,
        "pass_rate": round(passed / total, 3) if total else 0.0,
        "by_category": dict(by_category),
        "retrieval_hit_rate": round(sum(hits) / len(hits), 3) if hits else None,
        "avg_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "p95_latency_s": round(p95(latencies), 3),
        "total_input_tokens": sum(r["input_tokens"] for r in rows),
        "total_output_tokens": sum(r["output_tokens"] for r in rows),
    }
