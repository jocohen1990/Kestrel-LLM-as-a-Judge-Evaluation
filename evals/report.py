"""Turn per-question results into a summary scorecard."""
import math
from collections import defaultdict


def p95(values: list[float]) -> float:
    """95th percentile: 95% of calls were at least this fast."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def rate(values: list) -> float | None:
    """Share of True values, ignoring None (= "not applicable" or "not measured")."""
    known = [v for v in values if v is not None]
    return round(sum(known) / len(known), 3) if known else None


def summarize(rows: list[dict]) -> dict:
    total = len(rows)
    passed = sum(r["passed"] for r in rows)

    by_category = defaultdict(lambda: {"passed": 0, "total": 0})
    for r in rows:
        by_category[r["category"]]["total"] += 1
        by_category[r["category"]]["passed"] += r["passed"]
    for stats in by_category.values():
        stats["pass_rate"] = round(stats["passed"] / stats["total"], 3)

    latencies = [r["latency_s"] for r in rows]

    return {
        "total_cases": total,
        "passed": passed,
        "pass_rate": round(passed / total, 3) if total else 0.0,
        "by_category": dict(by_category),
        "retrieval_hit_rate": rate([r["retrieval_hit"] for r in rows]),
        # Phase 2 metrics. .get() keeps older result rows (without these keys) working.
        "refusal_exact_rate": rate([r.get("refusal_exact") for r in rows]),
        "internals_mention_rate": rate([r.get("mentions_internals") for r in rows]),
        "judge_faithfulness_rate": rate([r.get("judge_faithful") for r in rows]),
        "judge_correct_rate": rate([r.get("judge_correct") for r in rows]),
        "rule_judge_agreement": rate([r.get("rule_judge_agree") for r in rows]),
        "judge_errors": sum(1 for r in rows if r.get("judge_error")),
        "avg_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "p95_latency_s": round(p95(latencies), 3),
        "total_input_tokens": sum(r["input_tokens"] for r in rows),
        "total_output_tokens": sum(r["output_tokens"] for r in rows),
    }
