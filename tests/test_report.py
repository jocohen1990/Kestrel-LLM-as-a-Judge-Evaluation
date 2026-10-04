from evals.report import p95, summarize


def row(category, passed, hit, latency=1.0):
    return {"category": category, "passed": passed, "retrieval_hit": hit,
            "latency_s": latency, "input_tokens": 100, "output_tokens": 10}


def test_summarize_counts_and_rates():
    rows = [row("numeric", True, True), row("numeric", False, True),
            row("out_of_scope", True, None)]
    summary = summarize(rows)
    assert summary["passed"] == 2
    assert summary["pass_rate"] == 0.667
    assert summary["by_category"]["numeric"] == {"passed": 1, "total": 2, "pass_rate": 0.5}
    assert summary["retrieval_hit_rate"] == 1.0    # out_of_scope (None) excluded
    assert summary["total_input_tokens"] == 300


def test_p95():
    assert p95([float(i) for i in range(1, 101)]) == 95.0
    assert p95([]) == 0.0
