from evals.metrics import number_present
from evals.mutations import add_false_claim, caught, make_mutations, swap_number


def test_swap_number_changes_the_expected_figure():
    answer = "Q1 2026 revenue was $108.3 million."
    mutated, is_key = swap_number(answer, [108.3])
    assert is_key is True
    assert not number_present(108.3, mutated)
    assert "Q1 2026" in mutated          # period label and year left alone


def test_swap_number_keeps_thousands_formatting():
    mutated, _ = swap_number("Kestrel had 1,240 employees.", [1240])
    assert "1,240" not in mutated
    assert "," in mutated


def test_swap_number_skips_years_and_returns_none_without_numbers():
    assert swap_number("Founded in 2017.", []) is None
    assert swap_number("Dana Whitfield is CEO.", []) is None


def test_swap_number_never_touches_period_labels():
    # Run 2 bug: "Q1 2026 earnings release" became "Q2 2026 earnings release".
    answer = "According to the Q1 2026 earnings release, it predicts congestion 14 days ahead."
    mutated, is_key = swap_number(answer, [])
    assert "Q1 2026" in mutated
    assert "14 days" not in mutated
    assert is_key is False
    assert swap_number("Results for Q2 and FY2025.", []) is None


def test_swap_number_reads_year_followed_by_comma_as_a_year():
    # Bug found while fixing the above: "2026," was read as one number.
    answer = "Launched in February 2026, it covers 14 days."
    mutated, _ = swap_number(answer, [])
    assert "February 2026," in mutated


def test_swap_of_an_unasked_figure_is_a_detail_swap():
    # g027-style: the question was about dividends; the buyback amount is a detail.
    rows = make_mutations(row("false_premise", "No dividend. A buyback of up to $25 million.", []))
    assert [m["mutation"] for m in rows] == ["detail_swap", "false_claim"]


def test_false_claim_is_appended_and_reproducible():
    first = add_false_claim("Revenue was $108.3 million.", "g006")
    second = add_false_claim("Revenue was $108.3 million.", "g006")
    assert first == second
    assert first.startswith("Revenue was $108.3 million. ")


def row(category, answer, expected_numbers=()):
    return {"id": "g001", "category": category, "question": "Q?", "expected_answer": "REF",
            "contexts": ["ctx"], "answer": answer, "expected_numbers": list(expected_numbers)}


def test_make_mutations_for_answerable_question():
    mutations = make_mutations(row("numeric", "Revenue was $412.6 million.", [412.6]))
    assert [m["mutation"] for m in mutations] == ["number_swap", "false_claim"]
    assert all(m["contexts"] == ["ctx"] for m in mutations)


def test_make_mutations_for_out_of_scope_fabricates_an_answer():
    mutations = make_mutations(row("out_of_scope", "I don't know based on the provided documents."))
    assert [m["mutation"] for m in mutations] == ["fabrication"]


def test_caught_rules():
    assert caught("number_swap", judge_correct=False, judge_faithful=False)
    assert not caught("number_swap", judge_correct=True, judge_faithful=False)
    # A swapped detail only needs to be flagged as unfaithful.
    assert caught("detail_swap", judge_correct=True, judge_faithful=False)
    assert not caught("detail_swap", judge_correct=True, judge_faithful=True)
    # A planted extra claim only needs to be flagged as unfaithful.
    assert caught("false_claim", judge_correct=True, judge_faithful=False)
    assert not caught("false_claim", judge_correct=True, judge_faithful=True)


def test_summarize_counts_judge_errors_separately():
    from evals.mutations import summarize_mutations
    rows = [
        {"mutation": "false_claim", "judge_caught": True, "rule_caught": False},
        {"mutation": "false_claim", "judge_caught": False, "rule_caught": True},
        {"mutation": "false_claim", "judge_caught": None, "rule_caught": False,
         "judge_error": "no valid verdict after 2 attempts"},
        {"mutation": "fabrication", "judge_caught": True, "rule_caught": True},
    ]
    by_type = summarize_mutations(rows)
    assert by_type["false_claim"] == {"total": 2, "judge_caught": 1, "rule_caught": 1, "judge_errors": 1}
    assert by_type["fabrication"]["judge_errors"] == 0
