from evals.metrics import number_present
from evals.mutations import add_false_claim, caught, make_mutations, swap_number


def test_swap_number_changes_the_expected_figure():
    answer = "Q1 2026 revenue was $108.3 million."
    mutated = swap_number(answer, [108.3])
    assert mutated != answer
    assert not number_present(108.3, mutated)
    assert "2026" in mutated            # the year was left alone


def test_swap_number_keeps_thousands_formatting():
    mutated = swap_number("Kestrel had 1,240 employees.", [1240])
    assert "1,240" not in mutated
    assert "," in mutated


def test_swap_number_skips_years_and_returns_none_without_numbers():
    assert swap_number("Founded in 2017.", []) is None
    assert swap_number("Dana Whitfield is CEO.", []) is None


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
    # A planted extra claim only needs to be flagged as unfaithful.
    assert caught("false_claim", judge_correct=True, judge_faithful=False)
    assert not caught("false_claim", judge_correct=True, judge_faithful=True)
