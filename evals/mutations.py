"""Mutation testing for the judge: plant known errors, check the judge catches them.

In software QA, mutation testing deliberately breaks the code to check that
the test suite notices. Here we deliberately break the ANSWERS to check the
GRADERS notice. Because we planted the error, we know the right verdict
without any human labeling.

Four mutation types:
  * number_swap:  change the figure the question asks about
                  ("$108.3 million" -> "$127.7 million").
                  Should be judged incorrect AND unfaithful.
  * detail_swap:  change a supporting figure the question did NOT ask about
                  (the buyback amount in an answer about dividends).
                  Should be judged unfaithful. The main answer is still
                  right, so "correct" can stay true.
  * false_claim:  keep the right answer but append a made-up fact.
                  Should be judged unfaithful. The rule grader CAN'T catch
                  this one, since the expected number is still there.
  * fabrication:  replace a correct refusal with a confident made-up answer.
                  Should be judged incorrect AND unfaithful.

History: in the first mutation run, number_swap and detail_swap were one
type that always expected "incorrect", and the swap could hit period labels
("Q1" -> "Q2"). The judge flagged all 25 swaps as unfaithful, but 5 were
scored as misses because of those two mistakes in this file.
"""
import re
import zlib

from evals.metrics import NUMBER_RE

# Facts that appear nowhere in the corpus.
FALSE_CLAIMS = [
    "The company also announced a 3-for-1 stock split in March 2026.",
    "Kestrel also opened a second headquarters in Denver, Colorado.",
    "The board also approved a quarterly dividend of $0.12 per share.",
    "Kestrel also reported that it had no outstanding debt.",
    "The company also expects revenue to double by 2027.",
]

FABRICATED_ANSWER = "According to the filings, the figure was $123.4 million, up 11% from the prior year."

# What a correct judge should say about each mutation type.
EXPECTED_VERDICTS = {
    "number_swap": {"correct": False, "faithful": False},
    "detail_swap": {"faithful": False},          # main answer may still be correct
    "false_claim": {"faithful": False},          # main answer may still be correct
    "fabrication": {"correct": False, "faithful": False},
}

# "Q1", "Q2 2026", "FY2025": a digit glued to a period label is a name, not a figure.
PERIOD_LABEL = re.compile(r"(?:\bQ|\bFY)$", re.IGNORECASE)


def _wrong_value_text(original: str) -> str:
    """Turn "108.3" into a clearly different number with the same formatting."""
    value = float(original.replace(",", ""))
    wrong = value * 1.17 + 1
    decimals = len(original.split(".")[1]) if "." in original else 0
    return f"{wrong:,.{decimals}f}" if "," in original else f"{wrong:.{decimals}f}"


def _is_year(number_text: str) -> bool:
    return bool(re.fullmatch(r"(19|20)\d\d", number_text))


def _is_period_label(answer: str, match: re.Match) -> bool:
    """True for the 1 in "Q1" or the 2025 in "FY2025"."""
    return bool(PERIOD_LABEL.search(answer[:match.start()]))


def swap_number(answer: str, expected_numbers: list[float]) -> tuple[str, bool] | None:
    """Replace one number in the answer with a wrong one.

    Returns (mutated_answer, is_key), where is_key says whether the swapped
    number is one the golden case expects (the figure the question asks
    for). Years and period labels are never swapped. Returns None if the
    answer has no suitable number.
    """
    expected = set(expected_numbers)
    candidates = [m for m in NUMBER_RE.finditer(answer)
                  if not _is_year(m.group(0)) and not _is_period_label(answer, m)]

    def value(m):
        return float(m.group(0).replace(",", ""))

    key = next((m for m in candidates if value(m) in expected), None)
    target = key or (candidates[0] if candidates else None)
    if target is None:
        return None
    mutated = answer[:target.start()] + _wrong_value_text(target.group(0)) + answer[target.end():]
    return mutated, key is not None


def add_false_claim(answer: str, case_id: str) -> str:
    """Append a fabricated fact. The choice depends on the case id, so the
    same case always gets the same claim (reproducible runs)."""
    claim = FALSE_CLAIMS[zlib.crc32(case_id.encode()) % len(FALSE_CLAIMS)]
    return f"{answer.rstrip()} {claim}"


def make_mutations(row: dict) -> list[dict]:
    """Build every applicable mutation for one result row from run_evals.

    Each mutation keeps the row's question, contexts and reference answer,
    so it can be judged exactly like the original.
    """
    base = {k: row[k] for k in ("id", "category", "question", "expected_answer", "contexts")}
    mutations = []

    if row["category"] == "out_of_scope":
        mutations.append({**base, "mutation": "fabrication", "answer": FABRICATED_ANSWER})
        return mutations

    swapped = swap_number(row["answer"], row.get("expected_numbers", []))
    if swapped:
        text, is_key = swapped
        mutations.append({**base, "mutation": "number_swap" if is_key else "detail_swap",
                          "answer": text})

    mutations.append({**base, "mutation": "false_claim",
                      "answer": add_false_claim(row["answer"], row["id"])})
    return mutations


def caught(mutation_type: str, judge_correct: bool, judge_faithful: bool) -> bool:
    """Did the judge give the verdict a correct judge should give?"""
    expected = EXPECTED_VERDICTS[mutation_type]
    actual = {"correct": judge_correct, "faithful": judge_faithful}
    return all(actual[key] == want for key, want in expected.items())
