"""Mutation testing for the judge: plant known errors, check the judge catches them.

In software QA, mutation testing deliberately breaks the code to check that
the test suite notices. Here we deliberately break the ANSWERS to check the
GRADERS notice. Because we planted the error, we know the right verdict
without any human labeling.

Three mutation types:
  * number_swap:    change a key figure ("$108.3 million" -> "$126.7 million").
                    Should be judged incorrect AND unfaithful.
  * false_claim:    keep the right answer but append a made-up fact.
                    Should be judged unfaithful. The rule grader CAN'T catch
                    this one, since the expected number is still there. This
                    is the gap from run 1 that the judge exists to close.
  * fabrication:    replace a correct refusal with a confident made-up answer.
                    Should be judged incorrect AND unfaithful.
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
    "false_claim": {"faithful": False},          # may still be "correct" overall
    "fabrication": {"correct": False, "faithful": False},
}


def _wrong_value_text(original: str) -> str:
    """Turn "108.3" into a clearly different number with the same formatting."""
    value = float(original.replace(",", ""))
    wrong = value * 1.17 + 1
    decimals = len(original.split(".")[1]) if "." in original else 0
    text = f"{wrong:,.{decimals}f}" if "," in original else f"{wrong:.{decimals}f}"
    return text


def _is_year(number_text: str) -> bool:
    return bool(re.fullmatch(r"(19|20)\d\d", number_text))


def swap_number(answer: str, expected_numbers: list[float]) -> str | None:
    """Replace one key number in the answer with a wrong one.

    Prefers a number the golden case expects (the "main" figure), then any
    other non-year number. Returns None if the answer has nothing to swap.
    """
    matches = list(NUMBER_RE.finditer(answer))
    expected = set(expected_numbers)

    def value(m):
        return float(m.group(0).replace(",", ""))

    target = next((m for m in matches if value(m) in expected), None)
    if target is None:
        target = next((m for m in matches if not _is_year(m.group(0))), None)
    if target is None:
        return None
    return answer[:target.start()] + _wrong_value_text(target.group(0)) + answer[target.end():]


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
    if swapped and swapped != row["answer"]:
        mutations.append({**base, "mutation": "number_swap", "answer": swapped})

    mutations.append({**base, "mutation": "false_claim",
                      "answer": add_false_claim(row["answer"], row["id"])})
    return mutations


def caught(mutation_type: str, judge_correct: bool, judge_faithful: bool) -> bool:
    """Did the judge give the verdict a correct judge should give?"""
    expected = EXPECTED_VERDICTS[mutation_type]
    actual = {"correct": judge_correct, "faithful": judge_faithful}
    return all(actual[key] == want for key, want in expected.items())
