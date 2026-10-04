"""Phase 1 metrics: deterministic, rule-based checks.

These are fast, free and repeatable, but they're also blunt. For example,
"9%" passes if the number 9 appears anywhere in the answer, even by
coincidence. That weakness is deliberate material for Phase 2, where an
LLM-as-judge grades meaning instead of matching strings, and you then
measure how well that judge agrees with your own labels.
"""
import re
from dataclasses import dataclass

# Matches numbers like 1240, 1,240, 108.3 and 0.38. Signs are ignored on
# purpose, so a range written "112-115" reads as 112 and 115.
NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")

REFUSAL_PHRASES = (
    "i don't know",
    "i do not know",
    "not in the provided documents",
    "not mentioned",
    "does not contain",
    "do not contain",
    "no information",
    "not provided",
)


def extract_numbers(text: str) -> list[float]:
    """Pull every number out of a piece of text."""
    numbers = []
    for match in NUMBER_RE.findall(text):
        cleaned = match.replace(",", "")
        try:
            numbers.append(float(cleaned))
        except ValueError:
            pass
    return numbers


def number_present(expected: float, text: str, rel_tol: float = 0.005) -> bool:
    """True if `expected` appears in the text, allowing for unit scaling.

    The model might write "$412.6 million" or "$412,600,000". We compare the
    expected value against each number as written AND scaled down by a
    thousand, million and billion, so both phrasings match.
    """
    tolerance = max(1e-9, abs(expected) * rel_tol)
    for found in extract_numbers(text):
        for scale in (1, 1e3, 1e6, 1e9):
            if abs(found / scale - expected) <= tolerance:
                return True
    return False


def is_refusal(text: str) -> bool:
    """Did the model say it couldn't answer?"""
    normalized = text.lower().replace("’", "'")  # curly apostrophe -> straight
    return any(phrase in normalized for phrase in REFUSAL_PHRASES)


@dataclass
class CaseResult:
    passed: bool
    reason: str


def score_case(case: dict, answer: str) -> CaseResult:
    """Grade one model answer against one golden case."""
    refused = is_refusal(answer)

    # Out-of-scope questions should be refused. Answering them = hallucinating.
    if case["category"] == "out_of_scope":
        if refused:
            return CaseResult(True, "correctly refused")
        return CaseResult(False, "answered a question that isn't covered by the documents")

    # Answerable questions should NOT be refused.
    if refused:
        return CaseResult(False, "refused, but the answer is in the documents")

    lowered = answer.lower()
    missing_numbers = [n for n in case["expected_numbers"] if not number_present(n, answer)]
    missing_words = [w for w in case["must_include"] if w.lower() not in lowered]
    any_of = case.get("must_include_any", [])
    missing_any = bool(any_of) and not any(phrase.lower() in lowered for phrase in any_of)

    if missing_numbers or missing_words or missing_any:
        problems = []
        if missing_numbers:
            problems.append(f"missing numbers {missing_numbers}")
        if missing_words:
            problems.append(f"missing text {missing_words}")
        if missing_any:
            problems.append(f"contains none of {any_of}")
        return CaseResult(False, "; ".join(problems))
    return CaseResult(True, "all expected values present")


def refusal_is_exact(answer: str, expected_refusal: str) -> bool:
    """Strict instruction-following check: did the model refuse with EXACTLY
    the wording the system prompt demanded, and nothing else?

    The lenient `is_refusal` above answers "did it refuse?". This answers
    "did it follow the formatting instruction?". In run 1, g020 passed the
    first check and would have failed this one.
    """
    normalized = answer.strip().replace("’", "'")
    return normalized == expected_refusal


# Words that expose the pipeline's internals to an end user. A real user
# never sees "the context" or "passage [2]", so answers shouldn't mention them.
INTERNALS_PATTERN = re.compile(
    r"\b(the context|context passages?|passages?\s*\[?\d*\]?|provided context)\b",
    re.IGNORECASE,
)


def mentions_internals(answer: str) -> bool:
    """True if the answer refers to the RAG machinery (e.g. "According to the context")."""
    return bool(INTERNALS_PATTERN.search(answer))


def retrieval_hit(case: dict, retrieved_sources: list[str]) -> bool | None:
    """Did the retriever fetch every document this question needs?

    Returns None for out-of-scope questions, which have no source document,
    so they're left out of the hit-rate average instead of skewing it.
    """
    if not case["source_docs"]:
        return None
    return all(doc in retrieved_sources for doc in case["source_docs"])
