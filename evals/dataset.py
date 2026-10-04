"""Load and validate the golden dataset.

The golden dataset is test data, so it gets the same care as test data in
any QA suite: a fixed schema, validation on load, and a version number in
the file name (golden_v1.jsonl). Changing the questions creates golden_v2,
so results from different dataset versions are never compared by accident.

Format: JSON Lines (.jsonl), one JSON object per line. It's easy to append
to, easy to diff in git, and easy to stream.
"""
import json
from pathlib import Path

REQUIRED_FIELDS = {
    "id", "category", "question", "expected_answer",
    "expected_numbers", "must_include", "source_docs",
}
VALID_CATEGORIES = {
    "numeric", "factual", "multi_hop", "out_of_scope",            # v1
    "calculation", "comparison", "false_premise", "ambiguous",    # added in v2
}


class DatasetError(ValueError):
    pass


def load_golden(path: Path) -> list[dict]:
    """Read the .jsonl file and fail loudly if any case is malformed."""
    cases = []
    seen_ids = set()
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        case = json.loads(line)
        # Optional field added in v2: the answer must contain AT LEAST ONE of
        # these phrases. Defaulting it to [] keeps v1 files loadable.
        case.setdefault("must_include_any", [])

        missing = REQUIRED_FIELDS - case.keys()
        if missing:
            raise DatasetError(f"Line {line_no}: missing fields {sorted(missing)}")
        if case["category"] not in VALID_CATEGORIES:
            raise DatasetError(f"Line {line_no}: unknown category {case['category']!r}")
        if case["id"] in seen_ids:
            raise DatasetError(f"Line {line_no}: duplicate id {case['id']!r}")

        # An answerable question needs at least one thing to check against.
        checks = case["expected_numbers"] or case["must_include"] or case["must_include_any"]
        if case["category"] != "out_of_scope" and not checks:
            raise DatasetError(f"Line {line_no}: {case['id']} has nothing to grade against")

        seen_ids.add(case["id"])
        cases.append(case)
    return cases
