import csv
import json

import pytest

from evals.calibration import (
    agreement,
    cohen_kappa,
    export_for_labeling,
    parse_label,
    score_labels,
)


def test_kappa_perfect_agreement():
    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0


def test_kappa_punishes_an_always_pass_judge():
    # 90% raw agreement, but the judge just says "pass" every time.
    judge = [True] * 10
    human = [True] * 9 + [False]
    assert cohen_kappa(judge, human) == 0.0


def test_kappa_undefined_when_everything_is_identical():
    assert cohen_kappa([True, True], [True, True]) is None


def test_agreement_confusion_matrix():
    result = agreement([("L1", True, True), ("L2", True, False),
                        ("L3", False, True), ("L4", False, False)])
    assert (result.both_pass, result.both_fail) == (1, 1)
    assert result.judge_pass_human_fail == 1
    assert result.judge_fail_human_pass == 1
    assert result.disagreements == ["L2", "L3"]
    assert result.accuracy == 0.5


@pytest.mark.parametrize("text, expected", [
    ("y", True), ("Yes", True), ("PASS", True), ("n", False), ("0", False), ("", None), ("  ", None),
])
def test_parse_label(text, expected):
    assert parse_label(text) == expected


def test_parse_label_rejects_typos():
    with pytest.raises(ValueError):
        parse_label("maybe")


def make_row(case_id, correct, faithful, mutation=None):
    return {"id": case_id, "question": f"Q {case_id}", "contexts": ["c1", "c2"],
            "expected_answer": "REF", "answer": "ANS", "judge_correct": correct,
            "judge_faithful": faithful, "judge_reasoning": "r", "mutation": mutation}


def test_export_is_blind_and_round_trips_through_score(tmp_path):
    rows = [make_row("g001", True, True), make_row("g002", False, False, "number_swap"),
            {**make_row("g003", None, None)}]   # no judge verdict -> left out
    sheet = tmp_path / "labels.csv"
    key_path = export_for_labeling(rows, sheet)

    with sheet.open(newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        labeled = list(reader)

    # Nothing in the sheet reveals the judge's opinion or which rows are mutations.
    assert not any("judge" in col or "mutation" in col for col in header)
    assert len(labeled) == 2
    assert all(r["label_id"].startswith("L") for r in labeled)

    key = json.loads(key_path.read_text())
    for r in labeled:   # human agrees with the judge on every row
        r["human_correct"] = "y" if key[r["label_id"]]["judge_correct"] else "n"
        r["human_faithful"] = "y" if key[r["label_id"]]["judge_faithful"] else "n"
    with sheet.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(labeled)

    results = score_labels(sheet)
    assert results["correct"].accuracy == 1.0
    assert results["correct"].kappa == 1.0
    assert results["faithful"].n == 2
