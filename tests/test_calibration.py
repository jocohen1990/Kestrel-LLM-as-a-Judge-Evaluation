import csv
import json

import pytest

from evals.calibration import (
    LABEL_COLUMNS,
    agreement,
    changed_verdicts,
    cohen_kappa,
    export_for_labeling,
    parse_label,
    read_label_rows,
    rejudge_label_rows,
    score_labels,
    split_passages,
)
from evals.judge import JudgeError, Verdict


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

    with sheet.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        labeled = list(reader)

    # Nothing in the sheet reveals the judge's opinion or which rows are mutations.
    assert not any("judge" in col or "mutation" in col for col in header)
    assert len(labeled) == 2
    assert all(r["label_id"].startswith("L") for r in labeled)

    key = json.loads(key_path.read_text(encoding="utf-8"))
    for r in labeled:   # human agrees with the judge on every row
        r["human_correct"] = "y" if key[r["label_id"]]["judge_correct"] else "n"
        r["human_faithful"] = "y" if key[r["label_id"]]["judge_faithful"] else "n"
    with sheet.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(labeled)

    results = score_labels(sheet)
    assert results["correct"].accuracy == 1.0
    assert results["correct"].kappa == 1.0
    assert results["faithful"].n == 2


# ---- Saving from Excel ---------------------------------------------------

def write_sheet(path, rows, encoding):
    """Write a labeling sheet the way Excel would save it."""
    lines = [",".join(LABEL_COLUMNS)]
    for r in rows:
        lines.append(",".join(r.get(col, "") for col in LABEL_COLUMNS))
    path.write_bytes(("\r\n".join(lines) + "\r\n").encode(encoding))


def test_export_starts_with_a_bom_for_excel(tmp_path):
    sheet = tmp_path / "labels.csv"
    export_for_labeling([make_row("g001", True, True)], sheet)
    assert sheet.read_bytes().startswith(b"\xef\xbb\xbf")


def test_reads_excel_csv_utf8_with_bom(tmp_path):
    sheet = tmp_path / "labels.csv"
    write_sheet(sheet, [{"label_id": "L001", "human_correct": "y"}], "utf-8-sig")
    assert read_label_rows(sheet)[0]["label_id"] == "L001"


def test_reads_excel_plain_csv_in_windows_1252(tmp_path):
    sheet = tmp_path / "labels.csv"
    write_sheet(sheet, [{"label_id": "L001", "notes": "caf\u00e9 \u2019quote\u2019"}], "cp1252")
    assert read_label_rows(sheet)[0]["notes"] == "caf\u00e9 \u2019quote\u2019"


def test_score_skips_empty_rows_excel_leaves_behind(tmp_path):
    sheet = tmp_path / "labels.csv"
    key_path = export_for_labeling([make_row("g001", True, False)], sheet)
    key = json.loads(key_path.read_text(encoding="utf-8"))
    label_id = next(iter(key))
    write_sheet(sheet, [{"label_id": label_id, "human_correct": "Y", "human_faithful": "n"},
                        {"label_id": ""}], "utf-8-sig")
    results = score_labels(sheet)
    assert results["correct"].n == 1 and results["correct"].accuracy == 1.0
    assert results["faithful"].accuracy == 1.0


def test_score_names_the_row_with_a_bad_label(tmp_path):
    sheet = tmp_path / "labels.csv"
    key_path = export_for_labeling([make_row("g001", True, True)], sheet)
    label_id = next(iter(json.loads(key_path.read_text(encoding="utf-8"))))
    write_sheet(sheet, [{"label_id": label_id, "human_correct": "maybe"}], "utf-8")
    with pytest.raises(ValueError, match=f"{label_id} human_correct"):
        score_labels(sheet)


# ---- Spacing between versions of the same question -------------------------

import random  # noqa: E402

from evals.calibration import MIN_GAP, min_distance, spread_out  # noqa: E402


def sheet_like_rows():
    """Same shape as a real export: 25 questions x 3 versions, 7 questions x 2."""
    rows = []
    for q in range(32):
        for v in range(3 if q < 25 else 2):
            rows.append({"id": f"g{q:03d}", "version": v})
    return rows


def test_versions_of_a_question_are_never_close_together():
    rows = sheet_like_rows()
    for seed in range(20):
        order = spread_out(rows, random.Random(seed))
        assert min_distance(order) >= MIN_GAP, f"seed {seed}"


def test_spread_out_keeps_every_row_exactly_once():
    rows = sheet_like_rows()
    order = spread_out(rows, random.Random(3))
    assert sorted((r["id"], r["version"]) for r in order) == sorted((r["id"], r["version"]) for r in rows)


def test_spread_out_is_reproducible_with_the_same_seed():
    rows = sheet_like_rows()
    assert spread_out(rows, random.Random(7)) == spread_out(rows, random.Random(7))


def test_spread_out_does_its_best_when_the_gap_is_impossible():
    # 2 questions x 3 versions can't be 10 rows apart; it should still return all rows.
    rows = [{"id": q, "version": v} for q in ("a", "b") for v in range(3)]
    order = spread_out(rows, random.Random(1))
    assert len(order) == 6
    assert min_distance(order) >= 1


def test_min_distance():
    rows = [{"id": "a"}, {"id": "b"}, {"id": "a"}, {"id": "c"}]
    assert min_distance(rows) == 2
    assert min_distance([{"id": "a"}, {"id": "b"}]) is None


def test_exported_sheet_keeps_the_gap(tmp_path):
    rows = [make_row(f"g{q:03d}", True, True, mutation=m)
            for q in range(30) for m in (None, "number_swap", "false_claim")]
    sheet = tmp_path / "labels.csv"
    key = json.loads(export_for_labeling(rows, sheet).read_text(encoding="utf-8"))
    ordered_ids = [{"id": key[r["label_id"]]["case_id"]} for r in read_label_rows(sheet)]
    assert min_distance(ordered_ids) >= MIN_GAP


# ---- Re-judging labeled rows with a new judge --------------------------

def test_split_passages_undoes_the_export_join_even_after_excel():
    assert split_passages("one\n---\ntwo") == ["one", "two"]
    assert split_passages("one\r\n---\r\ntwo") == ["one", "two"]   # Windows line endings
    assert split_passages("") == []


def labeled_sheet(tmp_path):
    """An exported sheet where the human agrees with the ORIGINAL judge on every row."""
    sheet = tmp_path / "labels.csv"
    rows = [make_row("g001", True, True), make_row("g002", False, False, "number_swap")]
    key = json.loads(export_for_labeling(rows, sheet).read_text(encoding="utf-8"))
    with sheet.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header, labeled = reader.fieldnames, list(reader)
    for r in labeled:
        r["human_correct"] = "y" if key[r["label_id"]]["judge_correct"] else "n"
        r["human_faithful"] = "y" if key[r["label_id"]]["judge_faithful"] else "n"
    with sheet.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(labeled)
    return sheet, key


def test_rejudge_writes_a_new_key_and_never_touches_the_labels(tmp_path):
    sheet, old_key = labeled_sheet(tmp_path)
    labels_before = sheet.read_bytes()
    seen = []

    def always_pass(question, contexts, reference, answer, **kwargs):
        seen.append((contexts, kwargs))
        return Verdict(faithful=True, correct=True, reasoning="new judge")

    new_path = rejudge_label_rows(sheet, always_pass, prompt_version="judge-v2")
    new_key = json.loads(new_path.read_text(encoding="utf-8"))

    assert sheet.read_bytes() == labels_before            # labels unchanged
    assert set(new_key) == set(old_key)                   # same rows
    assert seen[0] == (["c1", "c2"], {"prompt_version": "judge-v2"})   # passages split back out
    assert all(e["judge_correct"] is True for e in new_key.values())
    assert {e["case_id"] for e in new_key.values()} == {"g001", "g002"}

    # The new judge passes everything, so it now disagrees with the human on g002.
    assert score_labels(sheet)["correct"].accuracy == 1.0
    assert score_labels(sheet, new_path)["correct"].accuracy == 0.5
    changes = changed_verdicts(old_key, new_key)
    assert {(c["case_id"], c["criterion"]) for c in changes} == {("g002", "correct"),
                                                                 ("g002", "faithful")}


def test_rejudge_records_judge_failures_and_score_skips_them(tmp_path):
    sheet, _ = labeled_sheet(tmp_path)

    def broken(*args, **kwargs):
        raise JudgeError("garbled")

    new_key = json.loads(rejudge_label_rows(sheet, broken).read_text(encoding="utf-8"))
    assert all(e["judge_error"] == "garbled" and e["judge_correct"] is None
               for e in new_key.values())
    results = score_labels(sheet, sheet.with_suffix(".rejudged.key.json"))
    assert results["correct"].n == 0    # failed rows are left out, not counted as pass or fail


def test_rejudge_skips_rows_nobody_labeled(tmp_path):
    sheet, key = labeled_sheet(tmp_path)
    rows = read_label_rows(sheet)
    rows[1]["human_correct"] = rows[1]["human_faithful"] = ""   # leave the second row unlabeled
    with sheet.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    calls = []

    def judge(*args, **kwargs):
        calls.append(1)
        return Verdict(faithful=True, correct=True, reasoning="r")

    new_key = json.loads(rejudge_label_rows(sheet, judge).read_text(encoding="utf-8"))
    assert len(calls) == 1 and list(new_key) == [rows[0]["label_id"]]
