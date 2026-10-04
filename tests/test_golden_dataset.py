"""Validate the golden dataset itself.

Bad test data produces misleading scores, so this suite catches typos in
the dataset before they ever reach a model.
"""
import json

import pytest

from app.paths import CORPUS_DIR, GOLDEN_DIR
from evals.dataset import DatasetError, load_golden

GOLDEN_FILE = GOLDEN_DIR / "golden_v1.jsonl"


def test_golden_dataset_loads_and_has_20_cases():
    assert len(load_golden(GOLDEN_FILE)) == 20


def test_every_source_doc_exists_in_the_corpus():
    corpus_files = {p.name for p in CORPUS_DIR.glob("*.md")}
    for case in load_golden(GOLDEN_FILE):
        for doc in case["source_docs"]:
            assert doc in corpus_files, f"{case['id']} points at missing file {doc}"


def test_dataset_covers_every_category():
    categories = {case["category"] for case in load_golden(GOLDEN_FILE)}
    assert categories == {"numeric", "factual", "multi_hop", "out_of_scope"}


def test_loader_rejects_duplicate_ids(tmp_path):
    # tmp_path is a pytest fixture: a fresh temporary folder for this test.
    case = {"id": "x1", "category": "factual", "question": "q", "expected_answer": "a",
            "expected_numbers": [], "must_include": ["a"], "source_docs": []}
    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text(json.dumps(case) + "\n" + json.dumps(case) + "\n")
    with pytest.raises(DatasetError, match="duplicate id"):
        load_golden(bad_file)


def test_loader_rejects_unknown_category(tmp_path):
    case = {"id": "x1", "category": "trivia", "question": "q", "expected_answer": "a",
            "expected_numbers": [], "must_include": ["a"], "source_docs": []}
    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text(json.dumps(case) + "\n")
    with pytest.raises(DatasetError, match="unknown category"):
        load_golden(bad_file)
