"""Validate the golden dataset itself.

Bad test data produces misleading scores, so this suite catches typos in
the dataset before they ever reach a model.
"""
import json

import pytest

from app.paths import CORPUS_DIR, GOLDEN_DIR
from app.retriever import load_chunks
from evals.dataset import DatasetError, load_golden

GOLDEN_FILE = GOLDEN_DIR / "golden_v3.jsonl"
GOLDEN_V2 = GOLDEN_DIR / "golden_v2.jsonl"
GOLDEN_V1 = GOLDEN_DIR / "golden_v1.jsonl"


def test_golden_v3_loads_and_has_32_cases():
    assert len(load_golden(GOLDEN_FILE)) == 32


def test_v2_is_unchanged_and_still_loads():
    assert len(load_golden(GOLDEN_V2)) == 32


def test_v3_only_adds_evidence_to_v2():
    """v3 = v2 plus the evidence field. Nothing else may change."""
    v3 = {c["id"]: c for c in load_golden(GOLDEN_FILE)}
    for case in load_golden(GOLDEN_V2):
        new = dict(v3[case["id"]])
        new["evidence"] = []          # v2 cases load with evidence defaulted to []
        assert new == case


def test_every_evidence_string_is_in_its_source_documents():
    """Catches typos: evidence must be copied exactly from the corpus."""
    chunks = load_chunks(CORPUS_DIR)
    for case in load_golden(GOLDEN_FILE):
        for text in case["evidence"]:
            assert any(text in c.text and c.doc in case["source_docs"] for c in chunks), \
                f"{case['id']}: {text!r} not found in {case['source_docs']}"


def test_answerable_cases_have_evidence():
    no_single_passage = {"out_of_scope", "ambiguous"}
    for case in load_golden(GOLDEN_FILE):
        if case["category"] not in no_single_passage:
            assert case["evidence"], f"{case['id']} has no evidence strings"


def test_v1_is_unchanged_and_still_loads():
    # Old versions are frozen, so past results stay comparable.
    assert len(load_golden(GOLDEN_V1)) == 20


def test_v2_contains_every_v1_case_unchanged():
    v2 = {c["id"]: c for c in load_golden(GOLDEN_V2)}
    for case in load_golden(GOLDEN_V1):
        assert v2[case["id"]] == case


def test_every_source_doc_exists_in_the_corpus():
    corpus_files = {p.name for p in CORPUS_DIR.glob("*.md")}
    for case in load_golden(GOLDEN_FILE):
        for doc in case["source_docs"]:
            assert doc in corpus_files, f"{case['id']} points at missing file {doc}"


def test_dataset_covers_every_category():
    categories = {case["category"] for case in load_golden(GOLDEN_FILE)}
    assert categories == {"numeric", "factual", "multi_hop", "out_of_scope",
                          "calculation", "comparison", "false_premise", "ambiguous"}


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
