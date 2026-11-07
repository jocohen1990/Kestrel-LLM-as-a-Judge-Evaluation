"""Offline tests for Phase 6 step 3's other half: flagged production requests
become dataset cases (evals/candidates.py, golden_candidates.py, golden_v4,
agent_v2).

A case promoted from production must say exactly which request it came from,
repeat that request word for word, and be graded like every other case.
"""
import json

import pytest

import golden_candidates as candidates_cli
import traffic
from app.paths import AGENT_DIR, CORPUS_DIR, GOLDEN_DIR, TRAFFIC_DIR
from app.rag import REFUSAL
from app.retriever import load_chunks
from evals import candidates as cd
from evals.agent_eval import load_agent_cases
from evals.dataset import DatasetError, load_golden
from evals.monitors import load_reviews
from evals.runner import evaluate_case
from tests.test_compare import as_agent
from tests.test_monitors import rag_trace
from tests.test_runner import FakeAssistant

V3 = load_golden(GOLDEN_DIR / "golden_v3.jsonl")
V4 = load_golden(GOLDEN_DIR / "golden_v4.jsonl")
AGENT_V1 = load_agent_cases(AGENT_DIR / "agent_v1.jsonl")
AGENT_V2 = load_agent_cases(AGENT_DIR / "agent_v2.jsonl")
TRAFFIC = cd.load_traffic_files(TRAFFIC_DIR)
REVIEWS = [r for p in sorted(TRAFFIC_DIR.glob("reviews_v*.jsonl")) for r in load_reviews(p)]
PROMOTED = [c for c in V4 + AGENT_V2 if c.get("origin")]


# ---- The datasets --------------------------------------------------------------------

def test_golden_v4_is_v3_plus_eight_production_cases():
    assert len(V4) == 40
    v4 = {c["id"]: c for c in V4}
    for case in V3:
        assert v4[case["id"]] == case
    new = [c for c in V4 if c["id"] not in {x["id"] for x in V3}]
    assert [c["id"] for c in new] == [f"g{n:03d}" for n in range(33, 41)]
    assert all(c.get("origin") for c in new)


def test_agent_v2_is_v1_plus_seven_production_cases():
    assert len(AGENT_V2) == 34
    v2 = {c["id"]: c for c in AGENT_V2}
    for case in AGENT_V1:
        assert v2[case["id"]] == case
    new = [c for c in AGENT_V2 if c["id"] not in {x["id"] for x in AGENT_V1}]
    assert [c["id"] for c in new] == [f"ag{n:03d}" for n in range(28, 35)]
    assert all(c.get("origin") for c in new)


@pytest.mark.parametrize("case", PROMOTED, ids=lambda c: c["id"])
def test_a_promoted_case_repeats_its_request_word_for_word(case):
    o = case["origin"]
    assert o["traffic_id"] in TRAFFIC, f"{case['id']}: no traffic request {o['traffic_id']}"
    row = TRAFFIC[o["traffic_id"]]
    assert row["file"] == o["traffic_file"]
    assert case["question"] == row["question"]


@pytest.mark.parametrize("case", PROMOTED, ids=lambda c: c["id"])
def test_a_promoted_case_came_from_a_reviewed_problem(case):
    """Promotion follows a person reading the reply, not a flag alone."""
    tid = case["origin"]["traffic_id"]
    assert any(r["case_id"] == tid and r["verdict"] in ("wrong", "rule_break") for r in REVIEWS)


def test_golden_v4_evidence_is_copied_from_the_corpus():
    chunks = load_chunks(CORPUS_DIR)
    for case in V4:
        for text in case["evidence"]:
            assert any(text in c.text and c.doc in case["source_docs"] for c in chunks), \
                f"{case['id']}: {text!r} not found in {case['source_docs']}"
        if case["category"] not in ("out_of_scope", "ambiguous"):
            assert case["evidence"], f"{case['id']} has no evidence strings"


def test_agent_v2_search_evidence_is_in_a_filing():
    corpus = "\n".join(p.read_text(encoding="utf-8") for p in sorted(CORPUS_DIR.glob("*.md")))
    for case in AGENT_V2:
        for step in case["required_steps"]:
            if step.get("tool") == "search_documents":
                assert step["evidence"] in corpus, case["id"]


def test_the_loader_checks_origin_and_language(tmp_path):
    base = dict(V4[-1])
    bad = tmp_path / "g.jsonl"
    bad.write_text(json.dumps({**base, "origin": {"traffic_id": "u022"}}) + "\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="origin"):
        load_golden(bad)
    bad.write_text(json.dumps({**base, "language": ""}) + "\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="language"):
        load_golden(bad)


def test_older_versions_load_as_english():
    assert {c["language"] for c in V3} == {"en"}


# ---- Grading a case in another language ------------------------------------------------

FRENCH = next(c for c in V4 if c["id"] == "g038")


def test_a_french_answer_with_french_thousands_passes():
    row = evaluate_case(FRENCH, FakeAssistant("Kestrel avait 2 850 clients entreprises fin 2025."),
                        use_judge=False)
    assert row["passed"], row["reason"]


def test_the_same_answer_fails_an_english_case():
    """Without `language` the grader reads "2 850" as 2 and 850, as before."""
    english = {**FRENCH, "language": "en"}
    row = evaluate_case(english, FakeAssistant("Kestrel avait 2 850 clients entreprises fin 2025."),
                        use_judge=False)
    assert not row["passed"]


def test_a_french_refusal_still_fails_an_answerable_question():
    row = evaluate_case(FRENCH, FakeAssistant(REFUSAL), use_judge=False)
    assert not row["passed"] and "refused" in row["reason"]


# ---- Candidates ---------------------------------------------------------------------------

TRAFFIC_V2 = {"u001": {"file": "traffic_v2.jsonl", "id": "u001", "kind": "in_scope",
                       "question": "What city is Kestrel's head office in?"},
              "u002": {"file": "traffic_v2.jsonl", "id": "u002", "kind": "in_scope",
                       "question": "Who runs Kestrel day to day?"},
              "u003": {"file": "traffic_v2.jsonl", "id": "u003", "kind": "in_scope",
                       "question": "Q2 profit?"}}


def files(monkeypatch):
    rag = [rag_trace(monkeypatch, REFUSAL, case_id="u001"),
           rag_trace(monkeypatch, "- Dana Whitfield", case_id="u002"),
           rag_trace(monkeypatch, "Net income was $11.4 million.", case_id="u003")]
    agent = [as_agent(rag_trace(monkeypatch, "Omaha, Nebraska.", case_id="u001")),
             as_agent(rag_trace(monkeypatch, "Dana Whitfield, the CEO.", case_id="u002")),
             as_agent(rag_trace(monkeypatch, "**$11.4 million**", case_id="u003"))]
    return {"rag.jsonl": rag, "agent.jsonl": agent}


def test_each_reason_gives_its_priority(monkeypatch):
    reviews = [{"app": "rag", "case_id": "u001", "verdict": "wrong", "traffic": "traffic_v2",
                "note": "Omaha is in the annual report"},
               {"app": "agent", "case_id": "u003", "verdict": "rule_break", "traffic": "traffic_v2"}]
    result = cd.build_candidates(files(monkeypatch), reviews, TRAFFIC_V2)
    by_id = {c["traffic_id"]: c for c in result["candidates"]}
    assert result["comparisons"] == [["agent.jsonl", "rag.jsonl"]]
    assert by_id["u001"]["priority"] == 1                  # reviewed wrong
    assert {r["source"] for r in by_id["u001"]["reasons"]} == {"comparison", "review"}
    assert by_id["u003"]["priority"] == 2                  # rule break (bold)
    # RAG's list in u002 was flagged, and the RAG review of traffic_v2 didn't list it
    assert by_id["u002"]["priority"] == 4 and by_id["u002"]["status"] == "false_alarm"
    assert by_id["u001"]["question"] == "What city is Kestrel's head office in?"
    assert set(by_id["u001"]["answers"]) == {"rag (rag.jsonl)", "agent (agent.jsonl)"}


def test_an_unreviewed_flag_is_open_not_a_false_alarm(monkeypatch):
    result = cd.build_candidates(files(monkeypatch), [], TRAFFIC_V2)
    by_id = {c["traffic_id"]: c for c in result["candidates"]}
    assert by_id["u002"]["priority"] == 3 and by_id["u002"]["status"] == "open"
    assert by_id["u001"]["priority"] == 2                  # the disagreement alone


def test_a_case_with_a_matching_origin_shows_as_promoted(monkeypatch, tmp_path):
    ds = tmp_path / "golden_v9.jsonl"
    ds.write_text(json.dumps({"id": "g900", "origin": {"traffic_id": "u001"}}) + "\n",
                  encoding="utf-8")
    result = cd.build_candidates(files(monkeypatch), [], TRAFFIC_V2, cd.promoted_cases(ds))
    u001 = next(c for c in result["candidates"] if c["traffic_id"] == "u001")
    assert u001["status"] == "promoted" and u001["promoted_to"] == ["g900"]
    assert result["counts"]["promoted"] == 1
    assert "promoted to g900" in cd.format_candidates(result)


def test_files_of_different_traffic_are_not_compared(monkeypatch):
    f = files(monkeypatch)
    other = {"u001": {**TRAFFIC_V2["u001"], "file": "traffic_v9.jsonl"},
             "u002": TRAFFIC_V2["u002"], "u003": TRAFFIC_V2["u003"]}
    for t in f["agent.jsonl"]:
        t["attributes"]["traffic.file"] = "traffic_v9.jsonl"
    for t in f["rag.jsonl"]:
        t["attributes"]["traffic.file"] = "traffic_v2.jsonl"
    assert cd.build_candidates(f, [], other)["comparisons"] == []


def test_cli_end_to_end(monkeypatch, tmp_path, capsys):
    for name, traces in files(monkeypatch).items():
        (tmp_path / name).write_text("".join(json.dumps(t) + "\n" for t in traces), encoding="utf-8")
    monkeypatch.setattr(cd, "load_traffic_files", lambda folder: TRAFFIC_V2)
    monkeypatch.setattr(candidates_cli, "load_traffic_files", lambda folder: TRAFFIC_V2)
    code = candidates_cli.main([str(tmp_path / "rag.jsonl"), str(tmp_path / "agent.jsonl"),
                                "--reviews", "--datasets", "--results-dir", str(tmp_path)])
    assert code == 0
    out = capsys.readouterr().out
    assert "Priority 2" in out and "u001" in out
    saved = json.loads((tmp_path / "golden_candidates_latest.json").read_text(encoding="utf-8"))
    assert saved["counts"]["candidates"] == 3


def test_the_real_reviews_promote_into_the_real_datasets():
    """Every reply the reviews (reviews_v1 to v3) call wrong is now a test case, in
    golden_v4 (RAG) or agent_v2 (agent)."""
    promoted = cd.promoted_cases(GOLDEN_DIR / "golden_v4.jsonl", AGENT_DIR / "agent_v2.jsonl")
    wrong = {r["case_id"] for r in REVIEWS if r["verdict"] == "wrong"}
    assert wrong - set(promoted) == set()


def test_replay_records_the_traffic_file(monkeypatch, tmp_path):
    from app import rag
    from evals.traces import load_traces
    from tests.test_tracing import fake_generate
    monkeypatch.setattr(rag, "generate", fake_generate([]))
    monkeypatch.setattr(traffic, "warmup", lambda model=None: None)
    out = tmp_path / "traces_test.jsonl"
    assert traffic.main(["--limit", "1", "--out", str(out)]) == 0
    assert load_traces(out)[0]["attributes"]["traffic.file"] == "traffic_v1.jsonl"
