"""Retrieval tests. BM25 runs locally, so these are offline and free.

If the retriever doesn't fetch the right passage, the LLM can't answer
correctly no matter how good it is. Testing retrieval separately tells you
WHICH half of the pipeline is responsible when an answer is wrong.
"""
from app.paths import CORPUS_DIR, GOLDEN_DIR
from app.retriever import Retriever, load_chunks, tokenize
from evals.dataset import load_golden
from evals.metrics import passage_hit, retrieval_hit

K = 4
MIN_HIT_RATE = 0.9
# Regression guard: the passage-level rate measured when the metric was
# added (26/27 at k=4; g022 misses). Retrieval may improve, never get worse.
MIN_PASSAGE_HITS = 26


def test_tokenize_lowercases_drops_stopwords_and_keeps_decimals():
    assert tokenize("What was the revenue of $108.3 million?") == ["revenue", "108.3", "million"]


def test_chunks_carry_their_document_title():
    chunks = load_chunks(CORPUS_DIR)
    margin_chunks = [c for c in chunks if "gross margin" in c.text.lower()]
    assert margin_chunks, "expected at least one gross margin passage"
    for chunk in margin_chunks:
        assert "Q1 2026" in chunk.text or "Q2 2026" in chunk.text


def test_retrieval_hit_rate_on_golden_set():
    retriever = Retriever(load_chunks(CORPUS_DIR))
    hits, misses = 0, []
    answerable = [c for c in load_golden(GOLDEN_DIR / "golden_v3.jsonl") if c["source_docs"]]

    for case in answerable:
        sources = {chunk.doc for chunk in retriever.search(case["question"], k=K)}
        if retrieval_hit(case, sorted(sources)):
            hits += 1
        else:
            misses.append(f"{case['id']}: {case['question']}")

    hit_rate = hits / len(answerable)
    assert hit_rate >= MIN_HIT_RATE, f"hit rate {hit_rate:.0%}; misses: {misses}"


def test_passage_hit_baseline_does_not_regress():
    retriever = Retriever(load_chunks(CORPUS_DIR))
    cases = [c for c in load_golden(GOLDEN_DIR / "golden_v3.jsonl") if c["evidence"]]
    hits = sum(
        passage_hit(c, [chunk.text for chunk in retriever.search(c["question"], k=K)])
        for c in cases
    )
    assert hits >= MIN_PASSAGE_HITS, f"passage hits fell to {hits}/{len(cases)}"
