"""Offline tests for retriever versions r2 (typo tolerance) and r3 (query rewrite).

r1 is covered by test_retriever.py and must not change: every result before
Run 13 was measured with it.
"""
import json

import pytest

from app import rag
from app.llm import LLMResponse
from app.paths import ATTACKS_DIR, BIAS_DIR, CORPUS_DIR, GOLDEN_DIR, ROBUSTNESS_DIR
from app.retriever import Retriever, edit_distance, load_chunks, typo_limit
from evals.dataset import load_golden
from evals.injection import load_attacks, poison_chunk
from evals.metrics import passage_hit

GOLDEN = load_golden(GOLDEN_DIR / "golden_v3.jsonl")
CHUNKS = load_chunks(CORPUS_DIR)


# ---- Edit distance ------------------------------------------------------------

@pytest.mark.parametrize("a, b, expected", [
    ("revenue", "revenue", 0),
    ("revnue", "revenue", 1),       # missing letter
    ("revvenue", "revenue", 1),     # extra letter
    ("revenur", "revenue", 1),      # wrong letter
    ("feburary", "february", 1),    # two neighbours swapped: ONE edit
    ("heaadquartered", "headquartered", 1),
])
def test_edit_distance(a, b, expected):
    assert edit_distance(a, b, limit=2) == expected


def test_edit_distance_stops_early_past_the_limit():
    assert edit_distance("revenue", "dividend", limit=1) == 2   # limit + 1, not the true distance


def test_typo_limit_by_length():
    assert [typo_limit(w) for w in ("ceo", "beat", "revenue", "february")] == [0, 1, 1, 2]


# ---- r2: correcting query words ----------------------------------------------------

R2 = Retriever(CHUNKS, typo_tolerance=True)


@pytest.mark.parametrize("typo, fixed", [
    ("feburary", "february"),
    ("telematiccs", "telematics"),
    ("enteprise", "enterprise"),
    ("kestrels", "kestrel"),        # the lowercase style drops the apostrophe
    ("headquarters", "headquartered"),
])
def test_misspelled_words_are_corrected(typo, fixed):
    assert R2.correct(typo) == fixed


@pytest.mark.parametrize("word", ["revenue", "margin", "2025", "108.3", "ceo", "beat"])
def test_real_words_numbers_and_short_words_are_never_changed(word):
    # "2025" must never become "2026"; "beat" is too short to guess safely.
    assert R2.correct(word) == word


def test_r1_does_not_correct():
    r1 = Retriever(CHUNKS)
    assert r1.corrections("What prdouct launched in Feburary?") == {}
    assert R2.corrections("What prdouct launched in Feburary?") == {"prdouct": "product",
                                                                    "feburary": "february"}


def test_r2_finds_the_paragraphs_three_typos_lost_in_run_9():
    # Run 9: these five lost the answer's paragraph under r1.
    from evals.robustness import generate_variant
    lost = ["g003", "g008", "g011", "g012", "g024"]
    for case in [c for c in GOLDEN if c["id"] in lost]:
        question = generate_variant(case, "typos3")
        assert not passage_hit(case, [c.text for c in Retriever(CHUNKS).search(question)])
        assert passage_hit(case, [c.text for c in R2.search(question)]), case["id"]


def test_r2_keeps_every_golden_hit():
    for case in [c for c in GOLDEN if c["evidence"]]:
        r1_hit = passage_hit(case, [c.text for c in Retriever(CHUNKS).search(case["question"])])
        r2_hit = passage_hit(case, [c.text for c in R2.search(case["question"])])
        assert r2_hit or not r1_hit, case["id"]


def test_r2_still_retrieves_every_planted_injection():
    # An injection the retriever stops fetching would look "blocked" for the wrong reason.
    for case in load_attacks(ATTACKS_DIR / "injection_v1.jsonl"):
        if case["attack_type"] != "indirect":
            continue
        r = Retriever(CHUNKS + [poison_chunk(case)], typo_tolerance=True)
        contexts = [c.text for c in r.search(case["question"])]
        assert any(case["poison"]["text"] in c for c in contexts), case["id"]


# ---- r3: query rewrite -------------------------------------------------------------

def scripted_generate(calls, rewrite_text):
    def _fake(system, user, model=None, **kwargs):
        calls.append({"system": system, "user": user, **kwargs})
        is_rewrite = system == rag.REWRITE_SYSTEM
        return LLMResponse(text=rewrite_text if is_rewrite else "Omaha, Nebraska.",
                           model="fake", input_tokens=50 if is_rewrite else 300,
                           output_tokens=8 if is_rewrite else 5, latency_s=0.2)
    return _fake


def test_r3_rewrites_then_answers_the_original_question(monkeypatch):
    calls = []
    monkeypatch.setattr(rag, "generate",
                        scripted_generate(calls, "Kestrel headquartered city Omaha"))
    result = rag.RagAssistant(retriever_version="r3").answer("Kestrel HQ location?")

    assert [c["system"] == rag.REWRITE_SYSTEM for c in calls] == [True, False]
    assert calls[0]["user"] == "Kestrel HQ location?"
    # The answer step sees the user's words, not the rewrite.
    assert "Question: Kestrel HQ location?" in calls[1]["user"]
    assert "Kestrel headquartered city Omaha" not in calls[1]["user"]
    # The rewrite found the paragraph r1 missed in Run 9 (g003-casual).
    assert any("headquartered in Omaha" in c for c in result.contexts)
    assert result.search_query == "Kestrel HQ location?\nKestrel headquartered city Omaha"


def test_r3_cost_includes_the_rewrite(monkeypatch):
    monkeypatch.setattr(rag, "generate", scripted_generate([], "Kestrel headquarters"))
    result = rag.RagAssistant(retriever_version="r3").answer("Kestrel HQ?")
    assert (result.llm.input_tokens, result.llm.output_tokens) == (350, 13)
    assert result.llm.latency_s == pytest.approx(0.4)
    assert result.rewrite.input_tokens == 50


def test_r3_keeps_the_original_words_in_the_search(monkeypatch):
    # A useless rewrite can't make retrieval worse than r2: the question is still searched.
    monkeypatch.setattr(rag, "generate", scripted_generate([], "zzz"))
    result = rag.RagAssistant(retriever_version="r3").answer("What was Kestrel's revenue in Q1 2026?")
    assert any("108.3" in c for c in result.contexts)


@pytest.mark.parametrize("version", ["r1", "r2"])
def test_r1_and_r2_make_one_call(monkeypatch, version):
    calls = []
    monkeypatch.setattr(rag, "generate", scripted_generate(calls, "unused"))
    result = rag.RagAssistant(retriever_version=version).answer("Who is Kestrel's CEO?")
    assert len(calls) == 1 and result.rewrite is None


def test_unknown_retriever_version():
    with pytest.raises(ValueError, match="retriever"):
        rag.RagAssistant(retriever_version="r9")


def _all_eval_questions():
    files = [GOLDEN_DIR / "golden_v3.jsonl", ROBUSTNESS_DIR / "paraphrases_v1.jsonl",
             ROBUSTNESS_DIR / "paraphrases_v2.jsonl", ATTACKS_DIR / "injection_v1.jsonl",
             ATTACKS_DIR / "jailbreak_v1.jsonl", BIAS_DIR / "bias_v1.jsonl"]
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield json.loads(line)["question"].lower()


def test_rewrite_prompt_examples_come_from_no_eval_question():
    # Examples taken from the failing questions ("HQ", "revolver") would tune the
    # fix to the test. The prompt's examples must appear in no dataset.
    for example in ("yoy", "opex", "year over year", "operating expenses"):
        assert example in rag.REWRITE_SYSTEM.lower()
    questions = list(_all_eval_questions())
    for abbreviation in ("yoy", "opex"):
        assert not any(abbreviation in q for q in questions), abbreviation


def test_default_retriever_is_r3():
    """r3 chosen in Run 13 over r1 (Runs 1-12) and r2 (typo tolerance only)."""
    assert rag.DEFAULT_RETRIEVER_VERSION == "r3"
