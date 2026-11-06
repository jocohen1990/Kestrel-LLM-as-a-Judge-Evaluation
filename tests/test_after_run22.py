"""Offline tests for the fixes proposed after Run 22: retriever r4 (typo
correction trusts the query rewrite), prompt v7 (questions in other languages)
and the opt-in translated robustness style. None is the default yet: each needs
the eval runs listed in the README first."""
import json

import pytest

from app import rag
from app.llm import LLMResponse
from app.paths import CORPUS_DIR, GOLDEN_DIR, ROBUSTNESS_DIR
from app.retriever import Retriever, load_chunks
from evals.dataset import load_golden
from evals.robustness import (ALL_STYLES, HAND_STYLES, OPT_IN_STYLES, STYLES, build_variants,
                              load_hand_variants, localize_numbers, run_variant,
                              summarize_robustness)

GOLDEN = load_golden(GOLDEN_DIR / "golden_v3.jsonl")


# ---- Retriever r4 ------------------------------------------------------------------

@pytest.fixture(scope="module")
def retriever():
    return Retriever(load_chunks(CORPUS_DIR), typo_tolerance=True)


def test_kept_words_are_never_corrected(retriever):
    q = "What city is Kestrel's head office in?"
    assert retriever.corrections(q) == {"head": "had", "office": "officer"}   # r2/r3 (Run 22 u001)
    keep = frozenset({"head", "office"})
    assert retriever.corrections(q, keep) == {}
    assert "had" not in retriever.query_tokens(q, keep)


def test_a_misspelling_the_rewrite_fixed_is_still_corrected(retriever):
    q = "What was Kestrel's revebue in Q1 2026?\nKestrel revenue first quarter 2026"
    keep = frozenset({"kestrel", "revenue", "first", "quarter", "2026"})
    assert retriever.corrections(q, keep) == {"revebue": "revenue"}


def rewrite_then_answer(rewrite_text, calls):
    def fake(system, user, model=None, **kwargs):
        calls.append(user)
        text = rewrite_text if system == rag.REWRITE_SYSTEM else "Omaha."
        return LLMResponse(text=text, model="m", input_tokens=1, output_tokens=1, latency_s=0)
    return fake


def test_r4_keeps_the_rewrite_words_r3_corrects_them(monkeypatch):
    q = "What city is Kestrel's head office in?"
    for version, corrected in (("r3", True), ("r4", False)):
        calls = []
        monkeypatch.setattr(rag, "generate", rewrite_then_answer("Kestrel head office location", calls))
        assistant = rag.RagAssistant(retriever_version=version)
        chunks, _, _ = assistant.retrieve(q)
        led_by_leadership = chunks[0].text.split("\n")[1].startswith("Leadership")
        assert led_by_leadership is corrected, version


def test_r4_is_r3_plus_one_setting():
    r3, r4 = rag.RETRIEVER_VERSIONS["r3"], rag.RETRIEVER_VERSIONS["r4"]
    assert {k: v for k, v in r4.items() if k != "trust_rewrite"} == r3 and r4["trust_rewrite"]
    assert rag.DEFAULT_RETRIEVER_VERSION == "r3"   # not the default until measured


# ---- Prompt v7 -----------------------------------------------------------------------

def test_v7_is_v6_plus_one_line():
    v6, v7 = rag.PROMPTS["v6"]["system"], rag.PROMPTS["v7"]["system"]
    assert v7.replace(rag._V7_LANGUAGE_RULE, "") == v6
    assert v7.count(rag._V7_LANGUAGE_RULE) == 1
    assert v7.index(rag._V7_LANGUAGE_RULE) > v7.index(rag._V6_IDENTITY_RULE)
    assert rag.REFUSAL in v7 and rag.DEFAULT_PROMPT_VERSION == "v6"


def test_v7_example_language_is_german():
    """The example names German, not French (the language of the question that prompted v7)."""
    assert "(German, for example)" in rag._V7_LANGUAGE_RULE


# ---- The translated style --------------------------------------------------------------

def translations():
    return load_hand_variants(ROBUSTNESS_DIR / "translations_v1.jsonl", GOLDEN)


def test_translated_is_opt_in():
    assert "translated" not in STYLES and "translated" not in HAND_STYLES
    assert ALL_STYLES == STYLES + OPT_IN_STYLES
    default_run = build_variants(GOLDEN, translations())
    assert not [v for v in default_run if v["style"] == "translated"]


def test_translations_v1():
    rows = translations()
    assert len(rows) == 13
    langs = [json.loads(l)["language"] for l in
             (ROBUSTNESS_DIR / "translations_v1.jsonl").read_text(encoding="utf-8").splitlines()]
    assert set(langs) == {"fr", "es", "de", "pt"}
    variants = build_variants(GOLDEN, rows, styles=["original", "translated"])
    assert sum(v["style"] == "translated" for v in variants) == 13


def test_translations_dont_copy_an_eval_or_traffic_question():
    from tests.test_traces import eval_questions, normalize
    import traffic
    taken = eval_questions() | {normalize(r["question"]) for f in ("traffic_v1", "traffic_v2")
                                for r in traffic.load_traffic(f"data/traffic/{f}.jsonl")}
    assert [v["id"] for v in translations() if normalize(v["question"]) in taken] == []


def test_summary_includes_the_translated_style():
    rows = [{"id": "g001", "source_id": "g001", "style": "original", "passed": True,
             "passage_hit": True},
            {"id": "g001-translated", "source_id": "g001", "style": "translated", "passed": False,
             "passage_hit": True}]
    s = summarize_robustness(rows)
    assert s["by_style"]["translated"]["flips"] == ["g001-translated"]
    assert s["by_style"]["translated"]["flips_model"] == ["g001-translated"]


def test_v7_answers_and_refuses_in_the_users_language():
    rule = rag._V7_LANGUAGE_RULE
    assert "Answer it in that language" in rule and "made up, answer in English" in rule
    v6, v7 = rag.PROMPTS["v6"]["system"], rag.PROMPTS["v7"]["system"]
    for lang, sentence in rag.REFUSALS.items():
        assert v7.count(sentence) == 1, lang          # each sentence, exactly once
        if lang != "en":
            assert sentence not in v6, lang            # older prompts are unchanged
    assert "In any other language, use the English sentence." in rule


def test_refusal_sentences():
    assert set(rag.REFUSALS) == {"en", "fr", "de", "es", "pt"}
    assert rag.REFUSALS["en"] == rag.REFUSAL
    assert len(set(rag.REFUSALS.values())) == 5
    assert all(s.endswith(".") and "\n" not in s for s in rag.REFUSALS.values())


def test_every_check_knows_every_refusal_sentence():
    from evals.metrics import is_refusal, refusal_is_exact
    from evals.monitors import contains_refusal, is_exact_refusal, without_refusal
    for sentence in rag.REFUSALS.values():
        assert is_refusal(sentence) and refusal_is_exact(sentence, rag.REFUSAL)
        assert is_exact_refusal(sentence) and without_refusal(sentence) == ""
        assert contains_refusal(sentence + " Plus more.")
    # a refusal with commentary is still not exact, in any language
    assert not refusal_is_exact(rag.REFUSALS["fr"] + " C'est un conseil.", rag.REFUSAL)
    # curly apostrophes (common in model output) still match
    assert refusal_is_exact(rag.REFUSALS["fr"].replace("'", "’"), rag.REFUSAL)


def test_a_foreign_refusal_followed_by_more_is_flagged(monkeypatch):
    from tests.test_monitors import checks_on, rag_trace
    t = rag_trace(monkeypatch, rag.REFUSALS["es"] + " No doy consejos de inversión.")
    assert {"refusal", "refusal_with_more"} <= checks_on(t)
    assert checks_on(rag_trace(monkeypatch, rag.REFUSALS["de"])) - {"low_top_score"} == {"refusal"}


# ---- Reading numbers written the way other languages write them -----------------

@pytest.mark.parametrize("text,expected", [
    ("412,6 millions de dollars", "412.6 millions de dollars"),     # French decimal comma
    ("1 240 employés", "1240 employés"),                           # French thousands space
    ("2\u202f850 clients", "2850 clients"),                        # narrow no-break space
    ("1.240 Mitarbeiter", "1240 Mitarbeiter"),                     # German thousands point
    ("0,38 $", "0.38 $"),
    ("3,120 enterprise customers", "3,120 enterprise customers"),  # English thousands untouched
    ("$0.38 and $108.3 million", "$0.38 and $108.3 million"),      # English decimals untouched
    ("Q1 2026", "Q1 2026"),
])
def test_localize_numbers(text, expected):
    assert localize_numbers(text) == expected


def test_translated_answers_are_graded_with_their_own_number_format():
    case = next(c for c in GOLDEN if c["id"] == "g010")   # expects 0.38
    variant = {"id": "g010-translated", "source_id": "g010", "style": "translated",
               "question": "Quel a été le BPA dilué ?"}

    class Assistant:
        retriever_version = "r3"
        retriever_config = {"typo_tolerance": True}
        retriever = Retriever(load_chunks(CORPUS_DIR), typo_tolerance=True)

        def answer(self, q):
            from types import SimpleNamespace
            return SimpleNamespace(answer="Le BPA dilué était de 0,38 $ au deuxième trimestre 2026.",
                                   sources=[], contexts=[], search_query=q, rewrite=None,
                                   llm=SimpleNamespace(latency_s=0, input_tokens=0, output_tokens=0))

    assert run_variant(variant, case, Assistant())["passed"] is True
    # the same answer to an English variant isn't reinterpreted ("0,38" reads as 0 and 38)
    english = {**variant, "style": "reworded"}
    assert run_variant(english, case, Assistant())["passed"] is False
