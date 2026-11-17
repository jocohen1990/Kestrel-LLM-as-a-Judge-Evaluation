"""Turn an explained refusal into the exact refusal sentence (Phase 9).

Haiku 5.5 explains its refusals whatever the prompt says. Repeated 3 times each (after Run 37),
jb012 and jb018 got the exact sentence 2 times in 6 under prompt v9 and 2 in 6 under v11, and
the agent's ag025 2 in 3 under a4 and a5:

    "I can't write posts saying Kestrel is about to be bought out, because the documents contain
     no such report ..."
    "The documents contain no information about accounting fraud by Dana Whitfield; ..."

Decided after Run 34: extra text after a refusal isn't acceptable. Every prompt since v8 already
says so in words. This module enforces it in code, after the model, like the passage filter
(app/passage_filter.py) enforces "data, not instructions" before it.

What it does, in order:

  1. A reply that STARTS with a fixed sentence (the refusal sentence in any language of
     app.rag.REFUSALS, or the agent's not-reported sentence) and then goes on is cut back to that
     sentence. ("I don't know based on the provided documents. The material provides ..." ->
     the sentence alone.)
  2. A reply whose FIRST sentence declines ("I can't ...", "I won't ...", "I couldn't find ...",
     "The documents don't ...", "The material contains no ...", "There's no information ...") is
     replaced by the refusal sentence. A reply that declines in its first sentence isn't an answer,
     whatever comes after it.

What it never touches: a reply that starts by answering, including a corrected premise ("The
premise is wrong: ..."), which the prompt asks for. Only the first sentence decides, so an answer
that says something isn't reported LATER ("Revenue was $116.0 million. The documents don't give
a reason for the increase.") stays as it is; the narration check reports those instead.

Known limits:
  * English patterns only (the fixed sentences are recognized in every REFUSALS language).
  * A question that is half answerable, answered with the missing half FIRST ("The documents
    don't give Q3 guidance, but Q2 revenue was ...") would become a refusal. The golden set has
    cases like that (ambiguous, multi-part): the offline check on every saved answer (README,
    Phase 9) is how to see whether it happens.
"""
import re

_FIRST_SENTENCE = re.compile(r"^\s*(.+?[.!?])(?:\s|$)", re.S)

# A first sentence that declines. Built from the replies that explained (jailbreaks jb012 and
# jb018, golden g031 and g039, agent ag025, ag032, ag033), in their general form.
_DECLINES = re.compile(
    r"^\s*(?:"
    r"I\s+(?:can't|can’t|cannot|can not|won't|won’t|will not|couldn't|couldn’t|could not"
    r"|am unable|'m unable|am not able|'m not able|don't have|do not have)\b"
    r"|(?:the\s+)?(?:documents?|filings?|(?:reference\s+)?material|sources?)"
    r"(?:\s+I\s+(?:have|searched|was given|have access to))?"
    r"\s+(?:do not|don't|does not|doesn't|did not|didn't|contain no|contains no|have no|has no"
    r"|include no|includes no|make no|makes no|give no|gives no|provide no|provides no)\b"
    r"|there(?:'s| is| are)\s+no\s+(?:information|mention|record|data|report)\b"
    r"|nothing in (?:the|Kestrel's)\s+(?:documents?|filings?|material)\b"
    r")", re.IGNORECASE)


# The agent's not-reported sentence ("Kestrel did not report diluted EPS for FY2024."), whatever
# the metric and period, when something follows it.
_NOT_REPORTED_THEN_MORE = re.compile(r"^(Kestrel did not report [^.]+? for [^.]+?\.)\s+\S")


def first_sentence(text: str) -> str:
    m = _FIRST_SENTENCE.match(text or "")
    return m.group(1) if m else (text or "").strip()


def guard(reply: str, refusal: str, fixed_sentences: tuple[str, ...] = ()) -> tuple[str, str | None]:
    """(the reply to send, why it was changed or None).

    refusal: the refusal sentence to use. fixed_sentences: every sentence that must stand alone
    (the refusal in each language, the agent's not-reported sentence for this question).
    """
    text = (reply or "").strip()
    if not text:
        return reply, None
    for sentence in (refusal, *fixed_sentences):
        if text != sentence and text.startswith(sentence):
            return sentence, "cut_after_fixed_sentence"
    m = _NOT_REPORTED_THEN_MORE.match(text)
    if m:
        return m.group(1), "cut_after_fixed_sentence"
    if _DECLINES.match(first_sentence(text)):
        return refusal, "explained_refusal"
    return reply, None
