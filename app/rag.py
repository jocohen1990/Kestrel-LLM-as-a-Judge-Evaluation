"""The system under test: a retrieval-augmented (RAG) Q&A assistant.

Flow:  question -> retriever finds passages -> passages + question go to the
LLM -> LLM answers using only those passages.
"""
from dataclasses import dataclass

from app.llm import LLMResponse, generate
from app.paths import CORPUS_DIR
from app.retriever import Retriever, load_chunks

REFUSAL = "I don't know based on the provided documents."

# Bump this whenever SYSTEM_PROMPT changes. It's saved with every eval run,
# so a score change can always be traced to the prompt that caused it.
PROMPT_VERSION = "v1"

SYSTEM_PROMPT = f"""You are a financial research assistant for Kestrel Freight Analytics.
Answer the user's question using ONLY the context passages provided.

Rules:
- If the answer is not in the context, reply exactly: "{REFUSAL}"
- Do not guess or use outside knowledge.
- Keep answers to one or two sentences.
- Include the specific figures from the context when relevant.
"""


@dataclass
class RagAnswer:
    question: str
    answer: str
    sources: list[str]    # which files the retrieved passages came from
    contexts: list[str]   # the passages themselves (Phase 2 grades against these)
    llm: LLMResponse


class RagAssistant:
    def __init__(self, corpus_dir=CORPUS_DIR, k: int = 4, model: str | None = None):
        self.retriever = Retriever(load_chunks(corpus_dir))
        self.k = k
        self.model = model

    @staticmethod
    def build_prompt(question: str, chunks) -> str:
        """Number each passage and label its source, then append the question."""
        context = "\n\n".join(
            f"[{i}] (source: {c.doc})\n{c.text}" for i, c in enumerate(chunks, start=1)
        )
        return f"Context passages:\n{context}\n\nQuestion: {question}"

    def answer(self, question: str) -> RagAnswer:
        chunks = self.retriever.search(question, k=self.k)
        response = generate(SYSTEM_PROMPT, self.build_prompt(question, chunks), model=self.model)
        return RagAnswer(
            question=question,
            answer=response.text.strip(),
            sources=sorted({c.doc for c in chunks}),
            contexts=[c.text for c in chunks],
            llm=response,
        )
