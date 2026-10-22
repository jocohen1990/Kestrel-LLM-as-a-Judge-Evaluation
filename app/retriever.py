"""Keyword-based document retriever using BM25.

What a retriever does: given a question, find the few passages most likely to
contain the answer, so we can hand ONLY those passages to the LLM.

Why BM25 instead of embeddings for Phase 1? BM25 is a classic keyword-scoring
algorithm. It needs no API key, costs nothing, and is fully deterministic
(same input -> same output), which makes it easy to test. Swapping in an
embedding-based retriever later is a good "before vs after" experiment for
your eval suite.

BM25 is written out by hand below (no library) so you can read exactly how
the scoring works.
"""
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# Very common words that carry no meaning for search. Removing them stops
# "what" and "the" from dominating the scores.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for",
    "from", "has", "have", "how", "in", "is", "it", "its", "much", "of", "on",
    "or", "s", "that", "the", "to", "was", "were", "what", "when", "where",
    "which", "who", "with",
}


def tokenize(text: str) -> list[str]:
    """Lowercase the text and split it into searchable words.

    The regex keeps decimals together, so "$108.3" becomes "108.3".
    """
    words = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())
    return [w for w in words if w not in STOPWORDS]


def edit_distance(a: str, b: str, limit: int) -> int:
    """Damerau-Levenshtein distance (optimal string alignment): how many single-letter
    edits turn `a` into `b`. An edit is inserting, deleting or replacing a letter, or
    swapping two neighbours ("Feburary" -> "February" is ONE edit, not two).

    Returns limit + 1 as soon as the answer must be bigger than `limit`, so
    comparing a word against the whole vocabulary stays fast.
    """
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev2, prev = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1,          # delete a letter
                         cur[j - 1] + 1,       # insert a letter
                         prev[j - 1] + cost)   # replace (or keep) a letter
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)   # swap two neighbours
        if min(cur) > limit:
            return limit + 1
        prev2, prev = prev, cur
    return prev[-1]


def typo_limit(word: str) -> int:
    """How many edits a word can be away and still count as a misspelling of it.
    Short words get no tolerance: "beat" is one edit from "best" and "heat", and
    guessing wrong is worse than not guessing. 4-7 letters: 1 edit. 8+: 2 edits."""
    if len(word) < 4:
        return 0
    return 1 if len(word) <= 7 else 2


@dataclass
class Chunk:
    """One searchable passage, plus the file it came from."""
    doc: str
    text: str


def load_chunks(corpus_dir: Path) -> list[Chunk]:
    """Split every .md file in the folder into paragraph-sized chunks.

    Each chunk is prefixed with its document title. Without that, a paragraph
    like "gross margin was 71.4%" would lose the fact that it's about Q1 2026.
    This trick is often called a "contextual chunk header".
    """
    chunks = []
    for path in sorted(Path(corpus_dir).glob("*.md")):
        lines = path.read_text(encoding="utf-8").splitlines()
        title = lines[0].lstrip("# ").strip() if lines else path.stem
        body = "\n".join(lines[1:])
        for paragraph in body.split("\n\n"):
            paragraph = paragraph.strip()
            # Skip empty lines and the "> Fictional company..." disclaimer.
            if len(paragraph) < 40 or paragraph.startswith(">"):
                continue
            chunks.append(Chunk(doc=path.name, text=f"{title}\n{paragraph}"))
    return chunks


class Retriever:
    """BM25 scoring in plain Python.

    For each query word, a chunk earns points when:
      * the word appears in the chunk (more often = more points, with
        diminishing returns, controlled by k1),
      * the word is RARE across all chunks (the "idf" part: "lanesight"
        is far more informative than "kestrel", which is everywhere),
      * the chunk is short (long chunks are penalized a little, via b).
    """

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75,
                 typo_tolerance: bool = False):
        """typo_tolerance: correct query words that appear nowhere in the corpus to
        the closest word that does (retriever r2, added after Run 9: one misspelled
        word per question was harmless, but three lost the answer's paragraph
        5 times out of 31, because BM25 gives "Feburary" no credit for "February").
        Off by default, so r1 stays exactly as measured in Runs 1-12."""
        if not chunks:
            raise ValueError("No chunks to index. Is the corpus folder empty?")
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(c.text) for c in chunks]
        self.term_counts = [Counter(tokens) for tokens in self.doc_tokens]
        self.avg_len = sum(len(t) for t in self.doc_tokens) / len(self.doc_tokens)

        # How many chunks contain each word, then turn that into an idf weight.
        doc_freq = Counter(word for tokens in self.doc_tokens for word in set(tokens))
        n = len(chunks)
        self.idf = {w: math.log(1 + (n - df + 0.5) / (df + 0.5)) for w, df in doc_freq.items()}
        self.doc_freq = doc_freq
        self.typo_tolerance = typo_tolerance

    def correct(self, word: str) -> str:
        """The corpus word a misspelled query word most likely meant, or the word unchanged.

        Only words the corpus has never seen are corrected; a real word is never
        swapped for another. Numbers are never touched ("2025" must not become
        "2026"). Closest match wins; ties go to the word in more passages, the
        usual tie-break in spelling correction (a common word is a likelier target).
        """
        limit = typo_limit(word)
        if word in self.idf or not word.isalpha() or limit == 0:
            return word
        best, best_key = word, None
        for candidate in self.idf:
            if not candidate.isalpha():
                continue
            d = edit_distance(word, candidate, limit)
            if d <= limit:
                key = (d, -self.doc_freq[candidate], candidate)
                if best_key is None or key < best_key:
                    best, best_key = candidate, key
        return best

    def query_tokens(self, query: str) -> list[str]:
        tokens = tokenize(query)
        return [self.correct(t) for t in tokens] if self.typo_tolerance else tokens

    def corrections(self, query: str) -> dict[str, str]:
        """Which query words were corrected, and to what. For reports and debugging."""
        if not self.typo_tolerance:
            return {}
        return {t: c for t in tokenize(query) if (c := self.correct(t)) != t}

    def score(self, query_tokens: list[str], i: int) -> float:
        counts, length = self.term_counts[i], len(self.doc_tokens[i])
        total = 0.0
        for word in query_tokens:
            tf = counts.get(word, 0)
            if tf == 0:
                continue
            denominator = tf + self.k1 * (1 - self.b + self.b * length / self.avg_len)
            total += self.idf[word] * tf * (self.k1 + 1) / denominator
        return total

    def search(self, query: str, k: int = 4) -> list[Chunk]:
        """Return the k chunks with the highest BM25 score for the query."""
        query_tokens = self.query_tokens(query)
        scores = [self.score(query_tokens, i) for i in range(len(self.chunks))]
        ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        return [self.chunks[i] for i in ranked[:k]]
