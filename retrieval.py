"""Policy retrieval (RAG without a vector database).

1. The handbook is split into chunks at every `## heading` (one policy topic per chunk).
2. For a query, each chunk gets a TF-IDF-style keyword score (title words weigh more).
3. Only the top-k chunks above a minimum score are returned. If nothing scores high enough
   the function returns [] - the agent must then say the handbook has no answer.
Simple, deterministic, explainable in a viva; embeddings are unnecessary for a 11-section document."""
import math
import re

import config

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set("""a an the is are was were be been of to in on for and or but if then so as at by
from with without about into over after before can could may might must should would will
shall do does did done i me my we our you your it its this that these those there their they
what which who whom how when where why not no yes any all some more most than also per
policy policies rule rules handbook student students college procedure""".split())
MIN_SCORE = 3.0         # below this a chunk is treated as irrelevant
RELATIVE_CUTOFF = 0.35  # drop chunks scoring < 35% of the best chunk


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > 4 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def _norm(t: str) -> str:
    t = _stem(t)
    return t[:8] if len(t) > 8 else t   # plagiarise/plagiarism -> 'plagiari'


def tokenize(text: str) -> list[str]:
    return [_norm(t) for t in _TOKEN.findall((text or "").lower()) if t not in _STOP and len(t) > 1]


def load_chunks(path=None) -> list[dict]:
    """Split the markdown handbook into chunks at '## ' headings. Skips the disclaimer header."""
    text = (path or config.POLICY_MD).read_text(encoding="utf-8")
    chunks = []
    for part in re.split(r"(?m)^## ", text)[1:]:
        title, _, body = part.partition("\n")
        title, body = title.strip(), body.strip()
        chunk_id = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        chunks.append({"chunk_id": chunk_id, "title": title, "text": body})
    return chunks


class PolicyIndex:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.title_tokens = [set(tokenize(c["title"])) for c in chunks]
        self.body_tokens = [tokenize(c["text"]) for c in chunks]
        n = len(chunks)
        df: dict[str, int] = {}
        for i, c in enumerate(chunks):
            for t in set(self.body_tokens[i]) | self.title_tokens[i]:
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log(1 + n / d) for t, d in df.items()}

    def search(self, query: str, k: int | None = None) -> list[dict]:
        k = k or config.POLICY_TOP_K
        q = set(tokenize(query))
        scored = []
        for i, c in enumerate(self.chunks):
            body = set(self.body_tokens[i])
            score = 0.0
            for t in q:
                if t in self.title_tokens[i]:
                    score += 2 * self.idf.get(t, 0)       # title match counts double
                if t in body:
                    score += self.idf.get(t, 0)
            if score >= MIN_SCORE:
                scored.append({**c, "score": round(score, 2)})
        scored.sort(key=lambda c: -c["score"])
        if scored:
            scored = [c for c in scored if c["score"] >= RELATIVE_CUTOFF * scored[0]["score"]]
        return scored[:k]


_INDEX: PolicyIndex | None = None


def get_index() -> PolicyIndex:
    global _INDEX
    if _INDEX is None:
        _INDEX = PolicyIndex(load_chunks())
    return _INDEX


def search_policy(query: str, k: int | None = None) -> list[dict]:
    return get_index().search(query, k)
