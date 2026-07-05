from dataclasses import replace
from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.retrieval.types import SearchResult


def rerank(query: str, candidates: list[SearchResult], top_n: int = 5) -> list[SearchResult]:
    if not candidates:
        return []
    scores = _model().predict([(query, candidate.content) for candidate in candidates])
    scored = [replace(candidate, score=float(score)) for candidate, score in zip(candidates, scores, strict=True)]
    return sorted(scored, key=lambda result: result.score, reverse=True)[:top_n]


@lru_cache
def _model() -> CrossEncoder:
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
