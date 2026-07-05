from dataclasses import replace
from functools import lru_cache
import logging
from threading import Lock

from sentence_transformers import CrossEncoder

from app.retrieval.types import SearchResult

logger = logging.getLogger(__name__)
_MODEL_LOCK = Lock()


def rerank(query: str, candidates: list[SearchResult], top_n: int = 5) -> list[SearchResult]:
    if not candidates:
        return []
    with _MODEL_LOCK:
        try:
            scores = _model().predict(
                [(query, candidate.content) for candidate in candidates],
                show_progress_bar=False,
            )
        except Exception as exc:
            logger.warning("rerank_failed_returning_hybrid", extra={"error": str(exc), "error_type": type(exc).__name__})
            return candidates[:top_n]
    scored = [replace(candidate, score=float(score)) for candidate, score in zip(candidates, scores, strict=True)]
    return sorted(scored, key=lambda result: result.score, reverse=True)[:top_n]


@lru_cache
def _model() -> CrossEncoder:
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
