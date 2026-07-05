from typing import Literal

from app.retrieval.bm25 import bm25_search
from app.retrieval.dense import dense_search
from app.retrieval.fusion import rrf
from app.retrieval.rerank import rerank
from app.retrieval.types import SearchResult

SearchMode = Literal["dense", "bm25", "hybrid", "hybrid_rerank"]


def search(
    query: str,
    mode: SearchMode = "hybrid_rerank",
    k: int = 5,
    ticker: str | None = None,
    form: str | None = None,
) -> list[SearchResult]:
    if mode == "dense":
        return dense_search(query, k=k, ticker=ticker, form=form)
    if mode == "bm25":
        return bm25_search(query, k=k, ticker=ticker, form=form)
    if mode == "hybrid":
        return rrf(
            [
                dense_search(query, k=20, ticker=ticker, form=form),
                bm25_search(query, k=20, ticker=ticker, form=form),
            ]
        )[:k]
    if mode == "hybrid_rerank":
        candidates = rrf(
            [
                dense_search(query, k=20, ticker=ticker, form=form),
                bm25_search(query, k=20, ticker=ticker, form=form),
            ]
        )[:20]
        return rerank(query, candidates, top_n=k)
    raise ValueError(f"Unknown search mode: {mode}")
