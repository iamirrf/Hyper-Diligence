import logging
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from app.db import get_connection
from app.retrieval.types import SearchResult

logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"\b\w+\b")
_INDEX: "BM25Index | None" = None


@dataclass(frozen=True)
class BM25Document:
    result: SearchResult
    tokens: list[str]


class BM25Index:
    """In-process BM25 is deliberate at ~5k chunks; rebuild after ingestion."""

    def __init__(self, documents: list[BM25Document]) -> None:
        self.documents = documents
        self.model = BM25Okapi([doc.tokens for doc in documents]) if documents else None

    def search(self, query: str, k: int, ticker: str | None, form: str | None) -> list[SearchResult]:
        if self.model is None:
            return []
        scores = self.model.get_scores(tokenize(query))
        ranked_indexes = sorted(range(len(scores)), key=lambda idx: float(scores[idx]), reverse=True)
        results: list[SearchResult] = []
        for index in ranked_indexes:
            doc = self.documents[index]
            if ticker and doc.result.ticker != ticker.upper():
                continue
            if form and doc.result.form != form.upper():
                continue
            results.append(
                SearchResult(
                    chunk_id=doc.result.chunk_id,
                    content=doc.result.content,
                    section=doc.result.section,
                    ticker=doc.result.ticker,
                    form=doc.result.form,
                    filed=doc.result.filed,
                    score=float(scores[index]),
                )
            )
            if len(results) >= k:
                break
        return results


def bm25_search(query: str, k: int = 20, ticker: str | None = None, form: str | None = None) -> list[SearchResult]:
    return get_index().search(query, k=k, ticker=ticker, form=form)


def rebuild_index() -> BM25Index:
    global _INDEX
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                chunks.id AS chunk_id,
                chunks.content,
                chunks.section,
                filings.ticker,
                filings.form,
                filings.filed,
                0.0 AS score
            FROM chunks
            JOIN filings ON filings.id = chunks.filing_id
            ORDER BY chunks.id;
            """
        ).fetchall()
    documents = [
        BM25Document(
            result=SearchResult(
                chunk_id=int(row["chunk_id"]),
                content=str(row["content"]),
                section=row["section"],
                ticker=str(row["ticker"]),
                form=str(row["form"]),
                filed=row["filed"],
                score=0.0,
            ),
            tokens=tokenize(str(row["content"])),
        )
        for row in rows
    ]
    _INDEX = BM25Index(documents)
    logger.info("bm25_index_built", extra={"documents": len(documents)})
    return _INDEX


def get_index() -> BM25Index:
    global _INDEX
    if _INDEX is None:
        return rebuild_index()
    return _INDEX


def tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())
