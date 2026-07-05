from typing import Any

from app.db import get_connection
from app.ingest.embed import embed_texts
from app.retrieval.types import SearchResult


def dense_search(query: str, k: int = 20, ticker: str | None = None, form: str | None = None) -> list[SearchResult]:
    where, params = _filters(ticker, form)
    if _matching_chunk_count(where, params) == 0:
        return []
    vector = embed_texts([query])[0]
    sql = f"""
        SELECT
            chunks.id AS chunk_id,
            chunks.content,
            chunks.section,
            filings.ticker,
            filings.form,
            filings.filed,
            chunks.embedding <=> %s AS score
        FROM chunks
        JOIN filings ON filings.id = chunks.filing_id
        WHERE chunks.embedding IS NOT NULL{where}
        ORDER BY score ASC
        LIMIT %s;
    """
    with get_connection() as conn:
        rows = conn.execute(sql, [vector, *params, k]).fetchall()
    return [_row_to_result(row) for row in rows]


def _matching_chunk_count(where: str, params: list[Any]) -> int:
    sql = f"""
        SELECT COUNT(*) AS count
        FROM chunks
        JOIN filings ON filings.id = chunks.filing_id
        WHERE chunks.embedding IS NOT NULL{where};
    """
    with get_connection() as conn:
        row = conn.execute(sql, params).fetchone()
    return int(row["count"] if row else 0)


def _filters(ticker: str | None, form: str | None) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if ticker:
        clauses.append("filings.ticker = %s")
        params.append(ticker.upper())
    if form:
        clauses.append("filings.form = %s")
        params.append(form.upper())
    if not clauses:
        return "", params
    return " AND " + " AND ".join(clauses), params


def _row_to_result(row: dict[str, Any]) -> SearchResult:
    return SearchResult(
        chunk_id=int(row["chunk_id"]),
        content=str(row["content"]),
        section=row["section"],
        ticker=str(row["ticker"]),
        form=str(row["form"]),
        filed=row["filed"],
        score=float(row["score"]),
    )
