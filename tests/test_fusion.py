from datetime import date

from app.retrieval.fusion import rrf
from app.retrieval.types import SearchResult


def result(chunk_id: int) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        content=f"chunk {chunk_id}",
        section="Item 1.",
        ticker="AAPL",
        form="10-K",
        filed=date(2025, 1, 1),
        score=0.0,
    )


def test_rrf_combines_rankings_and_stably_breaks_ties() -> None:
    fused = rrf([[result(1), result(2), result(3)], [result(2), result(1), result(4)]], k=60)

    assert [item.chunk_id for item in fused] == [1, 2, 3, 4]
    assert fused[0].score == (1 / 61) + (1 / 62)
    assert fused[2].score == 1 / 63
