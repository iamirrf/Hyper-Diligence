from dataclasses import replace

from app.retrieval.types import SearchResult


def rrf(result_lists: list[list[SearchResult]], k: int = 60) -> list[SearchResult]:
    scores: dict[int, float] = {}
    first_seen: dict[int, SearchResult] = {}
    order: dict[int, int] = {}
    next_order = 0

    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            if result.chunk_id not in first_seen:
                first_seen[result.chunk_id] = result
                order[result.chunk_id] = next_order
                next_order += 1
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1.0 / (k + rank)

    ranked_ids = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], order[chunk_id]))
    return [replace(first_seen[chunk_id], score=scores[chunk_id]) for chunk_id in ranked_ids]
