from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class SearchResult:
    chunk_id: int
    content: str
    section: str | None
    ticker: str
    form: str
    filed: date
    score: float
