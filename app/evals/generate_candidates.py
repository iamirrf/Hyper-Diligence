import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from openai import OpenAI

from app.config import require_openai_api_key
from app.db import get_connection

logger = logging.getLogger(__name__)

CANDIDATES_PATH = Path("data/candidates.jsonl")
TARGET_CANDIDATES = 60


def generate_candidates() -> None:
    rows = _load_candidate_chunks()
    sampled = _sample_by_filing(rows)
    sampled = sampled[:TARGET_CANDIDATES]
    CANDIDATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=require_openai_api_key())

    with CANDIDATES_PATH.open("w", encoding="utf-8") as handle:
        for row in sampled:
            question = _make_question(client, str(row["content"]))
            payload = {
                "question": question,
                "chunk_id": row["chunk_id"],
                "ticker": row["ticker"],
                "form": row["form"],
                "section": row["section"],
                "content_preview": str(row["content"])[:200],
            }
            handle.write(json.dumps(payload) + "\n")
            logger.info("candidate_generated", extra={"chunk_id": row["chunk_id"]})

    print(f"Wrote {len(sampled)} draft candidates to {CANDIDATES_PATH}.")
    print("Review it manually. Keep only real, specific, passage-answerable questions and copy ~30 lines into app/evals/goldset.jsonl.")


def _load_candidate_chunks() -> list[dict[str, Any]]:
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT
                chunks.id AS chunk_id,
                chunks.filing_id,
                chunks.section,
                chunks.content,
                filings.ticker,
                filings.form
            FROM chunks
            JOIN filings ON filings.id = chunks.filing_id
            WHERE chunks.token_count > 200
            ORDER BY chunks.filing_id, chunks.section NULLS LAST, chunks.chunk_index;
            """
        ).fetchall()


def _sample_by_filing(rows: list[dict[str, Any]], per_filing: int = 5) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["filing_id"])].append(row)

    sampled: list[dict[str, Any]] = []
    for filing_rows in grouped.values():
        if len(filing_rows) <= per_filing:
            sampled.extend(filing_rows)
            continue
        positions = [round(index * (len(filing_rows) - 1) / (per_filing - 1)) for index in range(per_filing)]
        sampled.extend(filing_rows[position] for position in positions)
    return sampled


def _make_question(client: OpenAI, passage: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    "Write one specific question a financial analyst would ask that is answerable ONLY from this passage. "
                    'Return JSON {"question": "..."}.\n\n'
                    f"Passage:\n{passage[:4000]}"
                ),
            }
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    content = response.choices[0].message.content or "{}"
    payload = json.loads(content)
    return str(payload["question"]).strip()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    generate_candidates()
