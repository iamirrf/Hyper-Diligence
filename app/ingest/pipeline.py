import argparse
import logging

from psycopg import Connection

from app.config import get_settings
from app.db import get_connection
from app.ingest.chunk import TextChunk, chunk_filing
from app.ingest.edgar import TargetFiling, fetch_primary_document, list_target_filings, ticker_to_cik
from app.ingest.embed import embed_texts
from app.ingest.s3 import put_raw_filing

logger = logging.getLogger(__name__)


def run_pipeline(tickers: list[str]) -> dict[str, int]:
    settings = get_settings()
    totals = {"filings": 0, "chunks": 0, "skipped": 0}
    for ticker in tickers:
        cik = ticker_to_cik(ticker)
        for filing in list_target_filings(cik):
            with get_connection() as conn:
                if _accession_exists(conn, filing.accession):
                    totals["skipped"] += 1
                    logger.info("filing_skipped_existing", extra={"ticker": ticker, "accession": filing.accession})
                    continue

            html = fetch_primary_document(cik, filing.accession, filing.primary_document)
            s3_key = f"raw/{ticker.upper()}/{filing.form}/{filing.accession}.html"
            archived_key = put_raw_filing(settings.s3_bucket, s3_key, html)
            chunks = chunk_filing(filing.form, html)
            vectors = embed_texts([chunk.content for chunk in chunks])
            with get_connection() as conn:
                _insert_filing(conn, ticker.upper(), filing, archived_key, chunks, vectors)
            totals["filings"] += 1
            totals["chunks"] += len(chunks)
            logger.info(
                "filing_ingested",
                extra={
                    "ticker": ticker.upper(),
                    "form": filing.form,
                    "date": filing.filed.isoformat(),
                    "chunks": len(chunks),
                },
            )
    return totals


def _accession_exists(conn: Connection, accession: str) -> bool:
    row = conn.execute("SELECT 1 FROM filings WHERE accession = %s;", (accession,)).fetchone()
    return row is not None


def _insert_filing(
    conn: Connection,
    ticker: str,
    filing: TargetFiling,
    s3_key: str | None,
    chunks: list[TextChunk],
    vectors: list[list[float]],
) -> None:
    if len(chunks) != len(vectors):
        raise ValueError(f"Chunk/vector count mismatch: {len(chunks)} chunks, {len(vectors)} vectors")

    with conn.transaction():
        row = conn.execute(
            """
            INSERT INTO filings (ticker, form, filed, accession, source_url, s3_key)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id;
            """,
            (ticker, filing.form, filing.filed, filing.accession, filing.source_url, s3_key),
        ).fetchone()
        filing_id = row["id"] if isinstance(row, dict) else row[0]
        with conn.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO chunks (filing_id, section, chunk_index, content, token_count, embedding)
                VALUES (%s, %s, %s, %s, %s, %s);
                """,
                [
                    (filing_id, chunk.section, chunk.chunk_index, chunk.content, chunk.token_count, vector)
                    for chunk, vector in zip(chunks, vectors, strict=True)
                ],
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    totals = run_pipeline(args.tickers)
    print(f"Ingested {totals['filings']} filings, {totals['chunks']} chunks, skipped {totals['skipped']} existing filings.")


if __name__ == "__main__":
    main()
