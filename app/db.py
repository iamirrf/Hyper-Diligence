import argparse
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.config import get_settings

logger = logging.getLogger(__name__)

_pool: ConnectionPool[Any] | None = None


def _configure_connection(conn: psycopg.Connection[Any]) -> None:
    register_vector(conn)


def get_pool() -> ConnectionPool[Any]:
    """Laziness keeps imports cheap for CLIs, tests, and cold app startup."""

    global _pool
    if _pool is None:
        settings = get_settings()
        _pool = ConnectionPool(
            conninfo=settings.database_url,
            min_size=1,
            max_size=10,
            kwargs={"row_factory": dict_row},
            configure=_configure_connection,
            open=True,
        )
    return _pool


@contextmanager
def get_connection() -> Iterator[psycopg.Connection[Any]]:
    with get_pool().connection() as conn:
        yield conn


def init_schema() -> None:
    """Schema creation is idempotent so deploys and local setup can share it."""

    settings = get_settings()
    embedding_dimensions = int(settings.embedding_dimensions)
    if embedding_dimensions <= 0:
        raise ValueError("EMBEDDING_DIMENSIONS must be positive")
    logger.info("initializing_schema", extra={"database_url": _redact_url(settings.database_url)})
    with psycopg.connect(settings.database_url) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        register_vector(conn)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS filings (
                id SERIAL PRIMARY KEY,
                ticker TEXT NOT NULL,
                form TEXT NOT NULL,
                filed DATE NOT NULL,
                accession TEXT UNIQUE NOT NULL,
                source_url TEXT NOT NULL,
                s3_key TEXT
            );
            """
        )
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id SERIAL PRIMARY KEY,
                filing_id INT REFERENCES filings(id) ON DELETE CASCADE,
                section TEXT,
                chunk_index INT NOT NULL,
                content TEXT NOT NULL,
                token_count INT NOT NULL,
                embedding vector({embedding_dimensions})
            );
            """
        )
        conn.commit()
    logger.info("schema_ready")


def reset_schema() -> None:
    """A reset path keeps vector dimension changes explicit and reversible."""

    settings = get_settings()
    logger.warning("resetting_schema", extra={"database_url": _redact_url(settings.database_url)})
    with psycopg.connect(settings.database_url) as conn:
        conn.execute("DROP TABLE IF EXISTS chunks;")
        conn.execute("DROP TABLE IF EXISTS filings;")
        conn.commit()
    init_schema()


def count_chunks() -> int:
    with get_connection() as conn:
        row = conn.execute("SELECT COUNT(*) AS count FROM chunks;").fetchone()
    return int(row["count"] if row else 0)


def _redact_url(url: str) -> str:
    if "@" not in url:
        return url
    prefix, suffix = url.rsplit("@", 1)
    scheme = prefix.split("://", 1)[0] if "://" in prefix else "postgresql"
    return f"{scheme}://***@{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true", help="Initialize database schema")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate application tables")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    if args.reset:
        reset_schema()
        return
    if args.init:
        init_schema()
        return
    parser.print_help()


if __name__ == "__main__":
    main()
