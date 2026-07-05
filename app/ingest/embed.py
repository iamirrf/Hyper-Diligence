import logging
import time

from openai import OpenAI

from app.config import require_openai_api_key

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
BATCH_SIZE = 100


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    client = OpenAI(api_key=require_openai_api_key())
    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        response = _embed_batch(client, batch)
        vectors.extend(response)
        logger.info("embedded_batch", extra={"batch_size": len(batch), "total_vectors": len(vectors)})
    return vectors


def _embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            result = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
            vectors = [item.embedding for item in result.data]
            for vector in vectors:
                if len(vector) != EMBEDDING_DIMENSIONS:
                    raise ValueError(f"Expected {EMBEDDING_DIMENSIONS} dimensions, got {len(vector)}")
            return vectors
        except Exception as exc:
            last_error = exc
            sleep_seconds = 2**attempt
            logger.warning("embedding_batch_failed", extra={"attempt": attempt, "sleep_seconds": sleep_seconds, "error": str(exc)})
            time.sleep(sleep_seconds)
    raise RuntimeError("Embedding request failed after retries") from last_error
