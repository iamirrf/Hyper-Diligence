import logging
import time
from functools import lru_cache
from threading import Lock

from openai import OpenAI
from sentence_transformers import SentenceTransformer

from app.config import get_settings, require_openai_api_key

logger = logging.getLogger(__name__)
_LOCAL_MODEL_LOCK = Lock()

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
BATCH_SIZE = 100


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    settings = get_settings()
    if settings.embedding_provider == "local":
        return _embed_texts_local(texts)
    if settings.embedding_provider != "openai":
        raise ValueError(f"Unknown embedding provider: {settings.embedding_provider}")

    client = OpenAI(api_key=require_openai_api_key())
    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        response = _embed_batch(client, batch)
        vectors.extend(response)
        logger.info("embedded_batch", extra={"batch_size": len(batch), "total_vectors": len(vectors)})
    return vectors


def _embed_texts_local(texts: list[str]) -> list[list[float]]:
    settings = get_settings()
    with _LOCAL_MODEL_LOCK:
        model = _local_model(settings.embedding_model)
        vectors = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    result = [vector.tolist() for vector in vectors]
    for vector in result:
        if len(vector) != settings.embedding_dimensions:
            raise ValueError(f"Expected {settings.embedding_dimensions} dimensions, got {len(vector)}")
    logger.info("embedded_local_batch", extra={"texts": len(texts), "model": settings.embedding_model})
    return result


@lru_cache
def _local_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


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
