# Hyper-Diligence — project conventions
Hyper-Diligence is a retrieval-augmented due-diligence analyst over SEC filings (10-Ks and 8-K earnings releases).
Stack (pinned — do not substitute): Python 3.11, FastAPI, PostgreSQL 16 + pgvector (docker image pgvector/pgvector:pg16), OpenAI API (embeddings: text-embedding-3-small, 1536 dims; chat/agent: gpt-4o-mini), rank_bm25 for BM25, sentence-transformers CrossEncoder "cross-encoder/ms-marco-MiniLM-L-6-v2" for reranking (CPU), boto3 for S3, pytest, docker compose, GitHub Actions.
Explicitly banned: LangChain, LlamaIndex, or any agent/RAG framework — the agent loop is written raw against OpenAI tool-calling.

Directory layout:
app/main.py (FastAPI: /health, /search, /ask, /evals) · app/config.py (pydantic-settings, all config from env) · app/db.py (psycopg3 pool, schema init CLI) · app/ingest/{edgar.py, s3.py, chunk.py, embed.py, pipeline.py} · app/retrieval/{dense.py, bm25.py, fusion.py, rerank.py, service.py} · app/agent/{tools.py, loop.py} · app/evals/{generate_candidates.py, run_evals.py, goldset.jsonl} · tests/ · Dockerfile · docker-compose.yml · .env.example · .github/workflows/deploy.yml · README.md

Conventions: full type hints; pydantic models for all API request/response schemas; docstrings explain WHY not what; small pure functions; logging at INFO with structured messages; every module runnable/verifiable on its own; secrets only via .env (never committed).
SEC EDGAR etiquette: every request sends header User-Agent: "Hyper-Diligence research amirhosseinaref@outlook.com"; max 10 req/s (sleep 0.2s between requests).
Every phase must end with a single command that proves it works.
