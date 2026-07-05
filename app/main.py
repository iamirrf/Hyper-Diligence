import logging
from datetime import date
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from openai import OpenAIError
from pydantic import BaseModel, Field

from app.config import MissingConfigurationError
from app.db import count_chunks

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Hyper-Diligence", version="0.1.0")


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])
    chunks: int = Field(ge=0)


class SearchResultResponse(BaseModel):
    chunk_id: int
    content: str
    section: str | None
    ticker: str
    form: str
    filed: date
    score: float


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


class CitationResponse(BaseModel):
    chunk_id: int
    ticker: str | None
    form: str | None
    filed: str | None
    section: str | None


class ToolTraceResponse(BaseModel):
    tool: str
    args: dict[str, Any]


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]
    tool_trace: list[ToolTraceResponse]


class EvalResponse(BaseModel):
    table_markdown: str
    modes: dict[str, dict[str, float]]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", chunks=count_chunks())


@app.get("/search", response_model=list[SearchResultResponse])
def search_endpoint(
    q: str = Query(min_length=1),
    mode: Literal["dense", "bm25", "hybrid", "hybrid_rerank"] = "hybrid_rerank",
    k: int = Query(default=5, ge=1, le=20),
    ticker: str | None = None,
    form: str | None = None,
) -> list[SearchResultResponse]:
    from app.retrieval.service import search

    try:
        results = search(q, mode=mode, k=k, ticker=ticker, form=form)
    except MissingConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except OpenAIError as exc:
        raise HTTPException(status_code=503, detail=f"OpenAI request failed: {type(exc).__name__}") from exc
    return [SearchResultResponse(**result.__dict__) for result in results]


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    from app.agent.loop import run_agent

    try:
        result = run_agent(request.question)
    except MissingConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except OpenAIError as exc:
        raise HTTPException(status_code=503, detail=f"OpenAI request failed: {type(exc).__name__}") from exc
    logger.info("agent_tool_trace", extra={"tool_trace": result["tool_trace"]})
    return AskResponse(**result)


@app.get("/evals", response_model=EvalResponse)
def evals() -> EvalResponse:
    from app.evals.run_evals import load_last_results

    return EvalResponse(**load_last_results())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
