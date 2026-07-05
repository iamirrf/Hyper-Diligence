import json
import logging
import re
from typing import Any

from openai import OpenAI

from app.agent.tools import TOOL_DEFINITIONS, execute_tool
from app.config import get_settings, require_openai_api_key

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are Hyper-Diligence, a due-diligence analyst. Answer ONLY from retrieved filing content. "
    "Cite every claim inline as [TICKER form filed §section]. If retrieval is insufficient, say so. "
    "Use the calculator for any arithmetic."
)


def run_agent(question: str, max_iters: int = 6) -> dict[str, Any]:
    settings = get_settings()
    if settings.chat_provider == "extractive":
        return run_extractive_agent(question)
    if settings.chat_provider != "openai":
        raise ValueError(f"Unknown chat provider: {settings.chat_provider}")

    client = OpenAI(api_key=require_openai_api_key())
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    tool_trace: list[dict[str, Any]] = []
    citations: dict[int, dict[str, Any]] = {}

    for _ in range(max_iters):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
            temperature=0.1,
        )
        message = response.choices[0].message
        if not message.tool_calls:
            return {"answer": message.content or "", "citations": list(citations.values()), "tool_trace": tool_trace}

        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": call.type,
                        "function": {"name": call.function.name, "arguments": call.function.arguments},
                    }
                    for call in message.tool_calls
                ],
            }
        )

        for call in message.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            tool_trace.append({"tool": call.function.name, "args": args})
            result = execute_tool(call.function.name, args)
            _capture_citations(result, citations)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, default=str),
                }
            )

    forced = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            *messages,
            {
                "role": "system",
                "content": "Tool-call budget reached. Produce the best grounded final answer now, or say retrieval is insufficient.",
            },
        ],
        tools=TOOL_DEFINITIONS,
        tool_choice="none",
        temperature=0.1,
    )
    answer = forced.choices[0].message.content or ""
    return {"answer": answer, "citations": list(citations.values()), "tool_trace": tool_trace}


def run_extractive_agent(question: str) -> dict[str, Any]:
    """A local fallback keeps the demo honest when no hosted LLM is available."""

    args: dict[str, Any] = {"query": question, "k": 5}
    ticker = _infer_ticker(question)
    if ticker:
        args["ticker"] = ticker
    results = execute_tool("search_filings", args)
    tool_trace = [{"tool": "search_filings", "args": args}]
    citations: dict[int, dict[str, Any]] = {}
    _capture_citations(results, citations)

    if not results:
        return {
            "answer": "Retrieval is insufficient to answer from the ingested filings.",
            "citations": [],
            "tool_trace": tool_trace,
        }

    query_terms = _content_terms(question)
    bullets: list[str] = []
    for result in results[:3]:
        content = str(result["content"])
        sentence = _best_sentence(content, query_terms)
        citation = _citation_label(result)
        bullets.append(f"- {sentence} {citation}")

    answer = "Based on the retrieved filing passages:\n" + "\n".join(bullets)
    return {"answer": answer, "citations": list(citations.values()), "tool_trace": tool_trace}


def _capture_citations(result: Any, citations: dict[int, dict[str, Any]]) -> None:
    if not isinstance(result, list):
        return
    for item in result:
        if not isinstance(item, dict) or "chunk_id" not in item:
            continue
        chunk_id = int(item["chunk_id"])
        citations.setdefault(
            chunk_id,
            {
                "chunk_id": chunk_id,
                "ticker": item.get("ticker"),
                "form": item.get("form"),
                "filed": item.get("filed"),
                "section": item.get("section"),
            },
        )


def _best_sentence(content: str, query_terms: set[str]) -> str:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", content)
        if _is_useful_sentence(sentence)
    ]
    if not sentences:
        return content[:500].strip()
    ranked = sorted(sentences, key=lambda sentence: _overlap_score(sentence, query_terms), reverse=True)
    selected = ranked[0]
    if len(selected) > 500:
        return selected[:497].rstrip() + "..."
    return selected


def _is_useful_sentence(sentence: str) -> bool:
    cleaned = sentence.strip()
    if len(cleaned) < 80:
        return False
    boilerplate = {"apple inc.", "table of contents"}
    return cleaned.lower() not in boilerplate


def _overlap_score(sentence: str, query_terms: set[str]) -> tuple[int, int]:
    terms = _content_terms(sentence)
    return (len(terms & query_terms), -len(sentence))


def _content_terms(text: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "did",
        "does",
        "for",
        "from",
        "how",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "what",
        "with",
    }
    return {term for term in re.findall(r"[a-z0-9]+", text.lower()) if len(term) > 2 and term not in stopwords}


def _citation_label(item: dict[str, Any]) -> str:
    section = item.get("section") or "unknown"
    return f"[{item.get('ticker')} {item.get('form')} {item.get('filed')} §{section}]"


def _infer_ticker(question: str) -> str | None:
    normalized = question.lower()
    aliases = {
        "AAPL": {"aapl", "apple"},
        "MSFT": {"msft", "microsoft"},
        "NVDA": {"nvda", "nvidia"},
        "JPM": {"jpm", "jpmorgan", "jpmorgan chase", "jp morgan"},
        "TSLA": {"tsla", "tesla"},
    }
    for ticker, names in aliases.items():
        if any(re.search(rf"\b{re.escape(name)}\b", normalized) for name in names):
            return ticker
    return None
