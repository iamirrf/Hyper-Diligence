import json
import logging
from typing import Any

from openai import OpenAI

from app.agent.tools import TOOL_DEFINITIONS, execute_tool
from app.config import require_openai_api_key

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are Hyper-Diligence, a due-diligence analyst. Answer ONLY from retrieved filing content. "
    "Cite every claim inline as [TICKER form filed §section]. If retrieval is insufficient, say so. "
    "Use the calculator for any arithmetic."
)


def run_agent(question: str, max_iters: int = 6) -> dict[str, Any]:
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
