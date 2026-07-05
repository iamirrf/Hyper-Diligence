import ast
import operator
from typing import Any

from app.db import get_connection
from app.retrieval.service import search

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_filings",
            "description": "Search SEC filing chunks with hybrid retrieval and reranking.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "ticker": {"type": "string"},
                    "form": {"type": "string"},
                    "k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 6},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_filings",
            "description": "List ingested SEC filings for a ticker.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a safe arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    },
]


def search_filings(query: str, ticker: str | None = None, form: str | None = None, k: int = 6) -> list[dict[str, Any]]:
    results = search(query, mode="hybrid_rerank", k=k, ticker=ticker, form=form)
    return [
        {
            "chunk_id": result.chunk_id,
            "ticker": result.ticker,
            "form": result.form,
            "filed": result.filed.isoformat(),
            "section": result.section,
            "content": result.content,
        }
        for result in results
    ]


def list_filings(ticker: str) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, ticker, form, filed, accession, source_url, s3_key
            FROM filings
            WHERE ticker = %s
            ORDER BY filed DESC, form;
            """,
            (ticker.upper(),),
        ).fetchall()
    return [
        {
            "id": row["id"],
            "ticker": row["ticker"],
            "form": row["form"],
            "filed": row["filed"].isoformat(),
            "accession": row["accession"],
            "source_url": row["source_url"],
            "s3_key": row["s3_key"],
        }
        for row in rows
    ]


def calculator(expression: str) -> float:
    tree = ast.parse(expression, mode="eval")
    return float(_eval_node(tree.body))


def execute_tool(name: str, arguments: dict[str, Any]) -> Any:
    if name == "search_filings":
        return search_filings(**arguments)
    if name == "list_filings":
        return list_filings(**arguments)
    if name == "calculator":
        return {"result": calculator(**arguments)}
    raise ValueError(f"Unknown tool: {name}")


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("Only numeric arithmetic with + - * / ** and parentheses is allowed")
