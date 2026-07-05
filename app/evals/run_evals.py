import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GOLDSET_PATH = Path("app/evals/goldset.jsonl")
README_PATH = Path("README.md")
LAST_RESULTS_PATH = Path("data/evals_last.json")
MODES = ["dense", "bm25", "hybrid", "hybrid_rerank"]


@dataclass(frozen=True)
class Metrics:
    recall_at_5: float
    recall_at_10: float
    mrr: float


def recall_at_k(ranked_ids: list[int], gold_id: int, k: int) -> float:
    return 1.0 if gold_id in ranked_ids[:k] else 0.0


def reciprocal_rank(ranked_ids: list[int], gold_id: int) -> float:
    try:
        return 1.0 / (ranked_ids.index(gold_id) + 1)
    except ValueError:
        return 0.0


def compute_metrics(gold_ids: list[int], ranked_results: list[list[int]]) -> Metrics:
    if len(gold_ids) != len(ranked_results):
        raise ValueError("gold_ids and ranked_results must have the same length")
    if not gold_ids:
        return Metrics(recall_at_5=0.0, recall_at_10=0.0, mrr=0.0)
    total = len(gold_ids)
    return Metrics(
        recall_at_5=sum(recall_at_k(ids, gold_id, 5) for gold_id, ids in zip(gold_ids, ranked_results, strict=True)) / total,
        recall_at_10=sum(recall_at_k(ids, gold_id, 10) for gold_id, ids in zip(gold_ids, ranked_results, strict=True)) / total,
        mrr=sum(reciprocal_rank(ids, gold_id) for gold_id, ids in zip(gold_ids, ranked_results, strict=True)) / total,
    )


def run_evals() -> dict[str, Any]:
    from app.retrieval.service import search

    goldset = load_goldset()
    if not goldset:
        raise SystemExit("app/evals/goldset.jsonl is empty. Review data/candidates.jsonl and copy ~30 keeper lines first.")

    gold_ids = [int(item["chunk_id"]) for item in goldset]
    results: dict[str, Metrics] = {}
    for mode in MODES:
        ranked_results: list[list[int]] = []
        for item in goldset:
            found = search(str(item["question"]), mode=mode, k=10)
            ranked_results.append([result.chunk_id for result in found])
        results[mode] = compute_metrics(gold_ids, ranked_results)

    table = format_markdown_table(results)
    rewrite_readme(table)
    payload = {
        "table_markdown": table,
        "modes": {
            mode: {
                "recall_at_5": metrics.recall_at_5,
                "recall_at_10": metrics.recall_at_10,
                "mrr": metrics.mrr,
            }
            for mode, metrics in results.items()
        },
    }
    LAST_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LAST_RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(table)
    return payload


def load_goldset() -> list[dict[str, Any]]:
    if not GOLDSET_PATH.exists():
        return []
    items: list[dict[str, Any]] = []
    for line in GOLDSET_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(json.loads(line))
    return items


def load_last_results() -> dict[str, Any]:
    if not LAST_RESULTS_PATH.exists():
        return {"table_markdown": "", "modes": {}}
    return json.loads(LAST_RESULTS_PATH.read_text(encoding="utf-8"))


def format_markdown_table(results: dict[str, Metrics]) -> str:
    lines = [
        "| Mode | Recall@5 | Recall@10 | MRR |",
        "|---|---:|---:|---:|",
    ]
    for mode in MODES:
        metrics = results[mode]
        lines.append(f"| {mode} | {metrics.recall_at_5:.3f} | {metrics.recall_at_10:.3f} | {metrics.mrr:.3f} |")
    return "\n".join(lines)


def rewrite_readme(table: str) -> None:
    if not README_PATH.exists():
        return
    readme = README_PATH.read_text(encoding="utf-8")
    start = "<!-- EVALS_START -->"
    end = "<!-- EVALS_END -->"
    before, marker, rest = readme.partition(start)
    if not marker:
        return
    _, marker_end, after = rest.partition(end)
    if not marker_end:
        return
    README_PATH.write_text(f"{before}{start}\n{table}\n{end}{after}", encoding="utf-8")


if __name__ == "__main__":
    run_evals()
