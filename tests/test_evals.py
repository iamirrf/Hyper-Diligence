from app.evals.run_evals import compute_metrics, recall_at_k, reciprocal_rank


def test_recall_and_reciprocal_rank() -> None:
    ranked = [7, 3, 9, 10]

    assert recall_at_k(ranked, gold_id=3, k=1) == 0.0
    assert recall_at_k(ranked, gold_id=3, k=2) == 1.0
    assert reciprocal_rank(ranked, gold_id=3) == 0.5
    assert reciprocal_rank(ranked, gold_id=99) == 0.0


def test_compute_metrics() -> None:
    metrics = compute_metrics(
        gold_ids=[3, 8, 9],
        ranked_results=[
            [1, 2, 3],
            [8, 2, 1],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
        ],
    )

    assert metrics.recall_at_5 == 2 / 3
    assert metrics.recall_at_10 == 1.0
    assert metrics.mrr == ((1 / 3) + 1 + (1 / 9)) / 3
