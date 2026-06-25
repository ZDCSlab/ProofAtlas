from __future__ import annotations

from proofatlas.metrics import ranking_row, summarize_rows


def test_ranking_row_and_summary() -> None:
    row = ranking_row(["a", "b", "c"], {"b", "d"}, {"a", "b", "c"}, [1, 2, 3])
    assert row["gold_total_count"] == 2
    assert row["gold_in_pool_count"] == 1
    assert row["gold_missing_from_pool_count"] == 1
    assert row["rank_of_first_gold"] == 2
    assert row["Recall@1"] == 0.0
    assert row["Recall@2"] == 1.0

    summary = summarize_rows([row], [1, 2, 3])
    assert summary["evaluated_queries"] == 1
    assert summary["evaluated_retrievable_queries"] == 1
    assert summary["Recall@2"] == 1.0
