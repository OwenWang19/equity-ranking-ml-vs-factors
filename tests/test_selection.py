import pytest

from equity_alpha.selection import select_candidate


def test_zero_candidate_blocks_negative_winners_and_worse_than_zero_mse():
    metrics = {
        "a": {"rank_ic": -0.01, "mse": 0.5, "zero_mse": 1},
        "b": {"rank_ic": 0.2, "mse": 1.1, "zero_mse": 1},
    }
    result = select_candidate(["a", "b"], metrics, metric="sector_mse", cash_hurdle=True)
    assert result["candidate"] == "a" and result["active"] is False
    metrics["a"]["rank_ic"] = 0.1
    assert select_candidate(["a", "b"], metrics, metric="sector_mse", cash_hurdle=True)["active"]


def test_deterministic_ties_and_v1_negative_winner_preserved():
    metrics = {name: {"rank_ic": -0.01, "mse": 1, "zero_mse": 1} for name in ["a", "b"]}
    assert select_candidate(["b", "a"], metrics)["candidate"] == "b"
    assert select_candidate(["b", "a"], metrics)["active"] is True
    assert select_candidate(["b", "a"], metrics, metric="sector_mse", cash_hurdle=True)["active"] is False
    with pytest.raises(ValueError):
        select_candidate([], metrics)
