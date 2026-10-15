import importlib.util
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "pr29-first-result"))
spec = importlib.util.spec_from_file_location(
    "restart_training", Path(__file__).with_name("restart_training.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def rows(scores):
    return [dict(pair_id=i // 2, color=i % 2, score=s) for i, s in enumerate(scores)]


def test_comparison_joins_out_of_order_games_by_identity():
    before = rows([1, 1, 1, 0.5])
    after = rows([0, 1, 0.5, 1])
    comparison = module.compare(before, list(reversed(after)))
    assert comparison["white_score_change"] == -0.75
    assert comparison["black_score_change"] == 0.25
    assert comparison["change_in_white_minus_black_score"] == -1
    assert comparison == module.compare(before, after)


def test_draws_are_half_points_and_censored_games_are_explicit():
    result = module.color_summary(rows([0.5, None]))
    assert result["white"]["score_lower_bound"] == 0.5
    assert result["black"]["censored"] == 1
    assert result["black"]["score_lower_bound"] == 0
    assert result["black"]["score_upper_bound"] == 1
    with pytest.raises(AssertionError, match="Censored"):
        module.compare(rows([0.5, None]), rows([0.5, 1]))


def test_missing_or_duplicate_game_ids_cannot_be_paired():
    with pytest.raises(AssertionError):
        module.compare(rows([0, 1]), rows([0, 1]) * 2)
