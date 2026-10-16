"""Promotion games are future training data, with one label per position."""

from unittest.mock import patch

import numpy as np
import pytest

from config.model_config import ModelConfig
from model.self_play import create_batches
from model.training import IncompleteEvaluationError, Trainer


def history(marker=0):
    states = np.zeros((2, 19, 8, 8), dtype=np.float32)
    states[0, 12] = 1
    states[:, 17] = marker
    return [(state, np.array([1, 0, 0, 0], dtype=np.float32), None) for state in states]


def trainer(directory):
    config = ModelConfig(
        n_filters=2,
        n_residual_blocks=0,
        policy_output_dim=4,
        n_eval_games=2,
        batch_size=16,
    )
    instance = Trainer(config, directory)
    instance._accept_current_model()
    return instance


@pytest.mark.parametrize("outcome", [-1, 0, 1])
def test_both_candidate_colors_become_white_results_once(tmp_path, outcome):
    instance = trainer(tmp_path)
    results = [
        dict(result=outcome, perspective_color=color, history=history(color))
        for color in (0, 1)
    ]
    with patch("model.training.run_pipe_games", return_value=(results, 0, {})):
        instance.evaluate_against_best(2)
    assert len(instance.pending_evaluation_games) == 2
    _, _, targets = next(create_batches(instance.pending_evaluation_games, 16))
    np.testing.assert_array_equal(targets, [outcome, -outcome, -outcome, outcome])


@pytest.mark.parametrize("outcome", [-1, 1])
def test_accepted_and_rejected_games_are_consumed_only_next_epoch(tmp_path, outcome):
    instance = trainer(tmp_path)
    results = [
        dict(result=outcome, perspective_color=color, history=history(10 + color))
        for color in (0, 1)
    ]
    samples = []

    def train(batches):
        batches = list(batches)
        samples.append(
            np.concatenate([np.asarray(b[0])[:, 17, 0, 0] for b in batches]).tolist()
        )
        return (0, 0, 0, 0)

    with (
        patch(
            "model.training.generate_games", side_effect=lambda *args: [(history(1), 1)]
        ),
        patch("model.training.run_pipe_games", return_value=(results, 0, {})),
        patch.object(instance, "train_on_batches", side_effect=train),
    ):
        instance.train(n_epochs=3, n_workers=1)
    assert samples == [[1, 1], [1, 1, 10, 10, 11, 11], [1, 1, 10, 10, 11, 11]]
    assert len(instance.pending_evaluation_games) == 2


def test_pending_games_survive_checkpoint_resume(tmp_path):
    original = trainer(tmp_path)
    original.pending_evaluation_games = [(history(12), -1), (history(13), 0)]
    original._save_checkpoint(1)
    resumed = Trainer(original.config, tmp_path, resume_epoch=1)
    assert len(resumed.pending_evaluation_games) == 2
    for actual, expected in zip(
        resumed.pending_evaluation_games, original.pending_evaluation_games
    ):
        assert actual[1] == expected[1]
        for a, e in zip(actual[0], expected[0]):
            np.testing.assert_array_equal(a[0], e[0])
            np.testing.assert_array_equal(a[1], e[1])


def test_failed_match_and_random_evaluation_do_not_supply_training_games(tmp_path):
    instance = trainer(tmp_path)
    result = dict(result=1, perspective_color=0, history=history())
    for returned in (([result], 1, {}), ([result, {"result": 1}], 0, {})):
        with patch("model.training.run_pipe_games", return_value=returned):
            with pytest.raises(IncompleteEvaluationError):
                instance.evaluate_against_best(2)
        assert instance.pending_evaluation_games == []
    with patch("model.training.run_pipe_games", return_value=([result, result], 0, {})):
        instance.evaluate_against_random(2)
    assert instance.pending_evaluation_games == []


def test_failed_training_preserves_pending_examples(tmp_path):
    instance = trainer(tmp_path)
    instance.pending_evaluation_games = [(history(), 1)]
    with (
        patch("model.training.generate_games", return_value=[]),
        patch.object(
            instance, "train_on_batches", side_effect=RuntimeError("update failed")
        ),
    ):
        with pytest.raises(RuntimeError, match="update failed"):
            instance.train(n_epochs=1)
    assert len(instance.pending_evaluation_games) == 1
