import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from chess_engine.bitboard import BitBoard
from model.move_encoding import encode_move


@pytest.fixture
def driver(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "experiments/pr29-first-result"))
    monkeypatch.syspath_prepend(str(root / "experiments/pr29-random-restart"))
    spec = importlib.util.spec_from_file_location(
        "gated_driver", Path(__file__).with_name("train_gated.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "moves,white_result",
    [
        ("f2f3 e7e5 g2g4 d8h4", -1),
        ("e2e4 e7e5 f1c4 b8c6 d1h5 g8f6 h5f7", 1),
        ("g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8", 0),
    ],
)
def test_training_evidence_checks_actual_outcomes_and_both_label_signs(
    driver, tmp_path, moves, white_result
):
    board = BitBoard()
    history, played = [], []
    for u in moves.split():
        move = ((int(u[1]) - 1, ord(u[0]) - 97), (int(u[3]) - 1, ord(u[2]) - 97))
        policy = np.zeros(4672, dtype=np.float32)
        policy[encode_move(*move)] = 1
        history.append((board.state.copy(), policy, None))
        assert board.make_move(*move)
        played.append(move)
    row = dict(game_id=0, moves=played, history=history, white_result=white_result)
    assert driver.validate_games([row], tmp_path, 1)["games_verified"] == 1
    arrays = np.load(tmp_path / "training-samples-0001.npz")
    np.testing.assert_array_equal(
        arrays["targets"],
        [white_result if i % 2 == 0 else -white_result for i in range(len(played))],
    )
    if white_result:
        with pytest.raises(AssertionError):
            driver.validate_games([dict(row, white_result=-white_result)], tmp_path, 2)
    history[0][0][12].fill(0)
    with pytest.raises(AssertionError):
        driver.validate_games([row], tmp_path, 3)


def test_schedule_has_exact_500_game_reports_and_3000_total(driver):
    done, batches = 0, []
    while done < 3000:
        count = driver.batch_count(done)
        batches.append(count)
        done += count
        assert count in (20, 32)
        if len(batches) % 16 == 0:
            assert done == len(batches) // 16 * 500
    assert len(batches) == 96 and done == 3000


def test_retention_preserves_latest_and_milestone_checkpoints(driver, tmp_path):
    for epoch in (0, 1, 2, 16, 17, 18):
        name = f"model-{epoch}.safetensors"
        (tmp_path / name).write_text("test tensor placeholder")
        (tmp_path / f"state_epoch_{epoch}.json").write_text(
            json.dumps(dict(model_file=name))
        )
    driver.prune_checkpoints(tmp_path, 18)
    assert sorted(p.name for p in tmp_path.glob("state_*.json")) == [
        "state_epoch_0.json",
        "state_epoch_16.json",
        "state_epoch_17.json",
        "state_epoch_18.json",
    ]
    assert not (tmp_path / "model-1.safetensors").exists()
