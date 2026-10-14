"""Real GPU CLI cycle and the full random-game regression bank."""

import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, "/opt")
sys.path.insert(0, "/opt/chesszero/chesszero")
from run_training import evaluate
from config.model_config import ModelConfig


class NoPredictions:
    def eval(self):
        pass

    def __call__(self, states):
        raise AssertionError("random-only games must not request inference")


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "/opt/chesszero/chesszero/tests/test_pipe_inference.py",
            "/opt/chesszero/chesszero/tests/test_inference_mode.py",
            "/opt/chesszero/chesszero/tests/test_training_targets.py",
            "/opt/chesszero/chesszero/tests/test_gradient_accumulation.py",
            "/opt/chesszero/chesszero/tests/test_checkpoints.py",
        ],
        check=True,
    )
    evaluate(
        NoPredictions(),
        ModelConfig(),
        out,
        "random-bank-regression",
        32,
        20000,
        random_only=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-u",
            "/opt/chesszero/chesszero/train.py",
            "--checkpoint-dir",
            str(out / "checkpoints"),
            "--epochs",
            "2",
            "--eval-interval",
            "1",
            "--evaluation-games",
            "2",
            "--workers",
            "2",
            "--games-per-iteration",
            "2",
            "--simulations",
            "2",
            "--batch-size",
            "64",
            "--evaluate-random",
            "4",
        ],
        check=True,
    )
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from model.network import ChessNet

    model, state = ChessNet.load_checkpoint(out / "checkpoints", 2)
    assert state["completed_epochs"] == 2
    assert int(state["optimizer_state"]["step"]) >= 1
    assert all(
        bool(mx.all(mx.isfinite(v)).item()) for _, v in tree_flatten(model.parameters())
    )
    (out / "result.json").write_text(
        json.dumps(
            dict(
                passed=True,
                cli_epochs=2,
                original_architecture=True,
                completed_epochs=state["completed_epochs"],
                adam_step=int(state["optimizer_state"]["step"]),
            )
        )
    )
    print("PREFLIGHT_PASSED", flush=True)
