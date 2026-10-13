import sys
from pathlib import Path

sys.path.insert(0, "/opt/chesszero/chesszero")
import json
import time
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten
from config.model_config import ModelConfig
from model.training import Trainer
from model.self_play import play_single_game, create_batches
from model.game_workers import run_game_workers


def timed_game(model, config, game_id, directory, sender, max_workers):
    mx.set_default_device(mx.gpu)
    np.random.seed(9000 + game_id)
    mx.random.seed(9000 + game_id)
    start = time.monotonic()
    play_single_game(model, config, game_id, sender, max_workers)
    (Path(directory) / f"game-{game_id}.json").write_text(
        json.dumps({"seconds": time.monotonic() - start})
    )


def main():
    mx.set_default_device(mx.gpu)
    out = Path(sys.argv[1])
    source = Path("/results/learning-merged-20261013-205953/checkpoints")
    trainer = Trainer(ModelConfig(), checkpoint_dir=source, resume_epoch=4)
    # Save trial checkpoints separately from the resume source.
    trainer.checkpoint_dir = out / "checkpoints"
    trainer.config.n_games_per_iteration = 16
    trainer.config.n_simulations = 32
    trainer.config.batch_size = 64
    trainer.model.eval()
    mx.eval(trainer.model.parameters())
    mx.clear_cache()
    start = time.monotonic()
    jobs = [(trainer.model, trainer.config, i, str(out)) for i in range(16)]
    results, failures = run_game_workers(timed_game, jobs, 16)
    seconds = time.monotonic() - start
    if failures:
        raise RuntimeError(f"{failures} games failed")
    games = [(r["history"], r["white_result"]) for r in results]
    learn_start = time.monotonic()
    losses = trainer.train_on_batches(create_batches(games, 64))
    learning_seconds = time.monotonic() - learn_start
    finite = all(
        bool(mx.all(mx.isfinite(v)).item())
        for _, v in tree_flatten(trainer.model.parameters())
    )
    if not finite:
        raise RuntimeError("Nonfinite parameters")
    trainer._save_checkpoint(5)
    row = {
        "workers": 16,
        "games": len(games),
        "failures": failures,
        "positions": sum(len(h) for h, _ in games),
        "game_plies": [len(h) for h, _ in games],
        "self_play_seconds": seconds,
        "games_per_minute": len(games) * 60 / seconds,
        "individual_game_seconds": [
            json.loads((out / f"game-{i}.json").read_text())["seconds"]
            for i in range(16)
        ],
        "learning_seconds": learning_seconds,
        "losses": losses,
        "adam_step": int(trainer.optimizer.state["step"]),
        "finite_parameters": finite,
        "source_revision": Path("/opt/chesszero/REVISION").read_text().strip(),
        "initial_generation": 4,
        "n_simulations": 32,
        "seed_base": 9000,
    }
    (out / "result.json").write_text(json.dumps(row, indent=2))
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
