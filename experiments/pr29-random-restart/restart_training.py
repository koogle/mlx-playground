"""Repeat the original ten updates with a new random initialization only."""

import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, "/opt/chesszero/chesszero")
from run_training import evaluate, training_worker, write_json
from config.model_config import ModelConfig
from model.pipe_inference import run_pipe_games
from model.self_play import create_batches

SEED = 20261015
GENERATIONS = 10
PREVIOUS_RUN = "pr29-first-result-20261014-202746"


def color_summary(rows):
    result = {}
    for color, name in ((0, "white"), (1, "black")):
        games = [r for r in rows if r["color"] == color]
        assert games
        scores = [r["score"] for r in games]
        result[name] = dict(
            games=len(games),
            wins=scores.count(1.0),
            draws=scores.count(0.5),
            losses=scores.count(0.0),
            censored=scores.count(None),
            score_lower_bound=sum(s or 0 for s in scores) / len(scores),
            score_upper_bound=sum(1 if s is None else s for s in scores) / len(scores),
        )
    result["white_minus_black_score"] = (
        result["white"]["score_lower_bound"] - result["black"]["score_lower_bound"]
    )
    return result


def compare(before, after):
    # Workers complete out of order: join by identity, never zip result lists.
    keyed_before = {(r["pair_id"], r["color"]): r for r in before}
    keyed_after = {(r["pair_id"], r["color"]): r for r in after}
    assert len(keyed_before) == len(before) == len(after) == len(keyed_after)
    assert keyed_before.keys() == keyed_after.keys()
    assert all(r["score"] is not None for r in before + after), "Censored comparison"
    initial, final = color_summary(before), color_summary(after)
    pair_ids = sorted({p for p, _ in keyed_before})
    white_changes, black_changes = [], []
    for pair in pair_ids:
        white_changes.append(
            keyed_after[pair, 0]["score"] - keyed_before[pair, 0]["score"]
        )
        black_changes.append(
            keyed_after[pair, 1]["score"] - keyed_before[pair, 1]["score"]
        )
    deltas = np.array(white_changes) - np.array(black_changes)
    bootstrap = (
        np.random.default_rng(889).choice(deltas, (10000, len(deltas))).mean(axis=1)
    )
    return dict(
        initial=initial,
        final=final,
        white_score_change=float(np.mean(white_changes)),
        black_score_change=float(np.mean(black_changes)),
        change_in_white_minus_black_score=float(deltas.mean()),
        change_in_white_minus_black_pair_bootstrap_95=np.quantile(
            bootstrap, [0.025, 0.975]
        ).tolist(),
        interpretation="Negative gap change means training shifted relative performance toward Black. "
        "One seed is a replication check, not proof of a general cause. Draws count as half a point.",
    )


def main():
    import logging
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from model.training import Trainer
    from model.network import ChessNet

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    mx.set_default_device(mx.gpu)
    mx.random.seed(SEED)
    np.random.seed(SEED)
    out = Path(sys.argv[1])
    start = time.monotonic()
    config = ModelConfig(
        n_games_per_iteration=32,
        n_simulations=64,
        batch_size=64,
        learning_rate=1e-4,
        n_epochs=10,
    )
    write_json(
        out / "experiment.json",
        dict(
            source_revision=os.environ["CHESSZERO_SOURCE_REVISION"],
            source_hashes={
                str(p.relative_to("/opt/chesszero")): hashlib.sha256(
                    p.read_bytes()
                ).hexdigest()
                for p in Path("/opt/chesszero/chesszero").rglob("*.py")
            },
            config=vars(config),
            workers=32,
            seed=SEED,
            previous_seed=20261014,
            previous_run=PREVIOUS_RUN,
            initialization="Fresh Trainer; no model or optimizer checkpoint loaded",
            generations=GENERATIONS,
            training_seed_protocol="Unchanged: 1000000 + generation*1000 + game_id",
            protocol="Repeat original 10 generations, 320 complete fresh self-play games, 10 accumulated Adam updates. "
            "No augmentation, BN reset, replay, or early success stop. Evaluate 64 games vs random both "
            "before and after, identical openings/random seeds (bank 10000), 32 per color. "
            "Save per-color WDL and paired gap-change bootstrap. Evaluation never enters training. "
            "This diagnoses color imbalance, not superiority to the initial model.",
            job_budget_seconds=10800,
        ),
    )
    trainer = Trainer(config, checkpoint_dir=str(out / "checkpoints"))
    mx.eval(trainer.model.parameters())
    trainer._save_checkpoint(0)
    initial_fingerprint = hashlib.sha256()
    for name, value in tree_flatten(trainer.model.parameters()):
        initial_fingerprint.update(name.encode())
        initial_fingerprint.update(np.asarray(value).tobytes())
    write_json(
        out / "initialization.json",
        dict(
            seed=SEED,
            restored_checkpoint=False,
            parameter_sha256=initial_fingerprint.hexdigest(),
            optimizer_step=int(trainer.optimizer.state["step"]),
        ),
    )
    print(
        "FRESH_RANDOM_INITIALIZATION", SEED, initial_fingerprint.hexdigest(), flush=True
    )
    evaluate(trainer.model, config, out, "initial-vs-random", 32, 10000)
    initial_rows = json.loads((out / "initial-vs-random-games.json").read_text())
    initial_colors = color_summary(initial_rows)
    write_json(out / "initial-colors.json", initial_colors)
    print("INITIAL_COLOR_SCORES", json.dumps(initial_colors), flush=True)
    generations = []
    for generation in range(1, GENERATIONS + 1):
        trainer.model.eval()
        mx.clear_cache()
        print("GENERATION_START", generation, flush=True)
        began = time.monotonic()
        rows, failures, stats = run_pipe_games(
            trainer.model,
            config,
            training_worker,
            [(config, i, generation) for i in range(32)],
            32,
        )
        if failures or len(rows) != 32:
            raise RuntimeError(
                f"Incomplete generation: {len(rows)} games / {failures} failures"
            )
        games = [(r["history"], r["white_result"]) for r in rows]
        row = dict(
            generation=generation,
            games=len(games),
            failures=failures,
            positions=sum(len(h) for h, _ in games),
            white_wins=sum(v == 1 for _, v in games),
            black_wins=sum(v == -1 for _, v in games),
            draws=sum(v == 0 for _, v in games),
            inference=stats,
            self_play_seconds=time.monotonic() - began,
        )
        del rows
        began = time.monotonic()
        row["losses"] = trainer.train_on_batches(
            create_batches(games, config.batch_size)
        )
        row["learning_seconds"] = time.monotonic() - began
        row["adam_step"] = int(trainer.optimizer.state["step"])
        assert row["adam_step"] == generation
        assert all(
            bool(mx.all(mx.isfinite(v)).item())
            for _, v in tree_flatten(trainer.model.parameters())
        )
        trainer._save_checkpoint(generation)
        del games
        gc.collect()
        mx.clear_cache()
        generations.append(row)
        write_json(out / "generations.json", generations)
        print("GENERATION_COMPLETE", json.dumps(row), flush=True)
    restored, state = ChessNet.load_checkpoint(out / "checkpoints", GENERATIONS)
    assert int(state["optimizer_state"]["step"]) == GENERATIONS
    assert all(
        bool(mx.array_equal(a, b).item())
        for (_, a), (_, b) in zip(
            tree_flatten(trainer.model.parameters()),
            tree_flatten(restored.parameters()),
        )
    )
    del trainer, state
    gc.collect()
    mx.clear_cache()
    evaluate(restored, config, out, "trained-vs-random", 32, 10000)
    final_rows = json.loads((out / "trained-vs-random-games.json").read_text())
    result = dict(
        status="completed",
        seed=SEED,
        generations=len(generations),
        elapsed_seconds=time.monotonic() - start,
        checkpoint_roundtrip=True,
        comparison=compare(initial_rows, final_rows),
    )
    old = Path("/results") / PREVIOUS_RUN
    result["previous_run_comparison"] = compare(
        json.loads((old / "initial-vs-random-games.json").read_text()),
        json.loads((old / "trained-vs-random-games.json").read_text()),
    )
    write_json(out / "result.json", result)
    print("RUN_COMPLETE", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
