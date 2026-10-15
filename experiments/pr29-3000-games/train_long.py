"""Continue the fresh-seed checkpoint for exactly 3000 additional self-play games."""

import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time


sys.path.insert(0, "/opt/chesszero/chesszero")
from run_training import evaluate, training_worker, write_json
from restart_training import color_summary
from model.pipe_inference import run_pipe_games
from model.self_play import create_batches

SOURCE = Path("/results/pr29-random-restart-20261015-085706")
TOTAL_GAMES = 3000
INTERVAL = 500


def main():
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from config.model_config import ModelConfig
    from model.training import Trainer
    from model.network import ChessNet
    from matches import confirmation_openings, evaluate as evaluate_matches
    from run_training import OPENINGS

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    mx.set_default_device(mx.gpu)
    mx.random.seed(20261015)
    out = Path(sys.argv[1])
    checkpoints = out / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    progress_path = out / "progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
    else:
        assert json.loads((SOURCE / "result.json").read_text())["status"] == "completed"
        manifest = json.loads((SOURCE / "checkpoints/state_epoch_10.json").read_text())
        for key in (
            "model_file",
            "optimizer_file",
            "best_model_file",
            "best_optimizer_file",
        ):
            if manifest.get(key):
                shutil.copyfile(
                    SOURCE / "checkpoints" / manifest[key], checkpoints / manifest[key]
                )
        shutil.copyfile(
            SOURCE / "checkpoints/state_epoch_10.json",
            checkpoints / "state_epoch_10.json",
        )
        progress = dict(
            games_completed=0,
            checkpoint_epoch=10,
            additional_updates=0,
            phase="initializing",
            generations=[],
            milestones=[],
        )
        write_json(progress_path, progress)
        write_json(
            out / "experiment.json",
            dict(
                source_run=SOURCE.name,
                source_epoch=10,
                source_revision=os.environ["CHESSZERO_SOURCE_REVISION"],
                source_hashes={
                    str(p.relative_to("/opt/chesszero")): hashlib.sha256(
                        p.read_bytes()
                    ).hexdigest()
                    for p in Path("/opt/chesszero/chesszero").rglob("*.py")
                },
                additional_games=TOTAL_GAMES,
                previous_games=320,
                reporting_interval=INTERVAL,
                expected_additional_updates=96,
                workers=32,
                simulations=64,
                microbatch=64,
                learning_rate=1e-4,
                protocol="Resume model AND Adam at epoch10. Fresh on-policy self-play, no replay or augmentation. "
                "One position-weighted accumulated Adam update after each batch of up to32 complete games. "
                "Use15 batches of32 plusone batch of20 per500-game milestone, exactly3000 additional games. "
                "Evaluate64 games vs random at every milestone, draws half; no early success stop. "
                "Final additional64-game direct match against frozen epoch0, no evaluation data training.",
            ),
        )
    trainer = Trainer(
        ModelConfig(),
        checkpoint_dir=str(checkpoints),
        resume_epoch=progress["checkpoint_epoch"],
    )
    config = trainer.config
    assert (
        config.n_simulations == 64
        and config.batch_size == 64
        and config.learning_rate == 1e-4
    )
    assert int(trainer.optimizer.state["step"]) == progress["checkpoint_epoch"]
    config.n_epochs = 106
    print(
        "RESUMED",
        progress["checkpoint_epoch"],
        "ADAM_STEP",
        int(trainer.optimizer.state["step"]),
        "ADDITIONAL_GAMES",
        progress["games_completed"],
        flush=True,
    )
    for target in range(INTERVAL, TOTAL_GAMES + 1, INTERVAL):
        while progress["games_completed"] < target:
            count = min(32, target - progress["games_completed"])
            epoch = progress["checkpoint_epoch"] + 1
            progress["phase"] = "self_play"
            progress["active_batch_games"] = count
            write_json(progress_path, progress)
            trainer.model.eval()
            mx.clear_cache()
            began = time.monotonic()
            print(
                "SELF_PLAY_START",
                epoch,
                count,
                "TOTAL_DONE",
                progress["games_completed"],
                flush=True,
            )
            rows, failures, stats = run_pipe_games(
                trainer.model,
                config,
                training_worker,
                [(config, i, 10000 + epoch) for i in range(count)],
                count,
            )
            if failures or len(rows) != count:
                raise RuntimeError(
                    f"Incomplete self-play: {len(rows)}/{count}, {failures} failures; no update"
                )
            games = [(r["history"], r["white_result"]) for r in rows]
            record = dict(
                epoch=epoch,
                games=count,
                positions=sum(len(h) for h, _ in games),
                failures=failures,
                white_wins=sum(v == 1 for _, v in games),
                black_wins=sum(v == -1 for _, v in games),
                draws=sum(v == 0 for _, v in games),
                inference=stats,
                self_play_seconds=time.monotonic() - began,
            )
            del rows
            progress["phase"] = "training"
            write_json(progress_path, progress)
            record["losses"] = trainer.train_on_batches(
                create_batches(games, config.batch_size)
            )
            assert int(trainer.optimizer.state["step"]) == epoch
            assert all(
                bool(mx.all(mx.isfinite(v)).item())
                for _, v in tree_flatten(trainer.model.parameters())
            )
            trainer._save_checkpoint(epoch)
            # Publish progress only after a complete model+optimizer checkpoint exists.
            progress["games_completed"] += count
            progress["checkpoint_epoch"] = epoch
            progress["additional_updates"] += 1
            progress["generations"].append(record)
            write_json(progress_path, progress)
            print(
                "UPDATE_COMPLETE",
                json.dumps(
                    dict(
                        games_completed=progress["games_completed"],
                        additional_updates=progress["additional_updates"],
                        checkpoint_epoch=epoch,
                        losses=record["losses"],
                    )
                ),
                flush=True,
            )
            del games
            gc.collect()
            mx.clear_cache()
        report_path = out / f"milestone-{target:04d}.json"
        if report_path.exists():
            continue
        progress["phase"] = "evaluation"
        write_json(progress_path, progress)
        evaluation = evaluate(
            trainer.model, config, out, f"vs-random-{target:04d}", 32, 10000
        )
        rows = json.loads((out / f"vs-random-{target:04d}-games.json").read_text())
        report = dict(
            games_completed=target,
            cumulative_self_play_games=320 + target,
            additional_updates=progress["additional_updates"],
            checkpoint_epoch=progress["checkpoint_epoch"],
            training_positions=sum(g["positions"] for g in progress["generations"]),
            failed_training_games=sum(g["failures"] for g in progress["generations"]),
            evaluation=evaluation,
            by_color=color_summary(rows),
        )
        write_json(report_path, report)
        progress["milestones"].append(target)
        write_json(progress_path, progress)
        print("MILESTONE", json.dumps(report), flush=True)
    progress["phase"] = "final_baseline_match"
    write_json(progress_path, progress)
    baseline, state = ChessNet.load_checkpoint(SOURCE / "checkpoints", 0)
    del state
    openings = confirmation_openings(OPENINGS, 80000, pairs=32)
    write_json(out / "baseline-openings.json", openings)
    baseline_result = evaluate_matches(
        trainer.model,
        baseline,
        config,
        openings,
        write_json,
        out,
        "final-vs-initial",
        workers=32,
        confirmation=True,
    )
    # Verify the latest saved tensors and optimizer step.
    restored, state = ChessNet.load_checkpoint(
        checkpoints, progress["checkpoint_epoch"]
    )
    assert int(state["optimizer_state"]["step"]) == 106
    assert all(
        bool(mx.array_equal(a, b).item())
        for (_, a), (_, b) in zip(
            tree_flatten(restored.parameters()),
            tree_flatten(trainer.model.parameters()),
        )
    )
    progress["phase"] = "completed"
    write_json(progress_path, progress)
    write_json(
        out / "result.json",
        dict(
            status="completed",
            games=TOTAL_GAMES,
            additional_updates=96,
            checkpoint_epoch=106,
            checkpoint_roundtrip=True,
            final_milestone=json.loads((out / "milestone-3000.json").read_text()),
            direct_vs_initial=baseline_result,
        ),
    )
    print("RUN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
