"""Fresh-seed rerun: per-minibatch Adam and the production promotion rule."""

import gc
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, "/opt/chesszero/chesszero")
from run_training import OPENINGS, evaluate, training_worker, write_json
from restart_training import color_summary
from model.pipe_inference import run_pipe_games
from model.self_play import create_batches

SOURCE = Path("/results/pr29-random-restart-20261015-085706")
TOTAL = 3000


def batch_count(done):
    return min(32, 500 - done % 500, TOTAL - done)


def validate_games(rows, out, generation):
    """Replay actual training moves independently; retain histories and samples."""
    import chess
    from model.move_encoding import encode_move

    histories, states, policies, targets = [], [], [], []
    for row in rows:
        board = chess.Board()
        assert len(row["moves"]) == len(row["history"])
        sample_indices = set(
            np.linspace(
                0, len(row["history"]) - 1, min(8, len(row["history"])), dtype=int
            )
        )
        moves, samples = [], []
        for ply, ((source, target), (state, policy, _)) in enumerate(
            zip(row["moves"], row["history"])
        ):
            move = chess.Move(
                chess.square(source[1], source[0]),
                chess.square(target[1], target[0]),
                promotion=target[2] + 1 if len(target) == 3 else None,
            )
            assert move in board.legal_moves
            assert state[12, 0, 0] == int(board.turn)
            assert np.isfinite(policy).all() and (policy >= 0).all()
            assert np.isclose(policy.sum(), 1, atol=1e-5)
            assert policy[encode_move(tuple(source), tuple(target))] > 0
            label = row["white_result"] if board.turn else -row["white_result"]
            if ply in sample_indices:
                # Build all planes independently from the oracle, not native code.
                expected = np.zeros((19, 8, 8), dtype=np.float32)
                for square, piece in board.piece_map().items():
                    color = 0 if piece.color else 1
                    rank, file = divmod(square, 8)
                    expected[piece.piece_type - 1 + 6 * color, rank, file] = 1
                expected[12].fill(board.turn)
                for color in (0, 1):
                    expected[13 + 2 * color].fill(
                        board.has_kingside_castling_rights(not color)
                    )
                    expected[14 + 2 * color].fill(
                        board.has_queenside_castling_rights(not color)
                    )
                expected[17].fill(ply / 200)
                expected[18].fill(board.halfmove_clock / 100)
                np.testing.assert_allclose(state, expected, atol=1e-6)
                samples.append(
                    dict(ply=ply, fen=board.fen(), side_to_move_target=label)
                )
                states.append(state)
                policies.append(policy)
                targets.append(label)
            board.push(move)
            moves.append(move.uci())
        outcome = board.outcome(claim_draw=True)
        assert outcome is not None
        white_result = 0 if outcome.winner is None else 1 if outcome.winner else -1
        assert row["white_result"] == white_result
        histories.append(
            dict(
                game_id=row["game_id"],
                moves=moves,
                white_result=white_result,
                termination=outcome.termination.name,
                samples=samples,
            )
        )
    write_json(out / f"training-games-{generation:04d}.json", histories)
    np.savez_compressed(
        out / f"training-samples-{generation:04d}.npz",
        states=np.stack(states),
        policies=np.stack(policies),
        targets=np.asarray(targets, dtype=np.float32),
    )
    return dict(games_verified=len(rows), sampled_positions_verified=len(states))


def prune_checkpoints(directory, epoch):
    """Retain latest two checkpoints and each completed 500-game milestone."""
    for manifest_path in directory.glob("state_epoch_*.json"):
        number = int(manifest_path.stem.split("_")[-1])
        if number == 0 or number % 16 == 0 or number >= epoch - 1:
            continue
        manifest = json.loads(manifest_path.read_text())
        manifest_path.unlink()
        for key in (
            "model_file",
            "optimizer_file",
            "best_model_file",
            "best_optimizer_file",
            "pending_training_file",
        ):
            if manifest.get(key):
                (directory / manifest[key]).unlink(missing_ok=True)


def random_report(model, config, out, name):
    result = evaluate(model, config, out, name, 32, 10000)
    result["by_color"] = color_summary(
        json.loads((out / f"{name}-games.json").read_text())
    )
    write_json(out / f"{name}.json", result)
    return result


def main():
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from config.model_config import ModelConfig
    from model.network import ChessNet
    from model.training import Trainer
    from matches import confirmation_openings, evaluate as evaluate_matches

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
        source_manifest = json.loads(
            (SOURCE / "checkpoints/state_epoch_0.json").read_text()
        )
        for key in ("model_file", "optimizer_file"):
            filename = source_manifest[key]
            shutil.copyfile(SOURCE / "checkpoints" / filename, checkpoints / filename)
        shutil.copyfile(
            SOURCE / "checkpoints/state_epoch_0.json",
            checkpoints / "state_epoch_0.json",
        )
        progress = dict(
            games_completed=0,
            generation=0,
            attempted_updates=0,
            accepted_updates=0,
            accepted_generation=0,
            promotions=0,
            rejections=0,
            phase="initializing",
            generations=[],
            milestones=[],
        )
        write_json(progress_path, progress)
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
                initial_source_run=SOURCE.name,
                initial_source_epoch=0,
                total_self_play_games=TOTAL,
                seed=20261015,
                simulations=64,
                minibatch=64,
                learning_rate=1e-4,
                promotion_games=40,
                promotion_threshold=0.55,
                protocol="Exact initial model/Adam from fresh-seed epoch0. One update per minibatch. "
                "Self-play uses accepted model; gate candidate after each up-to32 games using production rule. "
                "Reject <=55% score, draws half, or incomplete match; restore full accepted state. "
                "Every500 games report candidate AND accepted model versus fixed random bank; final direct initial-model match. "
                "All training games independently replayed; keep moves and bounded state/policy/target samples. No replay training.",
            ),
        )
    trainer = Trainer(
        ModelConfig(), str(checkpoints), resume_epoch=progress["generation"]
    )
    config = trainer.config
    config.accumulate_gradients = False
    config.batch_size = 64
    config.learning_rate = 1e-4
    config.n_simulations = 64
    config.n_eval_games = 40
    config.eval_interval_epochs = 1
    trainer.optimizer.learning_rate = config.learning_rate
    if trainer.best_model is None:
        trainer._accept_current_model()
        trainer._save_checkpoint(0)
    assert int(trainer.optimizer.state["step"]) == progress["accepted_updates"]
    print(
        "RESTORED_INITIAL_OR_ACCEPTED",
        json.dumps({k: v for k, v in progress.items() if k not in ("generations",)}),
        flush=True,
    )
    if not (out / "initial-vs-random.json").exists():
        progress["phase"] = "initial_evaluation"
        write_json(progress_path, progress)
        random_report(trainer.best_model, config, out, "initial-vs-random")
    while progress["games_completed"] < TOTAL:
        # Complete a milestone report interrupted after checkpoint publication.
        done = progress["games_completed"]
        if done and done % 500 == 0 and done not in progress["milestones"]:
            finish_milestone(trainer, progress, out)
        generation = progress["generation"] + 1
        count = batch_count(done)
        progress["phase"] = "self_play"
        write_json(progress_path, progress)
        mx.clear_cache()
        print("SELF_PLAY_START", generation, count, "TOTAL_DONE", done, flush=True)
        rows, failures, statistics = run_pipe_games(
            trainer.best_model,
            config,
            training_worker,
            [(config, i, generation) for i in range(count)],
            count,
        )
        if failures or len(rows) != count:
            raise RuntimeError(
                f"Incomplete self-play: {len(rows)}/{count}, {failures} failures"
            )
        verified = validate_games(rows, out, generation)
        games = [(r["history"], r["white_result"]) for r in rows]
        positions = sum(len(h) for h, _ in games)
        record = dict(
            generation=generation,
            games=count,
            positions=positions,
            white_wins=sum(r["white_result"] == 1 for r in rows),
            draws=sum(r["white_result"] == 0 for r in rows),
            black_wins=sum(r["white_result"] == -1 for r in rows),
            inference=statistics,
            validation=verified,
        )
        del rows
        progress["phase"] = "training"
        write_json(progress_path, progress)
        before = int(trainer.optimizer.state["step"])
        record["recycled_promotion_games"] = len(trainer.pending_evaluation_games)
        record["recycled_promotion_positions"] = sum(
            len(history) for history, _ in trainer.pending_evaluation_games
        )
        games.extend(trainer.pending_evaluation_games)
        record["losses"] = trainer.train_on_batches(
            create_batches(games, config.batch_size)
        )
        trainer.pending_evaluation_games = []
        updates = int(trainer.optimizer.state["step"]) - before
        training_positions = positions + record["recycled_promotion_positions"]
        assert (
            updates == math.ceil(training_positions / config.batch_size) and updates > 1
        )
        assert all(
            bool(mx.all(mx.isfinite(v)).item())
            for _, v in tree_flatten(trainer.model.parameters())
        )
        record["attempted_updates"] = updates
        record["candidate_adam_step"] = before + updates
        del games
        gc.collect()
        mx.clear_cache()
        milestone = (done + count) % 500 == 0
        if milestone:
            progress["phase"] = "candidate_evaluation"
            write_json(progress_path, progress)
            trainer.model.save_checkpoint(
                out / "candidates", generation, optimizer_state=trainer.optimizer.state
            )
            record["candidate_vs_random"] = random_report(
                trainer.model, config, out, f"candidate-{done+count:04d}"
            )
        progress["phase"] = "promotion_match"
        write_json(progress_path, progress)
        record["gate"] = trainer.consider_candidate(
            n_workers=32, seed=20000 + generation * 40
        )
        record["accepted_adam_step"] = int(trainer.optimizer.state["step"])
        # Save after the decision: rejection must restore model, BN and Adam.
        trainer._save_checkpoint(generation)
        progress["games_completed"] += count
        progress["generation"] = generation
        progress["attempted_updates"] += updates
        progress["accepted_updates"] = record["accepted_adam_step"]
        if record["gate"]["accepted"]:
            progress["promotions"] += 1
            progress["accepted_generation"] = generation
        else:
            progress["rejections"] += 1
            assert record["accepted_adam_step"] == before
        progress["generations"].append(record)
        write_json(progress_path, progress)
        prune_checkpoints(checkpoints, generation)
        print(
            "GENERATION_COMPLETE",
            json.dumps({k: v for k, v in record.items() if k != "candidate_vs_random"}),
            flush=True,
        )
        if milestone:
            finish_milestone(trainer, progress, out)
    if TOTAL not in progress["milestones"]:
        finish_milestone(trainer, progress, out)
    progress["phase"] = "final_baseline_match"
    write_json(progress_path, progress)
    source_manifest = json.loads(
        (SOURCE / "checkpoints/state_epoch_0.json").read_text()
    )
    baseline = ChessNet(ModelConfig(**source_manifest["config"]))
    baseline.load_weights(str(SOURCE / "checkpoints" / source_manifest["model_file"]))
    openings = confirmation_openings(OPENINGS, 80000, pairs=32)
    write_json(out / "baseline-openings.json", openings)
    direct = evaluate_matches(
        trainer.best_model,
        baseline,
        config,
        openings,
        write_json,
        out,
        "final-vs-initial",
        confirmation=True,
    )
    del baseline
    restored, state = ChessNet.load_checkpoint(checkpoints, progress["generation"])
    assert int(state["optimizer_state"]["step"]) == progress["accepted_updates"]
    for name in ("model", "best_model"):
        comparison = restored if name == "model" else state["best_model"]
        assert all(
            bool(mx.array_equal(a, b).item())
            for (_, a), (_, b) in zip(
                tree_flatten(comparison.parameters()),
                tree_flatten(trainer.best_model.parameters()),
            )
        )
    for (_, a), (_, b) in zip(
        tree_flatten(state["optimizer_state"]), tree_flatten(trainer.optimizer.state)
    ):
        assert bool(mx.array_equal(a, b).item())
    progress["phase"] = "completed"
    write_json(progress_path, progress)
    write_json(
        out / "result.json",
        dict(
            status="completed",
            games=TOTAL,
            attempted_updates=progress["attempted_updates"],
            accepted_updates=progress["accepted_updates"],
            promotions=progress["promotions"],
            rejections=progress["rejections"],
            checkpoint_roundtrip=True,
            final_milestone=json.loads((out / "milestone-3000.json").read_text()),
            direct_vs_initial=direct,
        ),
    )
    print("RUN_COMPLETE", flush=True)


def finish_milestone(trainer, progress, out):
    target = progress["games_completed"]
    progress["phase"] = "accepted_evaluation"
    write_json(out / "progress.json", progress)
    record = progress["generations"][-1]
    accepted = record.get("accepted_vs_random")
    if accepted is None:
        accepted = (
            record["candidate_vs_random"]
            if record["gate"]["accepted"]
            else random_report(
                trainer.best_model, trainer.config, out, f"accepted-{target:04d}"
            )
        )
        record["accepted_vs_random"] = accepted
    report = dict(
        games_completed=target,
        attempted_updates=progress["attempted_updates"],
        accepted_updates=progress["accepted_updates"],
        promotions=progress["promotions"],
        rejections=progress["rejections"],
        accepted_generation=progress["accepted_generation"],
        training_positions=sum(g["positions"] for g in progress["generations"]),
        training_draws=sum(g["draws"] for g in progress["generations"]),
        candidate_vs_random=record["candidate_vs_random"],
        accepted_vs_random=accepted,
        latest_gate=record["gate"],
    )
    write_json(out / f"milestone-{target:04d}.json", report)
    if target not in progress["milestones"]:
        progress["milestones"].append(target)
    write_json(out / "progress.json", progress)
    print("MILESTONE", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
