"""Bounded low-rate continuation with full learned policy/value and accepted teacher."""

import json
import logging
import os
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, "/opt/chesszero/chesszero")
from run_training import OPENINGS, training_worker, write_json
from train_gated import validate_games
from learning import arrays_from_games, batches, fingerprint, objective
from matches import confirmation_openings, evaluate
from model.pipe_inference import run_pipe_games

INITIAL = Path("/results/pr29-random-restart-20261015-085706/checkpoints")
SOURCE = Path("/results/pr29-learning-rate-20261016-232717/lr_1e6")


def main():
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from model.network import ChessNet
    from model.training import Trainer
    from config.model_config import ModelConfig

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    mx.set_default_device(mx.gpu)
    mx.random.seed(102016)
    out = Path(sys.argv[1])
    baseline, _ = ChessNet.load_checkpoint(INITIAL, 0)
    initial_hash = fingerprint(baseline)
    config = baseline.config
    config.n_simulations = 64
    config.search_batch_norm = True
    config.search_batch_norm_per_position = True
    config.inference_pad_batches = True
    config.inference_batch_size = 16
    config.batch_size = 64
    config.learning_rate = 1e-6
    config.accumulate_gradients = False

    class ContinuationTrainer(Trainer):
        def _loss_fn(self, states, policies, values):
            return objective(self.model, states, policies, values, "joint_ordered")

    trainer = ContinuationTrainer(ModelConfig(), out / "candidates", resume_epoch=None)
    trainer.model, state = ChessNet.load_checkpoint(SOURCE, 78)
    trainer.config = config
    trainer.optimizer = optim.Adam(learning_rate=1e-6, bias_correction=False)
    trainer.optimizer.state = state["optimizer_state"]
    trainer.optimizer.learning_rate = 1e-6
    assert np.isclose(float(trainer.optimizer.learning_rate), 1e-6, rtol=1e-6, atol=0)
    trainer._loss_and_grad = nn.value_and_grad(trainer.model, trainer._loss_fn)
    assert int(trainer.optimizer.state["step"]) == 78
    accepted_path, accepted_step = INITIAL, 0
    accepted, _ = ChessNet.load_checkpoint(accepted_path, accepted_step)
    source_hash = fingerprint(trainer.model)
    source_result = json.loads((SOURCE.parent / "result.json").read_text())
    assert source_hash == source_result["results"]["lr_1e6"]["fingerprint"]
    discovery = json.loads((SOURCE.parent / "discovery-openings.json").read_text())
    write_json(
        out / "protocol.json",
        dict(
            source_revision=os.environ["CHESSZERO_SOURCE_REVISION"],
            config=vars(config),
            source=str(SOURCE),
            source_step=78,
            source_fingerprint=source_hash,
            initial_fingerprint=initial_hash,
            fresh_games=96,
            generations=3,
            gate_games=40,
            gate_threshold=0.55,
            final_games_per_model=128,
            final_seed=182016,
            optimizer="continuous Adam, no bias correction",
            policy="Full learned policy and value; accepted teacher initially epoch0, low-rate learner resumes step78. No hybrid or mirror intervention.",
            rejection="teacher stays accepted; continuous learner and optimizer retained to test recovery over subsequent updates",
            recycling="none in this controlled continuation; promotion and held-out games never become training data",
            note="Exploratory continuation, not a proven fix. Final bank excluded from training and promotion. All candidate checkpoints and fresh data retained.",
        ),
    )
    progress = dict(
        phase="self_play",
        fresh_games=0,
        attempted_updates=0,
        accepted_updates=0,
        promotions=0,
        rejections=0,
        generations=[],
    )
    for generation in range(1, 4):
        progress.update(phase="self_play", active_generation=generation)
        write_json(out / "progress.json", progress)
        teacher = accepted
        teacher_hash = fingerprint(teacher)
        rows, failures, inference = run_pipe_games(
            teacher,
            config,
            training_worker,
            [(config, i, 182016 + generation) for i in range(32)],
            16,
        )
        assert not failures and len(rows) == 32
        assert fingerprint(teacher) == teacher_hash
        validate_games(rows, out, generation)
        arrays = arrays_from_games(rows)
        np.savez_compressed(
            out / f"fresh-{generation}.npz",
            states=arrays[0],
            policies=arrays[1],
            values=arrays[2],
        )
        progress["fresh_games"] += 32
        progress["phase"] = "training"
        write_json(out / "progress.json", progress)
        losses = []
        for batch in batches(arrays):
            assert np.isclose(
                float(trainer.optimizer.learning_rate), 1e-6, rtol=1e-6, atol=0
            )
            losses.append(trainer.train_on_batch(batch))
        step = int(trainer.optimizer.state["step"])
        progress["attempted_updates"] = step - 78
        trainer._save_checkpoint(step)
        candidate, _ = ChessNet.load_checkpoint(out / "candidates", step)
        assert fingerprint(candidate) == fingerprint(trainer.model)
        candidate_hash = fingerprint(candidate)

        progress["phase"] = "evaluating"
        write_json(out / "progress.json", progress)
        raw = evaluate(
            candidate,
            baseline,
            config,
            discovery,
            write_json,
            out,
            f"raw-{generation}",
            workers=16,
        )
        gate_openings = confirmation_openings(OPENINGS, 192016 + generation, pairs=20)
        write_json(out / f"gate-openings-{generation}.json", gate_openings)
        gate = evaluate(
            candidate,
            teacher,
            config,
            gate_openings,
            write_json,
            out,
            f"gate-{generation}",
            workers=16,
        )
        assert fingerprint(candidate) == candidate_hash
        assert fingerprint(teacher) == teacher_hash
        assert fingerprint(baseline) == initial_hash
        promote = gate["censored"] == 0 and gate["score"] > 0.55
        if promote:
            accepted_path, accepted_step = out / "candidates", step
            accepted, _ = ChessNet.load_checkpoint(accepted_path, accepted_step)
            progress["promotions"] += 1
            progress["accepted_updates"] = step
        else:
            progress["rejections"] += 1
        progress["generations"].append(
            dict(
                generation=generation,
                step=step,
                fresh_games=32,
                positions=len(arrays[0]),
                losses=losses,
                raw=raw,
                gate=gate,
                promoted=promote,
                fingerprint=candidate_hash,
                checkpoint_verified=True,
                accepted_step=accepted_step,
                accepted_path=str(accepted_path),
            )
        )
        write_json(out / "progress.json", progress)
        print("GENERATION_COMPLETE", generation, "promoted", promote, flush=True)
    progress["phase"] = "held_out_confirmation"
    write_json(out / "progress.json", progress)
    heldout = confirmation_openings(OPENINGS, 182016, pairs=64)
    assert set(heldout).isdisjoint(discovery)
    write_json(out / "heldout-openings.json", heldout)
    starting, _ = ChessNet.load_checkpoint(SOURCE, 78)
    final = {}
    for name, model in [
        ("starting_candidate", starting),
        ("accepted_model", accepted),
    ]:
        if name == "accepted_model" and fingerprint(accepted) == fingerprint(starting):
            final[name] = dict(
                final["starting_candidate"], reused_identical_checkpoint=True
            )
            continue
        before = fingerprint(model)
        final[name] = evaluate(
            model,
            baseline,
            config,
            heldout,
            write_json,
            out,
            name,
            workers=16,
            alpha=0.025,
            confirmation=True,
        )
        assert fingerprint(model) == before and fingerprint(baseline) == initial_hash
    progress.update(
        phase="completed",
        final=final,
        checkpoint_verified=True,
        accepted_fingerprint=fingerprint(accepted),
        accepted_step=accepted_step,
    )
    write_json(out / "progress.json", progress)
    write_json(out / "result.json", dict(status="completed", **progress))


if __name__ == "__main__":
    main()
