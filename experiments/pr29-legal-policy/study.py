"""Matched first-generation experiments; all candidates and data are retained."""

import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, "/opt/chesszero/chesszero")
from run_training import OPENINGS, write_json
from matches import confirmation_openings, evaluate

SOURCE = Path("/results/pr29-random-restart-20261015-085706/checkpoints")
ARMS = ("legal_policy",)


def objective(model, states, policies, values, kind, legal=None):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    logits, predicted = model.forward_logits(states)
    if legal is not None:
        logits = mx.where(legal, logits, mx.array(-1e9, dtype=logits.dtype))
    if kind == "legacy_policy_loss":
        logits = mx.softmax(logits, axis=-1)
    policy = nn.losses.cross_entropy(logits, policies, reduction="mean")
    value = nn.losses.mse_loss(predicted.reshape(-1), values, reduction="mean")
    regularization = sum(
        1e-6 * mx.sum(p * p)
        for _, p in tree_flatten(model.trainable_parameters())
        if len(p.shape) > 1
    )
    total = (
        (0 if kind == "value_only" else policy)
        + (0 if kind == "policy_only" else value)
        + regularization
    )
    return total, (policy, value, regularization)


def arrays_from_games(rows):
    states, policies, targets = [], [], []
    for row in rows:
        for state, policy, _ in row["history"]:
            states.append(state)
            policies.append(policy)
            targets.append(
                row["white_result"] if state[12, 0, 0] == 1 else -row["white_result"]
            )
    return np.stack(states), np.stack(policies), np.asarray(targets, dtype=np.float32)


def batches(arrays, shuffled=False):
    import mlx.core as mx

    indices = np.arange(len(arrays[0]))
    if shuffled:
        np.random.default_rng(62016).shuffle(indices)
    for start in range(0, len(indices), 64):
        subset = indices[start : start + 64]
        yield tuple(mx.array(a[subset]) for a in arrays)


def fingerprint(model):
    from mlx.utils import tree_flatten

    digest = hashlib.sha256()
    for key, value in tree_flatten(model.parameters()):
        digest.update(key.encode())
        digest.update(np.asarray(value).tobytes())
    return digest.hexdigest()


def probe(model, arrays):
    import mlx.nn as nn
    from model.inference_mode import search_inference_mode

    values, labels, sides, policy_losses = [], [], [], []
    before = fingerprint(model)
    with search_inference_mode(model, True, per_position=True):
        for states, policies, target in batches(arrays):
            logits, value = model.forward_logits(states)
            policy_losses.extend(
                np.asarray(
                    nn.losses.cross_entropy(logits, policies, reduction="none")
                ).tolist()
            )
            values.extend(np.asarray(value).reshape(-1).tolist())
            labels.extend(np.asarray(target).tolist())
            sides.extend(np.asarray(states[:, 12, 0, 0]).tolist())
    assert fingerprint(model) == before
    values, labels, sides = map(np.asarray, (values, labels, sides))
    return dict(
        policy_ce=float(np.mean(policy_losses)),
        value_mse=float(np.mean((values - labels) ** 2)),
        by_color={
            str(color): dict(
                count=int(np.sum(sides == white)),
                mean_value=float(np.mean(values[sides == white])),
                mean_target=float(np.mean(labels[sides == white])),
                mse=float(
                    np.mean((values[sides == white] - labels[sides == white]) ** 2)
                ),
            )
            for color, white in ((0, 1), (1, 0))
        },
    )


def main():
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from model.network import ChessNet
    from model.training import Trainer
    from config.model_config import ModelConfig

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    mx.set_default_device(mx.gpu)
    mx.random.seed(62016)
    out = Path(sys.argv[1])
    out.mkdir(exist_ok=True)
    baseline, source_state = ChessNet.load_checkpoint(SOURCE, 0)
    assert int(source_state["optimizer_state"]["step"]) == 0
    config = baseline.config
    config.search_batch_norm = True
    config.search_batch_norm_per_position = True
    config.inference_pad_batches = True
    config.inference_batch_size = 16
    config.n_simulations = 64
    config.batch_size = 64
    config.learning_rate = 1e-4
    config.accumulate_gradients = False
    write_json(
        out / "protocol.json",
        dict(
            source_revision=os.environ["CHESSZERO_SOURCE_REVISION"],
            source=str(SOURCE),
            config=vars(config),
            arms=list(ARMS),
            arm_learning_rates={"legal_policy": 1e-4},
            training_games=32,
            validation_games=8,
            match_games=32,
            data_seed_generation=62016,
            discovery_opening_seed=162016,
            confirmation_opening_seed=92016,
            baseline_fingerprint=fingerprint(baseline),
            note="One-factor legal-action CE intervention versus saved batch_training step78 control on the identical seed162016 discovery bank. Same initial model, data, order, 78 Adam updates at1e-4, value and L2 losses, training/search normalization. Exact legal masks from independently replayed saved UCI histories; targets unchanged. Hypothesis only, no production change.",
        ),
    )
    progress = dict(phase="generating_fixed_data", results={})
    write_json(out / "progress.json", progress)
    fixed = Path("/results/pr29-controlled-learning-20261016-113623")

    def load_arrays(name):
        with np.load(fixed / (name + ".npz")) as data:
            return tuple(data[k] for k in ("states", "policies", "values"))

    train, validation = load_arrays("train"), load_arrays("validation")
    from legal_masks import build_masks

    rows = __import__("json").loads((fixed / "training-games-0001.json").read_text())
    masks = build_masks(rows[:32], train)
    validation_masks = build_masks(rows[32:], validation)
    np.savez_compressed(
        out / "legal-masks.npz", train=masks, validation=validation_masks
    )
    write_json(
        out / "mask-verification.json",
        dict(
            train_positions=len(masks),
            validation_positions=len(validation_masks),
            all_states_labels_targets_verified=True,
            mask_sha256=hashlib.sha256(masks.tobytes()).hexdigest(),
        ),
    )
    statistics = dict(source=str(fixed), reused_data=True)
    progress.update(
        phase="training_candidates",
        data=dict(
            train_positions=len(train[0]),
            validation_positions=len(validation[0]),
            inference=statistics,
        ),
        initial_probe=probe(baseline, validation),
    )
    write_json(out / "progress.json", progress)

    class StudyTrainer(Trainer):
        def _loss_fn(self, states, policies, values, legal):
            return objective(self.model, states, policies, values, self.kind, legal)

    # Train every arm before playing any matches; no branch is selected using
    # the validation outcomes. Save rejected/weak candidates as well.
    for kind in ARMS:
        trainer = StudyTrainer(ModelConfig(), out / kind, resume_epoch=None)
        trainer.model, state = ChessNet.load_checkpoint(SOURCE, 0)
        rate = 1e-4
        trainer.config = config
        trainer.kind = kind
        trainer.optimizer = optim.Adam(learning_rate=rate, bias_correction=False)
        trainer.optimizer.state = state["optimizer_state"]
        trainer.optimizer.learning_rate = rate
        assert np.isclose(
            float(trainer.optimizer.learning_rate), rate, rtol=1e-6, atol=0
        )
        trainer._loss_and_grad = nn.value_and_grad(trainer.model, trainer._loss_fn)
        assert fingerprint(trainer.model) == fingerprint(baseline)
        trace = []
        for step, batch in enumerate(batches((*train, masks)), 1):
            assert np.isclose(
                float(trainer.optimizer.learning_rate), rate, rtol=1e-6, atol=0
            )
            losses = trainer.train_on_batch(batch)
            if step in (1, 8, 32) or step == (len(train[0]) + 63) // 64:
                trainer._save_checkpoint(step)
                trace.append(
                    dict(
                        step=step,
                        losses=losses,
                        validation=probe(trainer.model, validation),
                    )
                )
                progress["active_arm"] = kind
                progress["active_step"] = step
                write_json(out / "progress.json", progress)
        progress["results"][kind] = dict(
            updates=step,
            learning_rate=rate,
            trace=trace,
            fingerprint=fingerprint(trainer.model),
        )
        write_json(out / "progress.json", progress)
        print("ARM_TRAINED", kind, json.dumps(progress["results"][kind]), flush=True)
        del trainer, state
        gc.collect()
        mx.clear_cache()

    openings = confirmation_openings(OPENINGS, 162016, pairs=16)
    write_json(out / "discovery-openings.json", openings)
    progress["phase"] = "discovery_matches"
    for kind in ARMS:
        progress["active_arm"] = kind
        write_json(out / "progress.json", progress)
        if kind == "initial_control":
            candidate, _ = ChessNet.load_checkpoint(SOURCE, 0)
        else:
            source = "joint_ordered" if kind == "joint_zero_value" else kind
            candidate, _ = ChessNet.load_checkpoint(
                out / source, progress["results"][source]["updates"]
            )
        if kind == "joint_zero_value":

            class ZeroValue(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def __call__(self, states):
                    policy, value = self.model(states)
                    return policy, mx.zeros_like(value)

            candidate = ZeroValue(candidate)
        before = fingerprint(candidate), fingerprint(baseline)
        match = evaluate(
            candidate, baseline, config, openings, write_json, out, kind, workers=16
        )
        assert before == (fingerprint(candidate), fingerprint(baseline))
        progress["results"].setdefault(kind, {})["match"] = match
        write_json(out / "progress.json", progress)
        del candidate
        gc.collect()
        mx.clear_cache()
    progress["phase"] = "completed"
    write_json(out / "progress.json", progress)
    write_json(out / "result.json", dict(status="completed", **progress))


if __name__ == "__main__":
    main()
