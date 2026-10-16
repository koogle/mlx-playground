"""Frozen head substitution and policy flattening; no learning."""

import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, "/opt/chesszero/chesszero")
from run_training import write_json
from matches import evaluate

SOURCE = Path("/results/pr29-controlled-learning-20261016-113623")
INITIAL = Path("/results/pr29-random-restart-20261015-085706/checkpoints")


def fingerprint(model):
    from mlx.utils import tree_flatten

    digest = hashlib.sha256()
    for name, value in tree_flatten(model.parameters()):
        digest.update(name.encode())
        digest.update(np.asarray(value).tobytes())
    return digest.hexdigest()


def normalization_check(model, positions, config):
    import mlx.core as mx
    from model.inference_mode import search_inference_mode, predict_search_batch

    before = fingerprint(model)
    x = mx.array(positions)
    with search_inference_mode(model, True):
        serial = [model(x[i : i + 1]) for i in range(len(x))]
        mixed = model(x)
    expected = [np.concatenate([np.asarray(row[h]) for row in serial]) for h in (0, 1)]
    with search_inference_mode(model, True, per_position=True):
        independent = model(x)
        reversed_batch = model(x[::-1])
        padded = predict_search_batch(model, x, config)
        padded_reversed = predict_search_batch(model, x[::-1], config)
        padded_serial = [
            predict_search_batch(model, x[i : i + 1], config) for i in range(len(x))
        ]
    fixed_expected = [
        np.concatenate([np.asarray(row[h]) for row in padded_serial]) for h in (0, 1)
    ]
    errors = {}
    for head, label in enumerate(("policy", "value")):
        errors[label] = dict(
            original_batch_max_error=float(
                np.max(np.abs(np.asarray(mixed[head]) - expected[head]))
            ),
            independent_max_error=float(
                np.max(np.abs(np.asarray(independent[head]) - expected[head]))
            ),
        )
        errors[label]["fixed_shape_max_error"] = float(
            np.max(np.abs(np.asarray(padded[head]) - fixed_expected[head]))
        )
        errors[label]["reordering_max_error"] = float(
            np.max(
                np.abs(
                    np.asarray(reversed_batch[head])
                    - np.asarray(independent[head])[::-1]
                )
            )
        )
        np.testing.assert_allclose(
            padded[head], fixed_expected[head], atol=2e-5, rtol=2e-5
        )
        np.testing.assert_allclose(
            padded_reversed[head], fixed_expected[head][::-1], atol=2e-5, rtol=2e-5
        )
    assert fingerprint(model) == before
    return errors


def main():
    import mlx.core as mx
    import mlx.nn as nn
    from model.network import ChessNet

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    mx.set_default_device(mx.gpu)
    out = Path(sys.argv[1])
    source = json.loads((SOURCE / "progress.json").read_text())
    openings = json.loads((SOURCE / "discovery-openings.json").read_text())
    with np.load(SOURCE / "validation.npz") as data:
        positions = data["states"][
            np.linspace(0, len(data["states"]) - 1, 16, dtype=int)
        ]
    baseline, _ = ChessNet.load_checkpoint(INITIAL, 0)
    config = baseline.config
    config.n_simulations = 64
    config.search_batch_norm = True
    config.search_batch_norm_per_position = True
    config.inference_pad_batches = True
    config.inference_batch_size = 16
    write_json(
        out / "protocol.json",
        dict(
            source_revision=os.environ["CHESSZERO_SOURCE_REVISION"],
            source_study=SOURCE.name,
            config=vars(config),
            optimizer_updates=0,
            purpose="Same frozen checkpoints and discovery openings. Substitute initial policy or initial value into joint model; flatten policy-only probabilities at temperature4. All other search settings identical to completed independent-normalization study. Discovery mechanism tests, not confirmation of strength.",
        ),
    )
    progress = dict(phase="frozen_policy_mechanism", results={})
    kinds = ("joint_initial_policy", "joint_initial_value", "policy_temperature4")
    for kind in kinds:
        progress["active_arm"] = kind
        write_json(out / "progress.json", progress)
        arm = "policy_only" if kind == "policy_temperature4" else "joint_ordered"
        trained, _ = ChessNet.load_checkpoint(
            SOURCE / arm, source["results"][arm]["updates"]
        )
        initial, _ = ChessNet.load_checkpoint(INITIAL, 0)
        assert fingerprint(trained) == source["results"][arm]["fingerprint"]
        assert fingerprint(initial) == fingerprint(baseline)
        checks = normalization_check(trained, positions, config)

        class Intervention(nn.Module):
            def __init__(self, trained, initial, kind):
                super().__init__()
                self.trained = trained
                self.initial = initial
                self.kind = kind

            def __call__(self, states):
                policies, values = self.trained(states)
                if self.kind == "policy_temperature4":
                    policies = mx.power(mx.maximum(policies, 1e-30), 0.25)
                    return policies / mx.sum(policies, axis=-1, keepdims=True), values
                initial_policy, initial_value = self.initial(states)
                if self.kind == "joint_initial_policy":
                    return initial_policy, values
                return policies, initial_value

        candidate = Intervention(trained, initial, kind)
        before = fingerprint(candidate), fingerprint(baseline)
        match = evaluate(
            candidate, baseline, config, openings, write_json, out, kind, workers=16
        )
        assert before == (fingerprint(candidate), fingerprint(baseline))
        rows = json.loads((out / f"{kind}-games.json").read_text())
        paired_identical = sum(
            rows[2 * i]["moves"] == rows[2 * i + 1]["moves"]
            for i in range(len(openings))
        )
        progress["results"][kind] = dict(
            normalization=checks, match=match, paired_identical_games=paired_identical
        )
        write_json(out / "progress.json", progress)
        del candidate
        gc.collect()
        mx.clear_cache()
    progress["phase"] = "completed"
    write_json(out / "progress.json", progress)
    write_json(out / "result.json", dict(status="completed", **progress))


if __name__ == "__main__":
    main()
