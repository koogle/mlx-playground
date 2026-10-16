"""Frozen fresh-bank confirmation of accepted hybrid, no optimization."""

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

INITIAL = Path("/results/pr29-random-restart-20261015-085706/checkpoints")
START = Path("/results/pr29-controlled-learning-20261016-113623/joint_ordered")
CONTINUATION = Path("/results/pr29-policy-continuation-20261016-134346")


def mirror_states(states):
    channels = [*range(6, 12), *range(6), 12, 15, 16, 13, 14, 17, 18]
    out = np.take(np.asarray(states)[..., ::-1, :], channels, axis=-3)
    out[..., 12, :, :] = 1 - out[..., 12, :, :]
    return out


def fingerprint(model):
    from mlx.utils import tree_flatten

    digest = hashlib.sha256()
    for key, value in tree_flatten(model.parameters()):
        digest.update(key.encode())
        digest.update(np.asarray(value).tobytes())
    return digest.hexdigest()


def main():
    import mlx.core as mx
    import mlx.nn as nn
    from model.network import ChessNet

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    mx.set_default_device(mx.gpu)
    out = Path(sys.argv[1])
    source = json.loads((CONTINUATION / "result.json").read_text())
    assert source["status"] == "completed" and source["accepted_step"] == 337
    candidate, _ = ChessNet.load_checkpoint(CONTINUATION / "candidates", 337)
    assert fingerprint(candidate) == source["accepted_fingerprint"]
    baseline, _ = ChessNet.load_checkpoint(INITIAL, 0)

    class Hybrid(nn.Module):
        def __init__(self, learned):
            super().__init__()
            self.learned = learned
            self.initial, _ = ChessNet.load_checkpoint(INITIAL, 0)

        def __call__(self, states):
            policy, _ = self.initial(states)
            _, value = self.learned(states)
            _, mirrored_value = self.learned(mx.array(mirror_states(states)))
            return policy, (value + mirrored_value) / 2

    hybrid = Hybrid(candidate)
    config = baseline.config
    config.n_simulations = 64
    config.search_batch_norm = True
    config.search_batch_norm_per_position = True
    config.inference_pad_batches = True
    config.inference_batch_size = 16
    reference = Path("/results/pr29-hybrid-confirmation-20261016-170452")
    openings = json.loads((reference / "openings.json").read_text())
    reference_result = json.loads((reference / "result.json").read_text())
    assert reference_result["status"] == "completed"
    from model.inference_mode import search_inference_mode, predict_search_batch

    with np.load(CONTINUATION / "fresh-1.npz") as data:
        states = data["states"][np.linspace(0, len(data["states"]) - 1, 16, dtype=int)]
    np.testing.assert_array_equal(mirror_states(mirror_states(states)), states)
    before = fingerprint(hybrid)
    with search_inference_mode(hybrid, True, per_position=True):
        policy, value = predict_search_batch(hybrid, mx.array(states), config)
        _, mirror_value = predict_search_batch(
            hybrid, mx.array(mirror_states(states)), config
        )
        expected_policy, _ = predict_search_batch(
            hybrid.initial, mx.array(states), config
        )
        np.testing.assert_array_equal(policy, expected_policy)
        np.testing.assert_allclose(value, mirror_value, atol=2e-5, rtol=2e-5)
        mirror_error = float(
            np.max(np.abs(np.asarray(value) - np.asarray(mirror_value)))
        )
    assert fingerprint(hybrid) == before
    write_json(
        out / "invariance.json",
        dict(
            mirror_value_max_error=mirror_error,
            policy_unchanged=True,
            checkpoint_verified=True,
        ),
    )
    write_json(out / "openings.json", openings)
    write_json(
        out / "protocol.json",
        dict(
            source_revision=os.environ["CHESSZERO_SOURCE_REVISION"],
            source=str(CONTINUATION),
            step=337,
            config=vars(config),
            optimizer_updates=0,
            seed=132016,
            vs_initial_games=128,
            mirror_value_max_error=mirror_error,
            purpose="Frozen same-bank mechanism intervention: average original and color/rank-mirrored side-to-move neural value only. Initial policy unchanged, exact terminal values unchanged. This reused bank is discovery evidence, not an independent confirmation.",
            candidate_fingerprint=fingerprint(candidate),
            reference_result=reference_result["results"]["vs_initial"],
        ),
    )
    progress = dict(phase="mirror_value_mechanism", results={})
    for name, opponent, bank in [
        ("vs_initial", baseline, openings),
    ]:
        progress["active_arm"] = name
        write_json(out / "progress.json", progress)
        before = (fingerprint(hybrid), fingerprint(opponent))
        result = evaluate(
            hybrid,
            opponent,
            config,
            bank,
            write_json,
            out,
            name,
            workers=16,
            alpha=0.025,
            confirmation=True,
        )
        assert before == (fingerprint(hybrid), fingerprint(opponent))
        progress["results"][name] = dict(match=result, checkpoint_verified=True)
        write_json(out / "progress.json", progress)
    progress["phase"] = "completed"
    write_json(out / "progress.json", progress)
    write_json(out / "result.json", dict(status="completed", **progress))


if __name__ == "__main__":
    main()
