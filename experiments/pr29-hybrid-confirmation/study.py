"""Frozen fresh-bank confirmation of accepted hybrid, no optimization."""

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

INITIAL = Path("/results/pr29-random-restart-20261015-085706/checkpoints")
START = Path("/results/pr29-controlled-learning-20261016-113623/joint_ordered")
CONTINUATION = Path("/results/pr29-policy-continuation-20261016-134346")


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
    starting, _ = ChessNet.load_checkpoint(START, 78)

    class Hybrid(nn.Module):
        def __init__(self, learned):
            super().__init__()
            self.learned = learned
            self.initial, _ = ChessNet.load_checkpoint(INITIAL, 0)

        def __call__(self, states):
            policy, _ = self.initial(states)
            _, value = self.learned(states)
            return policy, value

    hybrid = Hybrid(candidate)
    start_hybrid = Hybrid(starting)
    config = baseline.config
    config.n_simulations = 64
    config.search_batch_norm = True
    config.search_batch_norm_per_position = True
    config.inference_pad_batches = True
    config.inference_batch_size = 16
    excluded = set(json.loads((CONTINUATION / "heldout-openings.json").read_text()))
    excluded.update(json.loads((START.parent / "discovery-openings.json").read_text()))
    for path in CONTINUATION.glob("gate-openings-*.json"):
        excluded.update(json.loads(path.read_text()))
    openings = confirmation_openings(OPENINGS, 132016, pairs=64)
    assert not excluded.intersection(openings)
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
            vs_starting_games=64,
            purpose="Frozen independent confirmation; no training or promotion using these games. Two contrasts; alpha .025 each. Opening base families clustered for uncertainty.",
            candidate_fingerprint=fingerprint(candidate),
            starting_fingerprint=fingerprint(starting),
        ),
    )
    progress = dict(phase="confirmation", results={})
    for name, opponent, bank in [
        ("vs_initial", baseline, openings),
        ("vs_starting_hybrid", start_hybrid, openings[:32]),
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
