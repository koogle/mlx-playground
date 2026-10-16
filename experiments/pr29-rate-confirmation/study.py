"""Frozen early-checkpoint strength trajectory; no optimizer updates."""

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

SOURCE = Path("/results/pr29-learning-rate-20261016-232717")
INITIAL = Path("/results/pr29-random-restart-20261015-085706/checkpoints")


def fingerprint(model):
    from mlx.utils import tree_flatten

    h = hashlib.sha256()
    for k, v in tree_flatten(model.parameters()):
        h.update(k.encode())
        h.update(np.asarray(v).tobytes())
    return h.hexdigest()


def main():
    import mlx.core as mx
    from model.network import ChessNet

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    mx.set_default_device(mx.gpu)
    out = Path(sys.argv[1])
    source = json.loads((SOURCE / "result.json").read_text())
    assert source["status"] == "completed"
    openings = confirmation_openings(OPENINGS, 172016, pairs=64)
    assert not set(openings).intersection(
        json.loads((SOURCE / "discovery-openings.json").read_text())
    )
    write_json(out / "openings.json", openings)
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
            source=str(SOURCE),
            optimizer_updates=0,
            arms=["lr_1e5", "lr_1e6"],
            seed=172016,
            games_per_checkpoint=128,
            config=vars(config),
            discovery_reference={k: r["match"] for k, r in source["results"].items()},
            purpose="Frozen fresh-bank confirmation of two preselected learning-rate candidates vs initial;128games each on identical openings. No updates. Alpha.025 each; family-clustered bootstrap. No model selection or training on this bank.",
        ),
    )
    progress = dict(phase="rate_confirmation", results={})
    for arm in ["lr_1e5", "lr_1e6"]:
        step = 78
        progress["active_arm"] = arm
        write_json(out / "progress.json", progress)
        candidate, state = ChessNet.load_checkpoint(SOURCE / arm, step)
        assert int(state["optimizer_state"]["step"]) == step
        assert fingerprint(candidate) == source["results"][arm]["fingerprint"]
        effective_lr = float(state["optimizer_state"]["learning_rate"])
        assert np.isclose(
            effective_lr, source["results"][arm]["learning_rate"], rtol=1e-6, atol=0
        )
        before = fingerprint(candidate), fingerprint(baseline)
        match = evaluate(
            candidate,
            baseline,
            config,
            openings,
            write_json,
            out,
            arm,
            workers=16,
            alpha=0.025,
            confirmation=True,
        )
        assert before == (fingerprint(candidate), fingerprint(baseline))
        progress["results"][arm] = dict(
            match=match,
            checkpoint_verified=True,
            fingerprint=before[0],
            effective_learning_rate=effective_lr,
        )
        write_json(out / "progress.json", progress)
    progress["phase"] = "completed"
    write_json(out / "progress.json", progress)
    write_json(out / "result.json", dict(status="completed", **progress))


if __name__ == "__main__":
    main()
