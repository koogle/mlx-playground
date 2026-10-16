"""Frozen early-checkpoint strength trajectory; no optimizer updates."""

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

SOURCE = Path("/results/pr29-training-normalization-20261016-210929")
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
    openings = json.loads((SOURCE / "discovery-openings.json").read_text())
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
            steps=[1, 8, 32],
            games_per_checkpoint=32,
            config=vars(config),
            final_step_reference=source["results"]["batch_training"]["match"],
            purpose="Locate onset of regression along saved standard-training trajectory using identical32game bank; checkpoints not selected for promotion. No retraining.",
        ),
    )
    progress = dict(phase="checkpoint_trajectory", results={})
    for step in [1, 8, 32]:
        progress["active_step"] = step
        write_json(out / "progress.json", progress)
        candidate, state = ChessNet.load_checkpoint(SOURCE / "batch_training", step)
        assert int(state["optimizer_state"]["step"]) == step
        before = fingerprint(candidate), fingerprint(baseline)
        match = evaluate(
            candidate,
            baseline,
            config,
            openings,
            write_json,
            out,
            f"step-{step}",
            workers=16,
        )
        assert before == (fingerprint(candidate), fingerprint(baseline))
        progress["results"][str(step)] = dict(
            match=match, checkpoint_verified=True, fingerprint=before[0]
        )
        write_json(out / "progress.json", progress)
    progress["phase"] = "completed"
    write_json(out / "progress.json", progress)
    write_json(out / "result.json", dict(status="completed", **progress))


if __name__ == "__main__":
    main()
