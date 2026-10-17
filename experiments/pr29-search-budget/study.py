"""Frozen identical-network search-budget comparison."""

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


def fingerprint(model):
    from mlx.utils import tree_flatten

    h = hashlib.sha256()
    for key, value in tree_flatten(model.parameters()):
        h.update(key.encode())
        h.update(np.asarray(value).tobytes())
    return h.hexdigest()


def main():
    import mlx.core as mx
    from model.network import ChessNet

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    mx.set_default_device(mx.gpu)
    out = Path(sys.argv[1])
    candidate, state = ChessNet.load_checkpoint(SOURCE, 0)
    baseline, _ = ChessNet.load_checkpoint(SOURCE, 0)
    assert int(state["optimizer_state"]["step"]) == 0
    before = fingerprint(candidate), fingerprint(baseline)
    assert before[0] == before[1]
    config = baseline.config
    config.n_simulations = 64
    config.search_batch_norm = True
    config.search_batch_norm_per_position = True
    config.inference_pad_batches = True
    config.inference_batch_size = 16
    openings = confirmation_openings(OPENINGS, 202016, pairs=16)
    write_json(out / "openings.json", openings)
    write_json(
        out / "protocol.json",
        dict(
            source_revision=os.environ["CHESSZERO_SOURCE_REVISION"],
            config=vars(config),
            candidate_cap=256,
            baseline_cap=64,
            early_stopping_unchanged=True,
            initial_fingerprint=before[0],
            optimizer_updates=0,
            games=32,
            seed=202016,
            purpose="Does a larger search budget improve the initial network before generating more policy targets? Frozen identical weights, paired colors; discovery not training success.",
        ),
    )
    write_json(out / "progress.json", dict(phase="search_budget_match"))
    match = evaluate(
        candidate,
        baseline,
        config,
        openings,
        write_json,
        out,
        "budget256_vs64",
        workers=16,
    )
    assert before == (fingerprint(candidate), fingerprint(baseline))
    rows = json.loads((out / "budget256_vs64-games.json").read_text())
    result = dict(
        status="completed",
        match=match,
        checkpoint_verified=True,
        identical_weights=True,
        optimizer_updates=0,
        candidate_simulations=sum(r["candidate_simulations"] for r in rows),
        baseline_simulations=sum(r["baseline_simulations"] for r in rows),
    )
    write_json(out / "result.json", result)
    write_json(out / "progress.json", result)


if __name__ == "__main__":
    main()
