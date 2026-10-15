"""Stats-only normalization intervention using fresh self-play, never eval data."""

import gc
import hashlib
import json
import logging
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, "/opt/chesszero/chesszero")
from run_training import evaluate, training_worker, write_json
from restart_training import color_summary
from ablate import mirror_states
from model.pipe_inference import run_pipe_games

SOURCE = Path("/results/pr29-first-result-20261014-202746")


def learned_fingerprint(model):
    from mlx.utils import tree_flatten

    digest = hashlib.sha256()
    for key, value in tree_flatten(model.trainable_parameters()):
        digest.update(key.encode())
        digest.update(np.asarray(value).tobytes())
    return digest.hexdigest()


def calibration_batches(states, kind):
    rng = np.random.default_rng(8801)
    order = (
        rng.permutation(len(states)) if kind != "ordered" else np.arange(len(states))
    )
    width = 32 if kind == "mirror_balanced" else 64
    for start in range(0, len(states), width):
        batch = states[order[start : start + width]]
        if kind == "mirror_balanced":
            batch = np.concatenate((batch, mirror_states(batch)))
        yield batch


def main():
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from model.network import ChessNet

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    mx.set_default_device(mx.gpu)
    out = Path(sys.argv[1])
    model, checkpoint = ChessNet.load_checkpoint(SOURCE / "checkpoints", 10)
    del checkpoint
    model.eval()
    original = dict(tree_flatten(model.parameters()))
    fingerprint = learned_fingerprint(model)
    config = model.config
    print("FRESH_CALIBRATION_SELF_PLAY_START", flush=True)
    rows, failures, stats = run_pipe_games(
        model, config, training_worker, [(config, i, 9000) for i in range(16)], 16
    )
    if failures or len(rows) != 16:
        raise RuntimeError("Incomplete fresh calibration self-play")
    states = np.stack([state for row in rows for state, _, _ in row["history"]])
    np.savez_compressed(out / "calibration_states.npz", states=states)
    data = dict(
        games=len(rows),
        positions=len(states),
        failures=failures,
        white_wins=sum(r["white_result"] == 1 for r in rows),
        black_wins=sum(r["white_result"] == -1 for r in rows),
        draws=sum(r["white_result"] == 0 for r in rows),
        inference=stats,
        seed_protocol="1000000 + 9000*1000 + game_id; games 0..15",
        source="New complete 64-simulation self-play, no evaluation positions or outcomes used for calibration",
    )
    write_json(out / "calibration_data.json", data)
    del rows
    gc.collect()
    write_json(
        out / "protocol.json",
        dict(
            source_run=SOURCE.name,
            source_epoch=10,
            optimizer_updates=0,
            learned_weights_fixed=True,
            interventions=["ordered", "shuffled", "mirror_balanced"],
            evaluation="Same first 16 color pairs as original random-opponent evaluation, discovery only",
            rationale="All variants restore the identical checkpoint first; only BatchNorm running statistics change. "
            "Ordered/shuffled use each identical fresh position once at batch64. Mirror-balanced adds "
            "one transformed position per original and uses 32 originals plus32 mirrors per batch.",
        ),
    )
    result = {"calibration_data": data}
    for kind in ("ordered", "shuffled", "mirror_balanced"):
        model.load_weights(list(original.items()))
        model.train()
        count = 0
        for batch in calibration_batches(states, kind):
            policies, values = model(mx.array(batch))
            mx.eval(policies, values, model.parameters())
            count += len(batch)
        model.eval()
        assert learned_fingerprint(model) == fingerprint
        assert any(
            not bool(mx.array_equal(value, original[key]).item())
            for key, value in tree_flatten(model.parameters())
            if key.endswith(("running_mean", "running_var"))
        )
        print("STATS_REFRESHED", kind, count, flush=True)
        evaluate(model, config, out, "calibrated_" + kind, 16, 10000)
        scores = json.loads((out / ("calibrated_" + kind + "-games.json")).read_text())
        assert learned_fingerprint(model) == fingerprint
        result[kind] = dict(
            by_color=color_summary(scores),
            positions_seen=count,
            learned_weights_unchanged=True,
            optimizer_updates=0,
        )
        write_json(out / "progress.json", result)
        print("CALIBRATION_RESULT", kind, json.dumps(result[kind]), flush=True)
        mx.clear_cache()
    write_json(out / "result.json", dict(status="completed", report=result))


if __name__ == "__main__":
    main()
