"""Frozen fresh-data checkpoint: remove candidate value only."""

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
    for k, v in tree_flatten(model.parameters()):
        h.update(k.encode())
        h.update(np.asarray(v).tobytes())
    return h.hexdigest()


def main():
    import mlx.core as mx
    import mlx.nn as nn
    from model.network import ChessNet
    from model.inference_mode import search_inference_mode, predict_search_batch

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    mx.set_default_device(mx.gpu)
    out = Path(sys.argv[1])
    source_run = Path("/results/pr29-data-refresh-20261017-080116")
    openings = confirmation_openings(OPENINGS, 242016, pairs=64)
    prior = json.loads((source_run / "discovery-openings.json").read_text())
    assert not set(openings).intersection(prior)
    source_result = json.loads((source_run / "result.json").read_text())["results"][
        "fresh_data"
    ]
    write_json(out / "openings.json", openings)
    progress = dict(phase="confirmation", results={})
    for arm in ["unchanged", "fresh_value_zero"]:
        candidate, state = ChessNet.load_checkpoint(source_run / "fresh_data", 78)
        baseline, _ = ChessNet.load_checkpoint(SOURCE, 0)
        assert int(state["optimizer_state"]["step"]) == 78
        assert np.isclose(
            float(state["optimizer_state"]["learning_rate"]), 1e-4, rtol=1e-6, atol=0
        )
        raw_candidate, raw_baseline = candidate, baseline
        before = fingerprint(candidate), fingerprint(baseline)
        assert before[0] == source_result["fingerprint"]
        assert before[0] != before[1]
        config = baseline.config
        config.n_simulations = 64
        config.search_batch_norm = True
        config.search_batch_norm_per_position = True
        config.inference_pad_batches = True
        config.inference_batch_size = 16

        class ZeroValue(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def __call__(self, states):
                policy, value = self.model(states)
                return policy, mx.zeros_like(value)

        if arm == "fresh_value_zero":
            candidate = ZeroValue(candidate)
            with np.load(
                "/results/pr29-controlled-learning-20261016-113623/validation.npz"
            ) as data:
                positions = mx.array(data["states"][:16])
            with search_inference_mode(candidate, True, per_position=True):
                p, v = predict_search_batch(candidate, positions, config)
                original, _ = predict_search_batch(raw_candidate, positions, config)
                np.testing.assert_array_equal(np.asarray(p), np.asarray(original))
                np.testing.assert_array_equal(np.asarray(v), 0)
        write_json(
            out / (arm + "-verification.json"),
            dict(
                policy_exact=True if arm == "fresh_value_zero" else None,
                candidate_value_zero=arm == "fresh_value_zero",
                candidate_step=78,
                effective_learning_rate=1e-4,
                source_fingerprint_verified=True,
                baseline_neural=True,
            ),
        )
        write_json(
            out / "protocol.json",
            dict(
                source_revision=os.environ["CHESSZERO_SOURCE_REVISION"],
                arms=["unchanged", "fresh_value_zero"],
                seed=242016,
                games_per_arm=128,
                optimizer_updates=0,
                config=vars(config),
                candidate_cap=64,
                baseline_cap=64,
                earlystop_unchanged=True,
                alpha_per_arm=0.025,
                contrast="fresh-data step78 unchanged and zero-value versions versus initial on fresh paired bank",
            ),
        )
        progress["active_arm"] = arm
        write_json(out / "progress.json", progress)
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
        assert before == (fingerprint(raw_candidate), fingerprint(raw_baseline))
        rows = json.loads((out / (arm + "-games.json")).read_text())
        progress["results"][arm] = dict(
            match=match,
            checkpoint_verified=True,
            candidate_fingerprint=before[0],
            baseline_fingerprint=before[1],
            candidate_simulations=sum(r["candidate_simulations"] for r in rows),
            baseline_simulations=sum(r["baseline_simulations"] for r in rows),
        )
        write_json(out / "progress.json", progress)
    rows = {
        a: {
            (r["pair_id"], r["color"]): r
            for r in json.loads((out / (a + "-games.json")).read_text())
        }
        for a in ["unchanged", "fresh_value_zero"]
    }
    assert rows["unchanged"].keys() == rows["fresh_value_zero"].keys()
    contrasts = {}
    for worst_case in [False, True]:
        families = [[] for _ in range(16)]
        for key, zero in rows["fresh_value_zero"].items():
            other = rows["unchanged"][key]
            assert zero["opening"] == other["opening"]
            z = zero["score"] if zero["score"] is not None else 0
            n = other["score"] if other["score"] is not None else int(worst_case)
            families[key[0] % 16].append(z - n)
        means = np.array([np.mean(f) for f in families])
        boot = means[np.random.default_rng(242016).integers(0, 16, (20000, 16))].mean(1)
        contrasts["worst_case" if worst_case else "censored_as_zero"] = dict(
            gain=float(means.mean()),
            interval95=np.quantile(boot, [0.025, 0.975]).tolist(),
        )
    progress["paired_comparison"] = contrasts
    write_json(out / "paired-comparison.json", contrasts)
    progress["phase"] = "completed"
    write_json(out / "result.json", dict(status="completed", **progress))
    write_json(out / "progress.json", progress)


if __name__ == "__main__":
    main()
