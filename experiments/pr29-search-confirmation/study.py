"""Fresh-bank confirmation of search budgets with neural and zero values."""

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
    openings = confirmation_openings(OPENINGS, 222016, pairs=64)
    prior = json.loads(
        Path("/results/pr29-search-budget-20261017-102401/openings.json").read_text()
    )
    assert not set(openings).intersection(prior)
    write_json(out / "openings.json", openings)
    progress = dict(phase="confirmation", results={})
    for arm in ["neural", "zero"]:
        candidate, state = ChessNet.load_checkpoint(SOURCE, 0)
        baseline, _ = ChessNet.load_checkpoint(SOURCE, 0)
        assert int(state["optimizer_state"]["step"]) == 0
        raw_candidate, raw_baseline = candidate, baseline
        before = fingerprint(candidate), fingerprint(baseline)
        assert before[0] == before[1]
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

        if arm == "zero":
            candidate, baseline = ZeroValue(candidate), ZeroValue(baseline)
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
            out / "protocol.json",
            dict(
                source_revision=os.environ["CHESSZERO_SOURCE_REVISION"],
                arms=["neural", "zero"],
                seed=222016,
                games_per_arm=128,
                optimizer_updates=0,
                config=vars(config),
                candidate_cap=256,
                baseline_cap=64,
                earlystop_unchanged=True,
                alpha_per_arm=0.025,
                paired_contrast="zero-minus-neural budget score;16opening-family bootstrap95%",
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
            initial_fingerprint=before[0],
            candidate_simulations=sum(r["candidate_simulations"] for r in rows),
            baseline_simulations=sum(r["baseline_simulations"] for r in rows),
        )
        write_json(out / "progress.json", progress)
    games = {
        arm: {
            (r["pair_id"], r["color"]): r
            for r in json.loads((out / (arm + "-games.json")).read_text())
        }
        for arm in ["neural", "zero"]
    }
    families = [[] for _ in range(16)]
    assert games["neural"].keys() == games["zero"].keys()
    for key, r in games["zero"].items():
        other = games["neural"][key]
        assert r["opening"] == other["opening"]
        families[key[0] % 16].append((r["score"] or 0) - (other["score"] or 0))
    means = np.array([np.mean(x) for x in families])
    rng = np.random.default_rng(222016)
    boot = means[rng.integers(0, 16, size=(20000, 16))].mean(axis=1)
    progress["paired_comparison"] = dict(
        zero_minus_neural=float(means.mean()),
        family_interval95=np.quantile(boot, [0.025, 0.975]).tolist(),
        note="Relative256vs64 scores under different value conditions, not direct zero-vs-neural strength. Censoring separately.",
    )
    progress["phase"] = "completed"
    write_json(out / "progress.json", progress)
    write_json(out / "result.json", dict(status="completed", **progress))


if __name__ == "__main__":
    main()
