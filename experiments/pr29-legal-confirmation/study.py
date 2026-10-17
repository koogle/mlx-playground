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

SOURCE = Path("/results/pr29-legal-policy-20261017-032425")
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
    control = Path("/results/pr29-training-normalization-20261016-210929")
    controls = json.loads((control / "result.json").read_text())
    openings = confirmation_openings(OPENINGS, 192016, pairs=64)
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
            arms=["legal_policy", "batch_training"],
            seed=192016,
            games_per_checkpoint=128,
            config=vars(config),
            discovery_reference={k: r["match"] for k, r in source["results"].items()},
            purpose="Fresh-bank confirmation of legal-action loss versus original global loss: frozen step78 checkpoints, identical128games each vs initial. Only source loss differs. Primary contrast paired16-family bootstrap of score difference; baseline tests alpha.025 each. No updates or selection on this bank.",
        ),
    )
    progress = dict(phase="rate_confirmation", results={})
    for arm in ["legal_policy", "batch_training"]:
        step = 78
        progress["active_arm"] = arm
        write_json(out / "progress.json", progress)
        arm_source = SOURCE if arm == "legal_policy" else control
        evidence = source if arm == "legal_policy" else controls
        candidate, state = ChessNet.load_checkpoint(arm_source / arm, step)
        assert int(state["optimizer_state"]["step"]) == step
        assert fingerprint(candidate) == evidence["results"][arm]["fingerprint"]
        effective_lr = float(state["optimizer_state"]["learning_rate"])
        assert np.isclose(effective_lr, 1e-4, rtol=1e-6, atol=0)
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
    # Same openings and seeds permit a paired contrast, clustered by base family.
    rows = {
        arm: json.loads((out / (arm + "-games.json")).read_text())
        for arm in ("legal_policy", "batch_training")
    }
    indexed = {
        arm: {(r["pair_id"], r["color"]): r for r in games}
        for arm, games in rows.items()
    }
    assert indexed["legal_policy"].keys() == indexed["batch_training"].keys()
    families = [[] for _ in range(16)]
    for key, row in indexed["legal_policy"].items():
        other = indexed["batch_training"][key]
        assert row["opening"] == other["opening"]
        families[key[0] % 16].append((row["score"] or 0) - (other["score"] or 0))
    means = np.array([np.mean(family) for family in families])
    rng = np.random.default_rng(192016)
    bootstrap = means[rng.integers(0, 16, size=(20000, 16))].mean(axis=1)
    contrast = dict(
        legal_minus_global=float(means.mean()),
        paired_family_interval95=np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        censored={arm: progress["results"][arm]["match"]["censored"] for arm in rows},
        note="Censored scores conservatively counted zero; strength claims require zero censoring.",
    )
    write_json(out / "paired-comparison.json", contrast)
    progress["paired_comparison"] = contrast
    progress["phase"] = "completed"
    write_json(out / "progress.json", progress)
    write_json(out / "result.json", dict(status="completed", **progress))


if __name__ == "__main__":
    main()
