"""Exact first-generation counterfactual from saved weights, BN stats and Adam moments."""

import json
from pathlib import Path
import sys

sys.path.insert(0, "/opt/chesszero/chesszero")
from run_training import evaluate, write_json
from restart_training import color_summary

SOURCE = Path("/results/pr29-first-result-20261014-202746")


def main():
    import logging
    import inspect
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from model.network import ChessNet

    logging.basicConfig(level=logging.INFO)
    mx.set_default_device(mx.gpu)
    out = Path(sys.argv[1])
    initial, state0 = ChessNet.load_checkpoint(SOURCE / "checkpoints", 0)
    model, state1 = ChessNet.load_checkpoint(SOURCE / "checkpoints", 1)
    del state0
    assert int(state1["optimizer_state"]["step"]) == 1
    p0 = dict(tree_flatten(initial.parameters()))
    p1 = dict(tree_flatten(model.parameters()))
    moments = dict(tree_flatten(state1["optimizer_state"]))
    lr = float(moments["learning_rate"].item())

    def is_stat(name):
        return name.endswith(("running_mean", "running_var"))

    before_update, corrected = {}, {}
    checks = []
    for key, value in p0.items():
        if is_stat(key):
            before_update[key] = corrected[key] = p1[key]
        else:
            before_update[key] = value
            m, v = moments[key + ".m"], moments[key + ".v"]
            reconstructed = value - lr * m / (mx.sqrt(v) + 1e-8)
            checks.append(mx.allclose(reconstructed, p1[key], rtol=1e-5, atol=1e-7))
            corrected[key] = value - (lr / (1 - 0.9) * m) / (
                mx.sqrt(v) / mx.sqrt(1 - 0.999) + 1e-8
            )
    mx.eval(checks, corrected)
    assert all(
        bool(c.item()) for c in checks
    ), "Saved first update does not match uncorrected Adam"
    write_json(
        out / "protocol.json",
        dict(
            source_run=SOURCE.name,
            checkpoint_epochs=[0, 1],
            saved_adam_step=1,
            learning_rate=lr,
            adam_signature=str(inspect.signature(optim.Adam)),
            reconstructed_uncorrected_update_verified=True,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            protocol="Generation-wide accumulation holds learned weights fixed while updating BN statistics. "
            "Epoch0 weights plus epoch1 BN stats therefore reconstruct the state immediately before "
            "the first Adam update. Corrected first update uses the exact saved first-step m/v moments "
            "with bias correction, keeping identical BN stats, data, gradients and LR. No new optimizer "
            "steps or fitting; evaluate each frozen counterfactual on the original first16 pairs.",
        ),
    )
    results = {}
    for label, weights in (
        ("before_adam_update", before_update),
        ("corrected_adam_update", corrected),
        ("actual_adam_update", p1),
    ):
        model.load_weights(list(weights.items()))
        model.eval()
        evaluate(model, model.config, out, label, 16, 10000)
        rows = json.loads((out / f"{label}-games.json").read_text())
        results[label] = color_summary(rows)
        write_json(out / "progress.json", results)
        print("FIRST_STEP_RESULT", label, json.dumps(results[label]), flush=True)
    write_json(
        out / "result.json",
        dict(
            status="completed",
            report=results,
            new_optimizer_updates=0,
            reconstructed_uncorrected_update_verified=True,
        ),
    )


if __name__ == "__main__":
    main()
