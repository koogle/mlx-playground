"""Can normalization warmup expose a harmful random value without any learning?"""

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, "/opt/chesszero/chesszero")
from ablate import Intervention
from calibrate import calibration_batches, learned_fingerprint
from run_training import evaluate, write_json
from restart_training import color_summary

SOURCE = Path("/results/pr29-first-result-20261014-202746")
CALIBRATION = Path("/results/pr29-normalization-diagnostic-20261015-093425")


def main():
    import logging
    import mlx.core as mx
    from model.network import ChessNet

    logging.basicConfig(level=logging.INFO)
    mx.set_default_device(mx.gpu)
    out = Path(sys.argv[1])
    model, checkpoint = ChessNet.load_checkpoint(SOURCE / "checkpoints", 0)
    del checkpoint
    model.eval()
    before = learned_fingerprint(model)
    states = np.load(CALIBRATION / "calibration_states.npz")["states"]
    write_json(
        out / "protocol.json",
        dict(
            source_run=SOURCE.name,
            epoch=0,
            learned_weights="Initial random weights held byte-identical",
            optimizer_updates=0,
            calibration_run=CALIBRATION.name,
            calibration_positions=len(states),
            protocol="One ordered batch64 BN-statistics-only pass on fresh self-play positions; no evaluation "
            "data used. Compare warmed random network and warmed policy with zero nonterminal value "
            "on the same original first16 pairs. Tests normalization warmup before any learned-weight update.",
        ),
    )
    model.train()
    for batch in calibration_batches(states, "ordered"):
        policies, values = model(mx.array(batch))
        mx.eval(policies, values, model.parameters())
    model.eval()
    assert learned_fingerprint(model) == before
    rows = json.loads((SOURCE / "initial-vs-random-games.json").read_text())
    result = {
        "initial_reference": color_summary([r for r in rows if r["pair_id"] < 16])
    }
    for name, network in (
        ("warmed_random", model),
        (
            "warmed_random_zero_value",
            Intervention(model, model, "trained_policy_zero_value"),
        ),
    ):
        evaluate(network, model.config, out, name, 16, 10000)
        rows = json.loads((out / f"{name}-games.json").read_text())
        result[name] = color_summary(rows)
        assert learned_fingerprint(model) == before
        write_json(out / "progress.json", result)
    write_json(
        out / "result.json",
        dict(
            status="completed",
            report=result,
            learned_weights_unchanged=True,
            optimizer_updates=0,
        ),
    )


if __name__ == "__main__":
    main()
