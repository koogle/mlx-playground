"""Untouched random-seed bank for the preselected mirror-value intervention."""

import json
from pathlib import Path
import sys

sys.path.insert(0, "/opt/chesszero/chesszero")
from ablate import Intervention, fingerprint
from run_training import evaluate, write_json
from restart_training import color_summary, compare

SOURCE = Path("/results/pr29-first-result-20261014-202746")


def main():
    import logging
    import mlx.core as mx
    from model.network import ChessNet

    logging.basicConfig(level=logging.INFO)
    mx.set_default_device(mx.gpu)
    out = Path(sys.argv[1])
    initial, state = ChessNet.load_checkpoint(SOURCE / "checkpoints", 0)
    del state
    trained, state = ChessNet.load_checkpoint(SOURCE / "checkpoints", 10)
    del state
    initial.eval()
    trained.eval()
    frozen = (fingerprint(initial), fingerprint(trained))
    write_json(
        out / "protocol.json",
        dict(
            source_run=SOURCE.name,
            epoch=10,
            bank_seed=60000,
            games_per_case=64,
            games_per_color_per_case=32,
            preselected_intervention="mirror_value",
            optimizer_updates=0,
            normalization_updates=0,
            protocol="Single fixed mirror-value formula selected before this new seed bank. Same openings "
            "as discovery but new random-opponent seeds. Compare unchanged and mirrored values "
            "with identical trained policy and exact terminal scores. No evaluation-data fitting. "
            "This confirms an imbalance intervention versus random, not direct superiority to initial model.",
        ),
    )
    rows = {}
    for label, model in (
        ("trained", trained),
        ("mirror_value", Intervention(initial, trained, "mirror_value")),
    ):
        evaluate(model, trained.config, out, label, 32, 60000)
        rows[label] = json.loads((out / f"{label}-games.json").read_text())
        assert frozen == (fingerprint(initial), fingerprint(trained))
        write_json(
            out / "progress.json", {k: color_summary(v) for k, v in rows.items()}
        )
    result = dict(
        status="completed",
        by_case={k: color_summary(v) for k, v in rows.items()},
        mirror_minus_trained=compare(rows["trained"], rows["mirror_value"]),
        frozen_parameters_verified=True,
        optimizer_updates=0,
        normalization_updates=0,
    )
    write_json(out / "result.json", result)
    print("CONFIRMATION_COMPLETE", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
