"""Frozen inference-only interventions; no weights or normalization stats change."""

import hashlib
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, "/opt/chesszero/chesszero")
from run_training import evaluate, write_json
from restart_training import color_summary

SOURCE = Path("/results/pr29-first-result-20261014-202746")


def mirror_states(states):
    channels = [*range(6, 12), *range(6), 12, 15, 16, 13, 14, 17, 18]
    out = np.take(np.asarray(states)[..., ::-1, :], channels, axis=-3)
    out[..., 12, :, :] = 1 - out[..., 12, :, :]
    return out


def mirror_policy_indices():
    square = np.arange(64).reshape(8, 8)[::-1].reshape(-1)
    out = np.arange(4672)
    out[:4096] = (square[:, None] * 64 + square[None, :]).reshape(-1)
    out[4096:4288] += 192
    out[4288:4480] -= 192
    return out


class Intervention:
    def __init__(self, initial, trained, kind):
        self.initial, self.trained, self.kind = initial, trained, kind

    def eval(self):
        self.initial.eval()
        self.trained.eval()

    def __call__(self, states):
        import mlx.core as mx

        policies, values = self.trained(states)
        if self.kind == "trained_policy_zero_value":
            return policies, mx.zeros_like(values)
        if self.kind == "trained_policy_initial_value":
            _, values = self.initial(states)
        elif self.kind == "initial_policy_trained_value":
            policies, _ = self.initial(states)
        elif self.kind == "trained_policy_negated_value":
            values = -values
        elif self.kind in ("mirror_value", "mirror_policy_and_value"):
            mirror_policies, mirror_values = self.trained(
                mx.array(mirror_states(states))
            )
            values = (values + mirror_values) / 2
            if self.kind == "mirror_policy_and_value":
                policies = (
                    policies + mirror_policies[:, mx.array(mirror_policy_indices())]
                ) / 2
        else:
            raise ValueError(self.kind)
        return policies, values


def fingerprint(model):
    from mlx.utils import tree_flatten

    digest = hashlib.sha256()
    for key, value in tree_flatten(model.parameters()):
        digest.update(key.encode())
        digest.update(np.asarray(value).tobytes())
    return digest.hexdigest()


def main():
    import mlx.core as mx
    from model.network import ChessNet

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    mx.set_default_device(mx.gpu)
    out = Path(sys.argv[1])
    initial, state0 = ChessNet.load_checkpoint(SOURCE / "checkpoints", 0)
    trained, state10 = ChessNet.load_checkpoint(SOURCE / "checkpoints", 10)
    del state0, state10
    initial.eval()
    trained.eval()
    before = (fingerprint(initial), fingerprint(trained))
    variants = (
        "trained_policy_zero_value",
        "trained_policy_initial_value",
        "initial_policy_trained_value",
        "mirror_value",
        "mirror_policy_and_value",
        "trained_policy_negated_value",
    )
    write_json(
        out / "protocol.json",
        dict(
            source=SOURCE.name,
            source_epoch=10,
            variants=variants,
            optimizer_updates=0,
            normalization_updates=0,
            protocol="Inference-output interventions only; same first 16 paired opening/random seeds "
            "as original 64-game bank, 32 games per variant. No evaluation data fitting. "
            "Reference rows restricted to identical pair IDs. Discovery, not confirmation.",
        ),
    )
    result = {"references": {}}
    for label in ("initial", "trained"):
        rows = json.loads((SOURCE / f"{label}-vs-random-games.json").read_text())
        rows = [r for r in rows if r["pair_id"] < 16]
        result["references"][label] = color_summary(rows)
    started = time.monotonic()
    for kind in variants:
        model = Intervention(initial, trained, kind)
        evaluate(model, trained.config, out, kind, 16, 10000)
        rows = json.loads((out / f"{kind}-games.json").read_text())
        result[kind] = color_summary(rows)
        assert before == (fingerprint(initial), fingerprint(trained))
        write_json(out / "progress.json", result)
        print("INTERVENTION_RESULT", kind, json.dumps(result[kind]), flush=True)
        mx.clear_cache()
    result.update(
        status="completed",
        elapsed_seconds=time.monotonic() - started,
        frozen_parameters_verified=True,
        optimizer_updates=0,
        normalization_updates=0,
    )
    write_json(out / "result.json", result)
    print("ABLATIONS_COMPLETE", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
