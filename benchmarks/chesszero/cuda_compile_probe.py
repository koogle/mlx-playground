"""Isolate CUDA compilation costs for varying final microbatch sizes."""

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument(
        "--mode", choices=["compiled", "shapeless", "eager"], required=True
    )
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.repo) / "chesszero"))
    import mlx.core as mx
    import mlx.nn as nn
    from config.model_config import ModelConfig
    from model.training import Trainer

    mx.set_default_device(mx.gpu)
    mx.random.seed(2026)
    config = ModelConfig(n_residual_blocks=2, n_filters=64, learning_rate=1e-3)
    original_compile = mx.compile

    original_value_and_grad = nn.value_and_grad

    def gradient_variant(model, loss_fn):
        function = original_value_and_grad(model, loss_fn)
        if args.mode == "eager":
            return function
        return original_compile(
            function,
            inputs=model.state,
            outputs=model.state,
            shapeless=args.mode == "shapeless",
        )

    rows = []
    with tempfile.TemporaryDirectory() as directory:
        trainer = Trainer(config, directory)
        mx.eval(trainer.model.parameters())
        # Override both old (explicit compile) and new (plain autodiff) Trainers.
        with (
            patch("model.training.nn.value_and_grad", gradient_variant),
            patch("model.training.mx.compile", lambda function, **kwargs: function),
        ):
            for count in (87, 84, 95, 87):
                batch = (
                    mx.random.normal((count, 19, 8, 8)),
                    mx.ones((count, 4672)) / 4672,
                    mx.zeros(count),
                )
                mx.eval(batch)
                start = time.perf_counter()
                loss = trainer.train_on_batches(
                    tuple(
                        tuple(a[i : i + 64] for a in batch) for i in range(0, count, 64)
                    )
                )
                row = {
                    "positions": count,
                    "seconds": time.perf_counter() - start,
                    "loss": loss,
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
    print(json.dumps({"mode": args.mode, "results": rows}), flush=True)


if __name__ == "__main__":
    main()
