# Random initialization repeat

This experiment asks whether the previous run's large White-only regression recurs with a different random initialization. It changes the initialization seed from `20261014` to `20261015`, while retaining the original model (19 residual blocks, 256 filters), 64 simulations, 32 workers, 32 complete self-play games per generation, batch size 64, learning rate 0.0001, and one accumulated Adam update per generation. It runs exactly ten generations (320 games, ten updates), with no early success stop. The self-play worker seeds are unchanged. No checkpoint is restored, no augmentation or normalization reset is applied, and evaluation data never enters training.

Before and after training, the candidate plays the same 64-game random-opponent bank, with 32 games as each color. Every evaluation move and terminal result is independently checked using python-chess. Draws count as half a point; censored games are explicit and prevent the paired comparison. Before/after games are joined by `(pair_id, color)` because workers return out of order. The final report includes per-color W/D/L, per-color score changes, and a paired bootstrap interval for the change in the White-minus-Black score gap. One repeated seed can show recurrence; it cannot establish a general causal explanation.

The CUDA worker runs all 77 mirror scoring cases before starting. It saves the fresh epoch-zero checkpoint and parameter fingerprint, then checkpoints model and Adam state every generation and verifies the final checkpoint round trip. Logs, heartbeat, game histories, and reports persist to the `chesszero-convergence` Modal volume every 15 seconds. The workload has a three-hour hard limit; failed or timed-out jobs do not count as completed results.

```sh
modal run --detach experiments/pr29-random-restart/modal_restart.py
python experiments/pr29-random-restart/status.py
PYTHONPATH=chesszero python -m pytest chesszero/tests experiments/pr29-random-restart/test_restart.py -q
```

In the managed workspace, use the trusted-CA Modal CLI wrapper:

```sh
/workspace/setup/modal/.venv/bin/python /workspace/setup/modal/run.py run --detach experiments/pr29-random-restart/modal_restart.py
```

Active launch: `pr29-random-restart-20261015-085706`, [Modal app](https://modal.com/apps/radiantai/main/ap-eaNi6LzYQdKDdJY1nsICqg), L40S. The ignored `launch.json` contains the function-call ID for read-only status checks. The reference run is `pr29-first-result-20261014-202746`; its before/after color changes are recomputed using the same identity-based join in the final report.
