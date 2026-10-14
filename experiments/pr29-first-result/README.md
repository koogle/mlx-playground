# PR #29 first-result run

The original launch stopped before learning when independent evaluation detected false repetition draws. The branch now retains full repetition keys, serves self-play and both evaluation models through parent-owned inference, keeps MLX out of spawned CLI workers, streams training batches, checkpoints each epoch, and prints progress into Modal's app logs.

## Validation

Run the local suite from the repository root, with the project and test requirements installed:

```sh
PYTHONPATH=chesszero python -m pytest -q chesszero/tests
```

The Modal preflight uses the original 19-block/256-filter model on an L40S. It runs selected protocol/training tests; independently verifies all 64 random-control games, including both previously failing games; runs two real CLI epochs with two complete self-play games per epoch and two simulations per move; evaluates against the accepted model and random moves; and reloads epoch 2 with finite parameters and Adam state. The low search budget is a smoke test, not a strength result.

```sh
modal run experiments/pr29-first-result/modal_train.py::preflight
```

The recorded 500-ply failure and a forced digest-collision test are permanent regressions. The validator checks every evaluation move and terminal result using python-chess; production rules/search never import that library.

Validation completed: **92 local tests passed**; the L40S preflight completed both CLI epochs, all 64 independent random-control games, model and random evaluation, and checkpoint reload. [Evidence](results/validation.json) and [raw preflight log](results/preflight.log). The four-game low-search random smoke scored 37.5%; it is not a strength result.

## Training protocol

- One L40S, 32 CPU cores, 64 GiB requested RAM with a 128 GiB maximum.
- Original 19 residual blocks, 256 filters, 4,672 policy outputs; fresh initial tensor snapshot, seed 20261014.
- 32 fresh complete games per generation, 32 CPU workers, 64 simulations per move, inference batch up to 64.
- Adam learning rate 0.0001, microbatch 64, one position-weighted update per generation. No replay, supervised data, or evaluation-data training.
- Save weights and optimizer every generation; check finite parameters, Adam advancement, and the final checkpoint round trip.
- Up to 60 generations; stop starting new generations after three hours, with a four-hour hard workload limit and five-minute cleanup allowance. Completed checkpoints survive failures/timeouts.

The initial tensor snapshot may be restored from the earlier run's epoch 0, but its 64-game initial evaluation is rerun with the corrected rules. The 64-game random-control bank may be reused from a successful preflight. Both references are explicit launch arguments and recorded in the experiment manifest; no trained weights from the failed run exist.

Evaluation uses balanced colors, deterministic MCTS without root noise, and a uniformly random legal-move opponent. Games beyond 1,024 plies or 600 seconds are censored, never counted as draws. Probe every 10 generations with 16 games on a separate seed bank; a probe scoring at least 60% triggers one final independent 64-game confirmation. Otherwise final confirmation happens at the budget. Success requires complete final games, one-sided binomial p<0.05 for wins versus losses, and a color-pair bootstrap interval with lower bound above 50%. Draws score half a point. Trained-minus-initial scores are reported separately: beating random alone does not establish improvement over untrained MCTS.

## Launch and progress

```sh
modal run --detach experiments/pr29-first-result/modal_train.py
python experiments/pr29-first-result/status.py
```

Optional launch flags `--baseline-run RUN_ID` restore only the untrained epoch-0 tensor snapshot, and `--preflight-run RUN_ID` reuse that preflight's verified random-control bank. Without them both baselines run fresh.

This workspace uses the existing trusted-CA Modal launcher:

```sh
/workspace/setup/modal/.venv/bin/python /workspace/setup/modal/run.py run --detach experiments/pr29-first-result/modal_train.py
/workspace/setup/modal/.venv/bin/python experiments/pr29-first-result/status.py
```

Launch IDs are saved to `launch.json` locally; they are not committed. All remote artifacts live on volume `chesszero-convergence` under the run ID: source/config manifest, checkpoints, full evaluation move histories, generation metrics, app/run logs, heartbeat, exit status, and final result. Only one bounded training job is launched; no recurring automation or automatic retries are configured.
