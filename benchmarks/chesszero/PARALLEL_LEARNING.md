# Larger pipelined self-play run

The run is paused at saved generation 3 while the focused PRs are reviewed.
Model weights and full Adam state are retained in
`learning-parallel-20261013-164630/checkpoints/state_epoch_3.json`.
No GPU training was resumed during the branch refresh.

This experiment uses the original 19-block, 256-filter ChessNet (32,601,345
parameters), starting with seed 2026 and untrained weights. It targets 500 Adam
updates, with 12 new complete games per update: up to 6,000 games. The Modal
allocation is one L40S, 16 CPUs and 64 GiB host memory, with a hard 12-hour
function timeout. Training admission stops at 11 hours to leave evaluation time;
actual completed updates, not the target, determine the reported sample size.

Eight spawned game processes work concurrently. Once batch N finishes, a frozen
snapshot starts batch N+1 before the learner updates on N. Consequently, after
the first batch, self-play weights are one update behind the learner at update
start. Each generation records its source model version. There is at most one
batch in flight. CPU arrays cross process boundaries; workers never share the
learner's mutable model. Each fresh game is used exactly once, in game-ID order,
with position-weighted gradient accumulation and one Adam update per batch.
There is no replay buffer or random resampling.

The experiment retains the earlier 32-search-simulations-per-move setting,
learning rate 0.001, batch size 64 and production temperature 1 throughout a game.
This increases the number of game sets rather than changing search strength.
The pipeline adds a bounded policy lag, so it is not a controlled replication of
the sequential run. No speedup or chess-strength improvement is assumed.

Native chess implements games and targets. Fixed-opening, color-swapped matches
compare against random and the initial network every 50 updates (16 games per
opponent) and at completion (32 per opponent). The initial random baseline uses
32 games. Evaluation uses an independent chess library only to validate legal
moves and outcomes. Games unfinished after the evaluation cap are censored,
never scored as draws. Evaluation games do not train the network.

A deterministic, evenly spaced selection of up to 64 positions from the current
fresh batch measures value MSE in evaluation and training modes at update 1 and
every 10 updates. All model parameters and BatchNorm statistics are restored
after this diagnostic. No BatchNorm recalibration treatment is applied.

Checkpoints include Adam state and are written every update. Retention keeps the
latest two and every 50th update. Metrics and checkpoints persist on Modal volume
`chesszero-convergence`. The deployed function is spawned independently of the
local client; its allocation ends on completion or timeout. A training deadline
may still be interrupted by the outer timeout if a game/evaluation runs long.
Worker errors propagate; incomplete games do not become outcome targets.

Launched 2026-10-13 at 16:46 UTC with source `a035232`:

- Run: `learning-parallel-20261013-164630`
- App: `ap-8er1j2jsC3UxfiQlOPSlQa`
- Function call: `fc-01M4V2H4CR7NHV13GKF0X2TCE5`
- [Modal deployment](https://modal.com/apps/radiantai/main/deployed/chesszero-convergence)

Read persisted progress without allocating a GPU:

```sh
python benchmarks/chesszero/learning_status.py learning-parallel-20261013-164630
```

After refreshing onto main and the remaining focused fixes, ten experiment tests
and 75 repository tests pass on MLX 0.32.3 CPU. They cover
snapshot isolation, game ordering, worker failure propagation, and preservation
of weights/running statistics by the value-mode diagnostic. Actual GPU results
remain pending; this protocol is not evidence of full-chess convergence.

The first launch (`learning-parallel-20261013-164331`) was stopped before any training update when a local integration check exposed missing NumPy-to-MLX conversion at the process handoff. The correction is covered by a production-batcher/optimizer integration test. Its initial evaluation completed (21 wins, 10 draws, 1 loss); the replacement independently repeats the baseline.
