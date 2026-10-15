# 3,000 additional self-play games

Run `pr29-3000-games-20261015-145659` continues the fresh-seed run's epoch-10
model and Adam state. Modal app:
https://modal.com/apps/radiantai/main/ap-bj5Yw15XT12kRlaRCEHIVL

The production source revision is `91671970d89ff89f65a892ddd3d34f764309736c`.
Artifacts persist in volume `chesszero-convergence`, under the run ID.

## Protocol

- Exactly 3,000 additional self-play games, with 64 MCTS simulations per move.
- Keep the existing model, learning rate (1e-4), and microbatch size (64).
- One accumulated Adam update per batch of up to 32 games, without replay or
  augmentation. Each 500-game interval has fifteen 32-game batches and one
  20-game batch: 96 additional updates, ending at Adam step 106.
- Save model and optimizer before publishing each batch's completed-game count.
- Every 500 games, evaluate 64 games against random moves, 32 per color, using
  bank seed 10000. Report WDL and scores with draws worth half a point, plus
  failures and censored games. Do not stop for an initial score decrease.
- After all games, run 64 direct matches against the frozen fresh-seed epoch-0
  model, using 32 paired openings from seed 80000, and verify the saved tensors
  and optimizer step. Random-opponent scores alone do not establish improvement
  over the initial model.

The worker uses one L40S, a 22-hour workload limit and a 23-hour Modal timeout.
Progress notifications check hourly and report only new 500-game milestones,
failure, required action, or completion.

## Operations

Read persisted status without launching a job:

```sh
/workspace/setup/modal/.venv/bin/python /workspace/chesszero-pr29/experiments/pr29-3000-games/status.py
```

`launch.json` is a local ignored record used by the status reader. Training can
resume using the launcher's `--resume-run` option after a terminal failure has
been confirmed. Resume uses the last published complete checkpoint and preserves
completed milestone reports; do not launch a second worker while one is active.
Preserve the configured runtime proxy and CA trust for all Modal operations.
