# ChessZero integration comparison

Status: **Modal NVIDIA/CUDA comparisons completed; no end-to-end speedup or playing-strength claim.**

Measured results and limitations: [RESULTS.md](RESULTS.md).

This branch combines PRs #5–#14 and the merged main updates (#4, #6, #7, #17). Its full unit suite passes 88 tests, including independent rule checks.
The engine itself has no dependency on python-chess; that package is a test oracle.

The comparison supports Apple Silicon/Metal or explicitly selected NVIDIA/CUDA.
GPU availability is required; there is no CPU fallback. NVIDIA results do not
predict Mac performance. Both revisions use the same hardware and packages.

## Reproduce on Modal NVIDIA

`modal_run.py` provisions one L4 (24 GB GPU), four CPU cores and 16 GiB RAM, runs
workloads serially, returns logs/results, and lets the ephemeral worker exit.
It uses NVIDIA’s CUDA 12.9.1 development image (headers are required by MLX’s
runtime compiler) and pins MLX 0.32.3 with CUDA 12, NumPy 2.2.6, psutil 7.0.0 and tqdm 4.67.1.
The result bundle includes the full installed package list.

Install `modal[api-proxy-support]` locally and configure Modal authentication.
Export only committed repository files, excluding local credentials and caches:

```sh
export CHESSZERO_EXPORT_DIR="$(mktemp -d)"
mkdir -p "$CHESSZERO_EXPORT_DIR/baseline" "$CHESSZERO_EXPORT_DIR/candidate"
git archive e754eecf08797d68797de135255a42e0ccac04ac | tar -x -C "$CHESSZERO_EXPORT_DIR/baseline"
git archive HEAD | tar -x -C "$CHESSZERO_EXPORT_DIR/candidate"
git rev-parse e754eecf08797d68797de135255a42e0ccac04ac > "$CHESSZERO_EXPORT_DIR/baseline/REVISION"
git rev-parse HEAD > "$CHESSZERO_EXPORT_DIR/candidate/REVISION"
modal run benchmarks/chesszero/modal_run.py --output benchmark-results/modal
```

On an already provisioned CUDA machine, install `mlx[cuda12]==0.32.3` and pass
`--backend cuda` to the comparison commands below.

## Reproduce on a Mac

Use Python 3.12 and the **same environment for both checkouts**. Install `mlx`,
`numpy`, `tqdm`, `psutil`, and `python-chess==1.999`; the output records the MLX and
Python versions. Record/pin the installed package versions alongside results.
The current Linux correctness tests use MLX 0.32.3 CPU; this is not a Mac result.

From a clone containing the integration branch:

```sh
git fetch origin
git worktree add --detach ../chesszero-baseline e754eecf08797d68797de135255a42e0ccac04ac
git worktree add --detach ../chesszero-candidate origin/chesszero-integration
cd ../chesszero-candidate/chesszero
python -m unittest discover -s tests
cd ..
python benchmarks/chesszero/compare.py \
  --baseline ../chesszero-baseline --candidate . \
  --output benchmark-results/smoke-2026 --profile smoke \
  --simulations 32 --games 2 --generations 3 --workers 2 --batch-size 64
python benchmarks/chesszero/compare.py \
  --baseline ../chesszero-baseline --candidate . \
  --output benchmark-results/default-search-2026 --profile default \
  --task search --simulations 32
```

`smoke` uses the real ChessNet architecture with 2 residual blocks and 64 filters
for a bounded training experiment; `default` keeps 19 blocks and 256 filters.
Both use the existing 4672-output policy head. Search count, batch size, and game
count are explicitly controlled rather than silently taking repository defaults.
Adam LR remains each branch's default. The original baseline used 0.2; the
updated baseline above includes #7, so both now use 0.001. This experiment
measures the entire proposed change. Repeat with multiple seeds
and alternate branch order externally before drawing stable throughput conclusions.

## What gets measured

- Board copy: seven batches of 5,000 copies after warm-up.
- Search: three fixed positions, three repeats, cold position caches, same
  model seed and architecture (branch initialization changes are retained). Logs elapsed time, network calls and evaluated positions,
  root/child visits, node count, and actually materialized boards. Search runs the
  branch's own mode/noise behavior as training would.
- Training: the actual Trainer and multiprocessing game-generation loop, complete
  fresh self-play games, then that branch's actual optimizer update schedule.
  Records generation/training times, game lengths/results, position counts,
  losses, optimizer steps, finite parameters, and final starting-position value
  and policy entropy. A deadline ends a workload without inventing draw targets.
- Hardware, commit, software versions, configuration, and parent-process peak RSS
  and GPU allocation are recorded. Parent peaks **exclude worker-process memory**.

Workloads run serially in separate processes to isolate caches and memory peaks.
Logs and intermediate JSON survive failures and timeouts. The launcher kills only
its own process group on timeout, including that run's self-play workers. It does
not provision, keep alive, or destroy a remote machine.

## Interpretation limits

The main branch's MCTS batching/backup and chess rules are defective. Equal
`n_simulations` values therefore do not mean equal useful search work; report
network evaluations and visits alongside latency. More correct work may take
longer even while board allocation becomes cheaper.

Self-play trajectories differ after the rule/search fixes. Worker scheduling and
randomness are not bit-for-bit deterministic, even with the same initial model
seed. Losses use different generated targets and update frequencies, so lower
loss alone does not demonstrate improved playing strength. A short smoke run
can establish functioning training and resource costs, not Elo or convergence.
A longer controlled strength evaluation remains a separate experiment.

Report only measured backend-specific speedups with the result files. Do not label timed-out
or incomplete games as completed training iterations.

Full-model learning follow-up: [protocol, negative results, and checkpoint artifacts](LEARNING.md). The final 21-update model regressed against both fixed opponents; lower loss did not establish convergence.

Controlled follow-up: [the original model learns a forced mate through fresh self-play](TACTICAL.md), with a statistics-only diagnostic isolating BatchNorm lag.

Larger experiment: [500-update target with parallel, pipelined fresh self-play](PARALLEL_LEARNING.md). GPU training is in progress; convergence is not yet established.
