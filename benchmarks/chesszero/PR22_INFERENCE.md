# Shared inference on L40S

The final process-based RPC implementation in PR #22 (`96cb5f6f0e16c73adc2bafa51f945a7f51b75884`) was tested on Modal L40S with 32 CPU cores and 64 GB requested container RAM. All trials use the original 19-block / 256-filter / 32,601,345-parameter model, the same generation-4 checkpoint, seeds `9000 + game_id`, 32 simulations per move, and an inference batch cap of 64. Each trial generates a complete fresh generation and then performs exactly one accumulated Adam update; learning is a separate phase.

CPU workers use NumPy only. MLX's CUDA build allocates GPU-managed memory even for its CPU arrays, so selecting `mx.cpu` was insufficient to eliminate worker CUDA contexts. The final worker modules import without MLX, receive no CUDA device visibility, and send plain NumPy data to a single model process through local manager RPC. Evaluation uses the same executor with current/accepted weight snapshots. PR #23's additional prediction cache and tree reuse are excluded from these GPU trials.

| Parallel games | Positions | Self-play time | Games/min | Positions/s | Sampled GPU peak | Mean batch |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 2,734 | 151.81 s | 6.32 | 18.01 | 2.15 GiB | 3.92 |
| 64 | 11,312 | 292.88 s | 13.11 | 38.62 | 7.43 GiB | 14.88 |
| 128 | 24,018 | 473.00 s | 16.24 | 50.78 | 10.90 GiB | 25.38 |
| 256 | 48,067 | 709.89 s | 21.64 | 67.71 | 10.68 GiB | 40.87 |

All 464 games across the four trials completed without failures. Every trial completed one update from Adam step 4 to 5 with finite parameters. The first sixteen game lengths matched across all settings, and the 16-game trial matched main's game lengths and losses exactly. The 128- and 256-game trials also matched their first 128 game lengths. Three later games differed between 64 and 128 workers (IDs 41, 44, and 54). Batch shapes can change floating-point evaluations and search trajectories; these three divergences have not been traced to a specific operation. Fixed seeds therefore do not establish full equivalence across concurrency settings.

At sixteen workers, main (`fa5539f`) took 231.97 seconds / 4.14 games per minute versus the final PR's 151.81 seconds / 6.32 games per minute: **52.8% higher throughput** and 34.6% shorter generation time. Sampled GPU peak fell from 10.79 to 2.15 GiB (80.1%). The 256-worker trial reached **21.64 games/minute**, 3.42 times the new 16-worker rate and 5.23 times main's recorded 16-worker rate. The latter comparison changes concurrency and is not an isolated code-change comparison.

256 is the largest tested successful concurrency, not a measured hardware limit. It was fastest here; its rate was 33.3% above 128 workers for twice the workers. GPU memory remained below 11 GiB, while CPU memory grew substantially. Peak summed process RSS was 11.51 / 24.84 / 43.85 / 79.30 GiB for 16 / 64 / 128 / 256 workers. **These RSS sums double-count shared pages and are not physical container RAM measurements.** More games remain possible in principle, but increasing them does not by itself saturate GPU compute. Approximate mean sampled self-play GPU utilisation was 29–38%; profiling inference execution, RPC scheduling, and CPU search would identify the next throughput limit.

Learning took 5.39 / 10.96 / 20.47 / 35.94 seconds respectively. These trials verify game generation, data transfer, gradient accumulation, and checkpoint publication. One generation at 32 simulations per move is not evidence of convergence or a measurement of the paused 128-simulation training configuration.

Sampling occurred about every fifteen seconds and may miss brief memory peaks. Utilisation averages omit the first thirty seconds and use approximate phase boundaries; complete generations include startup and the tail of long games. Different generation sizes include additional seeded games, so positions/second helps compare differing game lengths. These are single trials per setting, not repeated confidence intervals.

Raw results and sampling: `pr22-rpc-comparison.json`, `pr22-rpc-result-{16,64,128,256}.json`, `pr22-rpc-gpu-samples-{16,64,128,256}.json`, and `pr22-rpc-runs.json`. Reproduction uses `batched_gpu_throughput.py` and `modal_batched_throughput.py` with `CHESSZERO_EXPORT_DIR` pointing to an immutable archive of the recorded PR source. The harness commit is `312ef27`. All benchmark calls have ended; longer learning runs remain paused.

## Historical thread-based implementation

PR #22 (`07138c24a4aa630c9fb1b8002446751646eb4394`) was compared with merged main (`fa5539f`) on Modal L40S. Both used 32 CPU cores, sixteen game workers, the original 19-block / 256-filter network, generation-4 model/Adam state, sixteen fresh games, and 32 simulations per move. Per-game seeds were identical.

| Measurement | Main | PR #22 |
| --- | ---: | ---: |
| Complete self-play generation | 231.97 s | 202.26 s |
| Games per minute | 4.14 | 4.75 |
| Median individual game time | 140.65 s | 94.84 s |
| One accumulated Adam update | 6.50 s | 5.84 s |
| Sampled peak GPU memory | 10.79 GiB | 8.54 GiB |

Self-play throughput increased 14.7%; generation duration fell 12.8%. Sampled GPU peak fell 20.9%. Both trials completed all sixteen games with no failures, produced exactly the same game-length list and 2,734 positions, reported identical training losses, and saved a finite-parameter checkpoint at Adam step 5.

The shared owner evaluated 71,778 requested positions in 17,754 forwards (mean batch 4.04, largest batch 16). Search and game rules ran in CPU workers. This trial tests only shared inference, without PR #23's additional global prediction cache or retained subtrees.

This is one matched generation, not repeated throughput trials or evidence of convergence. Individual timings include result delivery; generation timing includes process startup. Container provisioning and checkpoint loading precede that timer. GPU memory was sampled every fifteen seconds and can miss brief peaks. Comparing median games alone overstates the whole-generation gain because the last long games determine the barrier time.

The bounded test has ended. The previous longer learning experiments are paused at the user's request while #22 and #23 are reviewed. The complete generation-5 model/Adam states remain on the `chesszero-convergence` volume.

Raw comparison: `pr22-gpu-comparison.json`. Modal call: `pr22-gpu-run.json`. The candidate trial used `batched_gpu_throughput.py` and `modal_batched_throughput.py`, with `CHESSZERO_EXPORT_DIR` pointing to a candidate export of exactly PR #22's recorded source revision. The baseline used `gpu_throughput.py` and `modal_gpu_throughput.py` on the recorded main revision. Both restore `/results/learning-merged-20261013-205953/checkpoints/state_epoch_4.json` from the shared Modal volume. All production move generation remains native.

## Streamed training tensors and successful 512-game retry

The initial 512-game attempt generated all 92,318 positions but stalled while creating a generation-wide policy tensor on CUDA, before backpropagation. A standalone L40S reproducer confirmed that NumPy stacking finished in 0.733 seconds but `mx.array` on the resulting 1.6 GiB policy array remained blocked until termination. PR #22 revision `e13893b` uploads one microbatch at a time; the same-sized synthetic fixture now completes all uploads in 0.624 seconds. The position order, all targets and one weighted-gradient update are preserved.

The real 512-game retry completed with no failures: 16.83 minutes of generation (30.43 games/minute), 57.92 seconds of training, finite model parameters, Adam step 4→5, and a saved checkpoint. Sampled peaks: GPU 10.59 GiB; actual CPU cgroup RAM 117.55 GiB. Average inference batch: 52.99 positions (cap 64). Approximate sampled GPU utilisation during self-play: 31.6%. The 16-game regression run also passed, with identical game lengths and losses to the prior version. See [CAPACITY.md](CAPACITY.md) for the raw evidence and limitations.

## 1024-game capacity test

The 1024-game retry passed: 181,645 positions, no game failures, one finite-parameter Adam update (4→5), and a saved checkpoint. Generation took 39.18 minutes (26.14 games/minute) and training 117.94 seconds. Peak sampled GPU memory was 10.65 GiB; actual CPU cgroup memory peaked at 226.96 GiB. This is 14.1% less generation throughput than 512 with about 1.93 times the host memory. Average inference batch was 59.48 (cap 64), while approximate sampled self-play GPU utilisation remained 28.8%. The first 512 game lengths matched the earlier run. 512 remains the fastest measured configuration; 1024 is the largest fully validated concurrency. The initial attempt completed generation at a similar rate but crashed during the benchmark's asynchronous stack dump; that diagnostic was reproduced and removed before retry. See `CAPACITY.md` for evidence and limitations.
