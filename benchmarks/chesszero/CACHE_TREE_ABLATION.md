# Cache/tree contribution experiment

Run `cache-tree-ablation-20261014-142409`, recorded in `cache-tree-ablation-run.json`, freezes production #22 at `d504652` and #27 at `da1f987`. Production PRs are unchanged. Each #22 treatment requests a separate L40S container with 32 CPUs, RAM reservation 64 GiB/cap 384 GiB, 512 fresh games/workers, 32 simulations/move, seeds `9000+game_id`, checkpoint generation 4, inference/microbatch cap 64, and exactly one final Adam update. The CUDA convolution cache is 1024; OpenBLAS/OMP thread limits are one. Treatments begin when Modal capacity is available; container provisioning/queue time is excluded from self-play timing.

Treatments: `neither`, `cache_only`, `tree_only`, `both`, and `both_raw_batch`. Prediction caching is disabled in both the per-game MCTS evaluation LRU and server LRU using a non-storing mapping; board and rule caches remain unchanged. In-flight duplicate coalescing remains enabled in all treatments. Tree reuse is disabled by dropping the root after every played move, retaining the identical board/rule caching and fresh simulation budget. Raw batching counts 64 queued requests before deduplication/cache lookup, rather than 64 unique misses. It still resolves cached responses before the forward, so it measures the batch-limit policy rather than perfectly recreating every detail of the old server.

All treatments apply the same benchmark-only instrumentation: selection steps, simulations, expansions, board copies, evaluation calls, RPC requests and cumulative wall-time of simulations/expansions/evaluations/RPC calls. Cumulative worker timings overlap and include CPU scheduling; they are not GPU kernel time or CPU process time. Network forwards/uncached positions/cache-hit metrics and sampled GPU/actual cgroup RAM are recorded. Summed RSS double-counts shared pages. Every dataset includes fresh self-play only and is used for one generation update, without replay or resampling.

#27 is a separate serial safety/baseline test: 16 requested games, 512 requested workers, the PR's enforced effective CUDA concurrency of one. It cannot provide a 512-parallel baseline. This short run verifies the actual GPU limit and end-to-end update rather than spending hours collecting 512 serial games.

The five prediction-cache toggles passed a CPU/IPC smoke check: repeated input invokes a second forward when caching is disabled, and reuses the prediction when enabled. GPU results completed successfully; see below. Seeds/configurations are matched, but tree reuse intentionally changes search paths and resulting game lengths; compare positions/sec, work per position/simulation, and neural-network work as well as games/minute. Single GPU trials do not provide confidence intervals. Fixed-workload diagnostics may be needed to distinguish per-search cost from changed trajectory length.

Reproduce using the archived revisions in the configured export directories, `ablation_generation.py`, `modal_ablation.py`, and the common `modal_batched_throughput.py` on the import path. All raw files are retained in `chesszero-convergence` under the recorded run ID and treatment name.

## Completed full-game results

| Treatment | Games/min | Positions/sec | Fresh positions | Self-play seconds | Host peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Neither prediction caching nor tree reuse | 19.60 | 58.92 | 92,318 | 1,566.97 | 87.69 |
| Prediction caching only | 23.82 | 71.59 | 92,318 | 1,289.58 | 118.03 |
| Tree reuse only | 25.50 | 57.42 | 69,186 | 1,204.91 | 88.79 |
| Both, current unique-miss batching | 29.72 | 66.94 | 69,179 | 1,033.52 | 118.22 |
| Both, raw-request batching | 30.25 | 68.12 | 69,179 | 1,015.49 | 118.72 |
| #27 serial CUDA baseline, 16 games | 0.82 | 1.82 | 2,143 | 1,177.14 | 4.81 |

Every treatment completed all requested games, with zero failures, finite learned parameters, exactly one Adam update (4→5), and a verified model/optimizer checkpoint round-trip. #27 logged the actual 512→1 CUDA worker limit and completed on the L40S without CUDA errors. GPU inference helpers still emitted the existing teardown warning after returning all predictions; this diagnostic does not fix that warning.

The cache-on/cache-off paths without tree reuse had identical full game lengths (92,318 total positions). Caching reduced uncached GPU positions from 3,025,201 to 2,413,994 (20.2%), with 588,974 local prediction hits. With tree reuse the benefit to network-work count was much smaller: 2,189,668 versus 2,164,712 uncached positions (1.1%); node statistics already retain evaluated positions. These single trials on different hosts should not be treated as precise causal timing percentages.

Tree reuse reduced dataset length by about 25.1% while increasing work per position. Caching-only versus both: selection steps per simulation grew from 1.256 to 1.520 (+21.0%); board copies per position from 824.9 to 1,048.3 (+27.1%); uncached network positions per played position from 26.15 to 31.29 (+19.7%); average GPU batch fell from 52.85 to 45.71. Position throughput was 6.5% lower with tree reuse, while games/minute was higher because games were shorter. Shorter games are not evidence of better training samples or playing strength; no convergence/strength evaluation was done here.

Current versus raw batching differs only about 1.8% in these trials. The result does not establish that either policy is faster. CPU-memory cost is measurable: retaining prediction caches added about 29–30 GiB at peak. All other board/rule caches remained enabled, so this is not a measurement of disabling every chess cache.

## Same-GPU crossover

To reduce host variance, five sequential diagnostics ran on one L40S UUID with the same CPU allocation, checkpoint, seeds and runtime. Each created 512 fresh opening rollouts, capped at 32 plies, and stopped early only at actual terminal positions. These partial rollouts were used for timing, without training or assigning artificial game outcomes. Connections were established while workers spawned, then a barrier released all 512 already-connected workers. Warm-up prediction counters were subtracted from timed statistics. Timing excludes model setup and worker startup; CPU process-time profiling also separates CPU execution from cumulative RPC wall waits.

| Same-GPU phase | Positions | Timed search seconds |
| --- | ---: | ---: |
| Legacy e138 RPC, original local cache/raw batching | 16,106 | 236.42 |
| Current RPC aligned to legacy cache/raw batching | 16,106 | 240.60 |
| Current RPC with shared cache, no tree reuse | 16,106 | 233.13 |
| Current RPC with shared cache and tree, profiled | 16,108 | 248.07 |
| Current RPC with shared cache and tree, unprofiled | 16,108 | 244.33 |

All 512 move-path digests match across the first three phases. The current/legacy aligned RPC difference is 1.8%, far smaller than the earlier apparent 16% full-run gap. The tree-on opening diagnostic is about 6.4% slower with identical profiling, or 4.8% comparing the unprofiled tree run to the profiled no-tree run; the profiled/unprofiled tree timing difference is about 1.5%, with 511/512 matching move paths and only 39 uncached network positions difference out of approximately 479,000. This is an approximate instrumentation comparison, not an exact overhead measurement. Treat the same-instrumentation comparison as the cleaner estimate. This confirms additional search cost per position; it does not imply slower completed-game throughput. See path digests in `crossover-result.json` for exact equivalence checks.

The earlier combined 512-game run took 1,204.47 s; the repeated profiled combined run here took 1,033.52 s with almost the same dataset (69,159 versus 69,179 positions), network work and mean batch (45.69 versus 45.71). The original apparent 16% regression is not reproduced and cannot be attributed wholesale to caching/tree reuse. The same-GPU test rules out a large RPC-code regression for its bounded opening workload; it cannot identify which host/scheduling/runtime factor caused the earlier full-game outlier. More repeats would be needed for a confidence interval.

The first crossover attempt failed before useful timing because synchronized new Unix-socket connections exceeded the manager accept backlog (`BlockingIOError: Resource temporarily unavailable`). The GPU remained healthy. The successful retry opened sockets before synchronization. Native self-play starts workers progressively and passed all six full-game treatments; nevertheless, burst connection setup is a transport robustness issue worth addressing separately. It is not a CUDA memory failure and should not be counted as a game or draw.

Raw evidence: `cache-tree-ablation-comparison.json`, per-treatment `ablation-*-result.json`, generation/job/telemetry/log files, and `crossover-result.json` with phase result/sample/job/log files. Reproduction for the crossover uses `prefix_crossover.py`, `modal_prefix_crossover.py`, archived source directories and `crossover-run.json`. Both experiments require the recorded enlarged CUDA convolution cache (1024); the earlier default-128 runtime failure is not resolved in production by this benchmark.
