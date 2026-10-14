# Higher-concurrency L40S sweep

## Successful 512-game retry

PR #22 source `e13893b` completed all **512 fresh self-play games**, yielding **92,318 positions**, with no failures. Generation took **1009.64 seconds (16.83 minutes)**: **30.43 games/minute**, or **91.44 positions/second**. Training took **57.92 seconds**, completed exactly one Adam update (step 4 to 5), and left all model parameters finite. The wrapper exited with code 0 after 1075.73 seconds, including startup, training, checkpoint saving and polling. The epoch-5 checkpoint manifest names nonempty model and complete optimizer tensor files.

Sampled peaks were **10.59 GiB GPU memory** and **117.55 GiB actual cgroup CPU memory**. Approximate sampled self-play GPU utilisation averaged **31.6%**. The inference service processed 2,443,368 positions in 46,112 forwards, averaging **52.99 positions per batch** (cap 64), with no backend error. GPU sampling is periodic and may miss brief peaks.

This is the largest fully validated concurrency so far, not a hardware ceiling. The measured generation throughput is 40.6% above the earlier 256-game trial, but both measurements are single trials, and the new source includes streamed training tensors. The test validates the execution of fresh self-play and backpropagation; one generation does not establish learning convergence. It uses the original full 19-block/256-filter model, the same generation-4 checkpoint and seeds, 32 simulations per move, 32 CPU cores and a 192 GiB host-memory cap. PR #23 remains excluded. The 1024/2048 stages have not been retried.

Raw results and telemetry are in `capacity-streamed-512-*`; the checkpoint manifest is in `capacity-streamed-512-checkpoint.json`.

## Original failed trial

The corrected 512-game probe finished generating all 512 game results (92,318 positions), then stalled during training-data preparation. The last log message is "Total positions collected: 92318", before the subsequent "Creating batches of size" message. The child was terminated at its one-hour deadline (exit code -15, elapsed 3606.85 seconds). There is no completed Adam update or final result.json for this trial. The controller stopped here, so 1024 and 2048 were not tested.

The longest individual game took 1244.41 seconds (20.74 minutes); these individual timers exclude process startup and server setup, so they cannot give the exact generation throughput. Sampled GPU memory peaked at 10,964 MiB (10.71 GiB), and actual container cgroup memory peaked at 117.90 GiB. GPU utilisation during the approximate self-play interval averaged about 29%; after generation it remained at zero while the CPU process stalled. This is a software/training-data preparation failure, not an established L40S capacity ceiling. The exact blocking operation remains unconfirmed. At that point, the 256-game trial remained the largest fully validated generation plus finite Adam update.

This probe used the full model, fixed PR #22 source `96cb5f6`, generation-4 checkpoint, 32 simulations/move, 32 CPU cores, inference batch cap 64, a 64 GiB host-memory reservation and 192 GiB host-memory cap. It does not include PR #23 changes and does not measure learning convergence.

Raw telemetry, per-game durations, termination status and log are preserved as `capacity-512-*`. `capacity-runs.json` records the completed controller state. The CPU-only controller invocation and volume directory are in `capacity-controller.json`; `capacity_sweep.py` and `modal_high_parallel.py` implement the staged sweep.

Earlier submissions were cancelled while queued or failed before executing a benchmark because their image omitted the shared `modal_batched_throughput` module. The corrected image includes that module. Those startup failures are not capacity measurements.

## Diagnosis and streamed-batch validation

The stall reproduces without any games, using 92,318 independent synthetic position/policy arrays on an L40S. A separate probe splits NumPy stacking from CUDA conversion: NumPy stacked the policy array in 0.733 seconds (1,725,238,784 bytes), then `mx.array(stacked)` did not return before termination after 158.87 seconds. Repeated stack traces point to that conversion. This identifies the blocking MLX CUDA operation; the underlying allocator/driver mechanism has not been established. It is not an observed out-of-memory error.

PR #22 source `e13893b` now creates and uploads one training microbatch at a time, rather than copying the whole generation into MLX arrays. The position order, perspective targets, unpadded tail, sample-weighted gradient accumulation and single Adam update remain the same. The synthetic L40S fixture uploaded all 92,318 positions in 0.624 seconds with this change. Raw evidence is in `batch-diagnostic-bulk.json` and `batch-diagnostic-streamed.json`.

An actual 16-game L40S run generated 2,734 positions in 168.04 seconds, completed training in 3.431 seconds, saved a checkpoint and advanced Adam from step 4 to 5 with finite parameters. Its game lengths and all four losses exactly match the prior 16-game run. All 82 PR #22 tests and 86 combined integration tests pass. This verifies execution and preserves the small-run result; it does not establish learning convergence.

The fresh 512-game rerun completed successfully. Its invocation/source metadata is in `capacity-streamed-512-run.json`. The benchmark now saves generation metrics before learning and emits repeated stack traces if training takes more than two minutes, so failures cannot hide all generation timings.

## First 1024-game attempt and diagnostic crash

The first 1024-game attempt (`scale-streamed-rpc-22-1024-20261014-075224`, source `e13893b`) completed all 1024 games with no game failures and produced 181,645 positions. Generation took 2360.06 seconds (39.33 minutes), or 26.03 games/minute and 76.97 positions/second. This is 14.4% fewer games/minute than the 512-game trial. Inference averaged 59.48 positions per forward (cap 64). Sampled peaks: GPU 10.74 GiB; actual cgroup CPU RAM 226.36 GiB.

Training was active when the benchmark's asynchronous `faulthandler` stack dump fired after two minutes. The process crashed mid-dump (SIGSEGV, exit code -11) before completing an Adam update or checkpoint. This trial therefore validates 1024-game generation only. Raw evidence is in `capacity-streamed-1024-*`.

Two 100-microbatch controls passed with and without one-second dumping. A stronger 5 ms dump stress test reproduced the same mid-stack-dump segmentation fault during MLX autodiff. A full-sized control without dumping then trained 181,645 synthetic positions, produced finite parameters and Adam step 5, and completed in 128.74 seconds. These controls isolate diagnostic safety and training execution, not chess learning. `training-diagnostic-comparison.json` and `training-full-control.json` preserve the results.

The benchmark now uses ordinary progress messages instead of asynchronous frame walking and saves each fresh generation to `games.pkl` before training. The second 1024-game end-to-end run completed successfully; its metadata is in `capacity-streamed-1024-retry-run.json`. The production model/search/training source remains unchanged.

The inference helper prints `terminate called without an active exception` during shutdown in both the successful 512-game trial and this first 1024-game attempt, after all game predictions have returned. This is separate from the diagnostic crash during training and has not been identified as a capacity failure.

## Successful 1024-game retry

The corrected benchmark completed all 1024 fresh self-play games with no game failures, yielding 181,645 positions. Generation took 2350.58 seconds (39.18 minutes): 26.14 games/minute and 77.28 positions/second. Training took 117.94 seconds, advanced Adam from step 4 to 5 exactly once, and left all model parameters finite. The epoch-5 checkpoint manifest names nonempty full-model and optimizer tensor files. The wrapper exited with code 0 after 2504.47 seconds (41.74 minutes), including startup, fresh-data persistence, checkpoint saving and polling. The fresh generation is saved in the Modal volume as `games.pkl` (4,290,320,247 bytes); it was not replayed or randomly sampled.

| Parallel games | Generation minutes | Games/minute | Actual CPU RAM peak | Sampled GPU peak | Training seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 16.83 | 30.43 | 117.55 GiB | 10.59 GiB | 57.92 |
| 1024 | 39.18 | 26.14 | 226.96 GiB | 10.65 GiB | 117.94 |

1024 is supported, but measured generation throughput is 14.1% lower than at 512, using about 1.93 times the host memory. Average inference batch increased from 52.99 to 59.48 positions (cap 64); approximate sampled self-play GPU utilisation was 28.8%. The first 512 game lengths exactly match the earlier 512-game trial. The first 1024 attempt also measured a similar generation rate (26.03 games/minute), before the diagnostic crash. There is no established GPU-memory ceiling; 512 remains the fastest measured configuration under these settings.

These comparisons use the same full original network, checkpoint, simulation count, seed scheme, CPU allocation, batch cap and frozen production source `e13893b`. The retry changes benchmark diagnostics and fresh-data persistence only. Tree reuse/shared prediction caching from the reordered PR stack were excluded. These are a few throughput trials, not evidence of learning convergence or a profile identifying the slowdown's cause.

Raw results, telemetry, termination status and checkpoint manifest are in `capacity-streamed-1024-retry-*`.

A combined-stack training run was launched on 2026-10-14: `reviewed-training-512-20261014-095942`, Modal call `fc-01M4WXMZ6G1PGTS26RZWVB1VE5`, source `e4bc1604d9c4b4a3738a2f06958b3f5616e26a39`. It requests three fresh 512-game generations with 32 MCTS simulations/move and 64-position training microbatches, followed by 40 games against the starting model. Unlike earlier throughput trials, it includes #25 and the shared server prediction cache. Each generation must complete all games, apply exactly one Adam update, keep finite parameters, and round-trip its model/optimizer checkpoint. Results are pending. Reproduction uses `reviewed_training.py`, `modal_reviewed_training.py`, and a git archive of the recorded source revision in `/workspace/chesszero-training-current-export/candidate`.
