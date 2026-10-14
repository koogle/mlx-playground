# Higher-concurrency L40S sweep

The corrected 512-game probe finished generating all 512 game results (92,318 positions), then stalled during training-data preparation. The last log message is "Total positions collected: 92318", before the subsequent "Creating batches of size" message. The child was terminated at its one-hour deadline (exit code -15, elapsed 3606.85 seconds). There is no completed Adam update or final result.json for this trial. The controller stopped here, so 1024 and 2048 were not tested.

The longest individual game took 1244.41 seconds (20.74 minutes); these individual timers exclude process startup and server setup, so they cannot give the exact generation throughput. Sampled GPU memory peaked at 10,964 MiB (10.71 GiB), and actual container cgroup memory peaked at 117.90 GiB. GPU utilisation during the approximate self-play interval averaged about 29%; after generation it remained at zero while the CPU process stalled. This is a software/training-data preparation failure, not an established L40S capacity ceiling. The exact blocking operation remains unconfirmed. The 256-game trial remains the largest fully validated generation plus finite Adam update.

This probe used the full model, fixed PR #22 source `96cb5f6`, generation-4 checkpoint, 32 simulations/move, 32 CPU cores, inference batch cap 64, a 64 GiB host-memory reservation and 192 GiB host-memory cap. It does not include PR #23 changes and does not measure learning convergence.

Raw telemetry, per-game durations, termination status and log are preserved as `capacity-512-*`. `capacity-runs.json` records the completed controller state. The CPU-only controller invocation and volume directory are in `capacity-controller.json`; `capacity_sweep.py` and `modal_high_parallel.py` implement the staged sweep.

Earlier submissions were cancelled while queued or failed before executing a benchmark because their image omitted the shared `modal_batched_throughput` module. The corrected image includes that module. Those startup failures are not capacity measurements.

## Diagnosis and streamed-batch validation

The stall reproduces without any games, using 92,318 independent synthetic position/policy arrays on an L40S. A separate probe splits NumPy stacking from CUDA conversion: NumPy stacked the policy array in 0.733 seconds (1,725,238,784 bytes), then `mx.array(stacked)` did not return before termination after 158.87 seconds. Repeated stack traces point to that conversion. This identifies the blocking MLX CUDA operation; the underlying allocator/driver mechanism has not been established. It is not an observed out-of-memory error.

PR #22 source `e13893b` now creates and uploads one training microbatch at a time, rather than copying the whole generation into MLX arrays. The position order, perspective targets, unpadded tail, sample-weighted gradient accumulation and single Adam update remain the same. The synthetic L40S fixture uploaded all 92,318 positions in 0.624 seconds with this change. Raw evidence is in `batch-diagnostic-bulk.json` and `batch-diagnostic-streamed.json`.

An actual 16-game L40S run generated 2,734 positions in 168.04 seconds, completed training in 3.431 seconds, saved a checkpoint and advanced Adam from step 4 to 5 with finite parameters. Its game lengths and all four losses exactly match the prior 16-game run. All 82 PR #22 tests and 86 combined integration tests pass. This verifies execution and preserves the small-run result; it does not establish learning convergence.

The fresh 512-game rerun is in progress. Its invocation/source metadata is in `capacity-streamed-512-run.json`. The benchmark now saves generation metrics before learning and emits repeated stack traces if training takes more than two minutes, so failures cannot hide all generation timings.
