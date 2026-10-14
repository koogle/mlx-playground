# Main versus PR #22 at 512 parallel games

Two independent Modal L40S containers were launched simultaneously on 2026-10-14. Each requests 32 CPUs, 64 GiB initial RAM and a 384 GiB RAM cap. Main is frozen at `6ab1e6f0ee32dd51c93f6aa9140143d43391c47c`; #22 is frozen at `d504652c2fc0a783565460002128b2c53165c260`.

Both use the original 19-block/256-filter model, generation-4 checkpoint in `learning-merged-20261013-205953/checkpoints`, 512 games/workers, 32 simulations per move, seeds `9000 + game_id`, and training microbatches of 64. Each must finish every game, apply exactly one Adam update, keep finite weights, and round-trip the model/Adam checkpoint. Main uses its own per-worker models and generation-wide training-data conversion; #22 uses its central server and streaming conversion. No production source is patched for either trial.

Both environments set `MLX_CUDA_CONV_CACHE_SIZE=1024` to avoid the 128-entry runtime error encountered by the earlier combined-stack trial; this is an environment adjustment, not a source change. Both set OpenBLAS/OMP threads to one per process. The self-play deadline is part of an overall 3,600-second subprocess limit; timeouts or failed games are failures, not successful throughput measurements. GPU memory/utilisation and actual cgroup host memory are sampled every 15 seconds. Summed process RSS double-counts shared pages.

Metadata: `paired-512-run.json`. Both trials finished; results below. Single trials on separate hosts do not establish a confidence interval or convergence. Changes in neural-network batch shape may change trajectories despite identical seeds.

To reproduce, archive the recorded commits into `$CHESSZERO_COMPARE_EXPORT_DIR/main` and `/candidate`, add their full SHA as `REVISION`, and deploy `modal_paired_compare.py` with its sibling `paired_compare_generation.py` and `modal_batched_throughput.py` on the import path. Spawn `main512` and `candidate512` with the same unique run ID. Raw output is retained under that ID in the `chesszero-convergence` Modal volume.

Main reported `cuGraphAddKernelNode` failure (`an illegal memory access was encountered`) in game 52, followed by `double free or corruption (!prev)`. GPU telemetry and worker progress then stopped advancing; its latest saved sample was 234.61 seconds. The main call was explicitly cancelled to release the failed container. This is an incomplete, manually aborted trial after observed failures, not a successful generation or normal timeout. Evidence: `paired-512-main-run.log`, `paired-512-main-gpu-samples.json`, `paired-512-main-abort.json`. The sampled main GPU peak was approximately 44.41 GiB. #22 subsequently completed the full generation and update.

## Completed comparison

| Metric | Main | PR #22 |
| --- | --- | --- |
| 512-game generation | Failed; manually stopped after CUDA errors and stalled progress | 512/512 completed, zero failures |
| Self-play | No valid full-generation timing | 1,204.47 s (20.07 min), 25.50 games/min |
| Fresh training positions | No complete dataset | 69,159 |
| Training | Not reached | 46.77 s; exactly one Adam update, step 4→5 |
| Sampled GPU peak | 44.41 GiB before progress stalled | 10.77 GiB |
| Actual host-memory peak | Incomplete trial, not comparable | 118.86 GiB |
| Parameter/checkpoint checks | Not reached | Finite parameters; weights and Adam state round-tripped |

#22 made 47,357 network forwards for 2,163,843 uncached positions (mean 45.69 per forward, maximum 64). It counted 29,790 cached/deduplicated requests. Approximate sampled self-play GPU utilisation was 28.25%. White won 192 games, Black won 182, and 138 were draws. This demonstrates an end-to-end update at 512 concurrency, not learning convergence or a speedup ratio against main: main did not complete its workload.

Both environments used the enlarged CUDA convolution cache described above. This setting is part of the passing runtime configuration; the prior default-128 run failed. The successful trial also emitted the existing `terminate called without an active exception` helper-shutdown warning after inference finished. All predictions/games returned, server error was null, the parent exited zero, and training/checkpoint checks passed; the warning remains an infrastructure issue, not resolved by this benchmark.

Raw evidence is in `paired-512-comparison.json`, the candidate generation/result/job/telemetry/log files, the main failure/telemetry/log files, and both configuration files beside this report. The calls used separate L40S containers under the same Modal app and started together; recorded code snapshots and settings are in `paired-512-run.json`. Main and #22 share the merged #25 cache/tree changes. Unlike the older server-only trials, this comparison includes the full reviewed stack.
