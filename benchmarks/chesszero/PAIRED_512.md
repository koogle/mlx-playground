# Main versus PR #22 at 512 parallel games

Two independent Modal L40S containers were launched simultaneously on 2026-10-14. Each requests 32 CPUs, 64 GiB initial RAM and a 384 GiB RAM cap. Main is frozen at `6ab1e6f0ee32dd51c93f6aa9140143d43391c47c`; #22 is frozen at `d504652c2fc0a783565460002128b2c53165c260`.

Both use the original 19-block/256-filter model, generation-4 checkpoint in `learning-merged-20261013-205953/checkpoints`, 512 games/workers, 32 simulations per move, seeds `9000 + game_id`, and training microbatches of 64. Each must finish every game, apply exactly one Adam update, keep finite weights, and round-trip the model/Adam checkpoint. Main uses its own per-worker models and generation-wide training-data conversion; #22 uses its central server and streaming conversion. No production source is patched for either trial.

Both environments set `MLX_CUDA_CONV_CACHE_SIZE=1024` to avoid the 128-entry runtime error encountered by the earlier combined-stack trial; this is an environment adjustment, not a source change. Both set OpenBLAS/OMP threads to one per process. The self-play deadline is part of an overall 3,600-second subprocess limit; timeouts or failed games are failures, not successful throughput measurements. GPU memory/utilisation and actual cgroup host memory are sampled every 15 seconds. Summed process RSS double-counts shared pages.

Metadata: `paired-512-run.json`. Results pending. Single trials on separate hosts do not establish a confidence interval or convergence. Changes in neural-network batch shape may change trajectories despite identical seeds.

To reproduce, archive the recorded commits into `$CHESSZERO_COMPARE_EXPORT_DIR/main` and `/candidate`, add their full SHA as `REVISION`, and deploy `modal_paired_compare.py` with its sibling `paired_compare_generation.py` and `modal_batched_throughput.py` on the import path. Spawn `main512` and `candidate512` with the same unique run ID. Raw output is retained under that ID in the `chesszero-convergence` Modal volume.
