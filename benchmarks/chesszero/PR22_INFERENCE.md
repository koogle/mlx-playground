# Shared inference on L40S

The process-based local-RPC implementation (`651efcd`) supersedes the thread-based version below. Fresh full-model trials at 16, 32, and 128 parallel games are in progress. These historical measurements do not validate the new architecture.

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
