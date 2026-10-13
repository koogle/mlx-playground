# Full-model GPU throughput comparison

The merged production model/search code (main `fa5539f`) completed sixteen fresh self-play games and one position-weighted Adam update on each GPU. Both trials restored generation 4, used the original 19-block / 256-filter model, sixteen workers, 32 CPU cores, 32 MCTS simulations per move, and the same per-game seeds. Both produced the same game-length list and 2,734 positions, with no worker failures, Adam step 5, and finite parameters. These are execution/performance measurements; they do not establish convergence.

| Measurement | L40S | H100 |
| --- | ---: | ---: |
| Complete self-play generation | 231.97 s | 217.89 s |
| Games per minute | 4.14 | 4.41 |
| Median individual game time | 140.65 s | 130.74 s |
| One accumulated Adam update | 6.50 s | 8.46 s |
| Sampled peak GPU memory | 10.79 GiB | 11.95 GiB |

The H100 improved self-play throughput by 6.5% and combined self-play/learning time by about 5.1%. One matched generation is an initial comparison, not a repeated hardware benchmark. Memory was sampled every fifteen seconds, so brief peaks may be missed. Individual game times include worker inference and result delivery; generation time also includes worker startup. GPU cold start and checkpoint loading are outside the self-play timer. Neither trial batches inference across games.

The preceding merged-code generation with eight workers and 16 CPU cores completed twelve games / 2,777 positions in 337.67 seconds (2.13 games/minute), then trained once in 8.53 seconds. It used different game seeds, weights and CPU allocation; this is context rather than a controlled worker-count comparison.

The longer experiment continues on L40S from the complete generation-5 model/Adam state produced by its trial, using sixteen workers, 32 CPU cores and 64 fresh games per update. Larger generations keep workers occupied by starting another game when a short game finishes. All games finish before the single Adam update. Checkpoints are saved after every update; independent matches against random and the initial model run every ten generations. The allocation has an eleven-hour training deadline and a twelve-hour hard limit.

Raw measurements: `throughput-16-results.json`. Modal call IDs: `throughput-16-run.json`. Continuing run: `learning-16-run.json`. Reproduce the trials with `modal_gpu_throughput.py` and `gpu_throughput.py`, using a candidate export containing merged production code and the saved generation-4 checkpoint on the `chesszero-convergence` volume. Production move generation remains native.
