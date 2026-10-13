# NVIDIA L4 comparison

The integrated changes make original-size fixed-position search 2.30× faster,
but the final short training run takes 1.94× longer than updated main. A CUDA
compilation bottleneck found during testing is fixed; self-play now dominates.
These experiments do not establish increased playing strength or faster convergence.

## Final comparison against updated main

Raw results and logs: [l4-current-main](results/l4-current-main/).
Main advanced during the experiment: this comparison uses
`e754eecf08797d68797de135255a42e0ccac04ac`, including the merged promotion (#6),
Adam (#7) and equivalent value-target conditional (#17) changes. Candidate
`76c33b3e121b96c8068d8d4bc32e6345d060b692` includes that main revision plus the
remaining fixes and the gradient compilation correction. Both use Adam LR 0.001.

All workloads completed on the same L4, with the same pinned environment and
four CPU cores / 16 GiB container RAM. All **87 tests passed on the GPU** (15.16 s)
and on CPU (1.11 s). The production engine implements its own chess rules;
python-chess is used only by tests.

Search uses three fixed positions repeated three times with 32 requested
simulations. Training uses 2 residual blocks / 64 filters, three generations of
two complete fresh games, two workers, and microbatches of 64. The original-size
search model has 19 blocks / 256 filters. No replay or random sampling is added.

| Measurement | Updated main | Integrated candidate |
| --- | ---: | ---: |
| Starting-board copy, median | 288.29 µs | 1.80 µs (**160× faster**) |
| Small-network fixed-position search, median | 270.90 ms | 86.25 ms (**3.14× faster**) |
| Original-size network search, median | 309.26 ms | 134.68 ms (**2.30× faster**) |
| Original-size search GPU peak, parent | 649.89 MB | 154.90 MB (**76% lower**) |
| Materialized boards per original-size search, median | 610 | 33 |
| Three fresh training generations / six games, elapsed | 65.71 s | **127.78 s** |
| Self-play time | 62.97 s | 125.56 s |
| Learning time | 2.56 s | 2.05 s |
| Training positions | 1,186 | 1,271 |
| Adam updates | 20 | 3 |
| Starting-position policy entropy after training | 2.29 | 8.42 |

**The integrated branch is not faster end to end in this run.** Its self-play
accounts for 98% of elapsed time. Fixed-position search wins do not transfer
directly to evolving whole games: the revisions have different search accounting,
mode/cache behavior, rules, learned policies and update schedules. Main records
median root/child visits of 54/74 for 32 requested simulations; the candidate
records 32/32, evaluating 33 rather than 22 median network positions. These are
combined changes, not an isolated comparison of equivalent search work.

Both models remain finite. Main no longer shows the original 0.2-learning-rate
loss spike; its six games are draws. The candidate produces three draws and three
decisive games. Neither this result nor the entropy difference establishes better
chess strength. The candidate's learning passes take 1.78, 0.15 and 0.12 seconds,
so optimizing that phase further has little impact on this workload.

The next useful performance investigation is a profile of **whole-game self-play**
at several stages of training, separating move generation, search traversal,
board work and neural inference. Cross-game inference batching and persistent
workers remain deferred as requested. A longer evaluation across multiple seeds
is needed to compare playing strength per unit of compute. The original-size
network was benchmarked for search only; training used the smaller network.

## Gradient compilation follow-up

The isolated [probe data](results/cuda-compile-probe.json) confirms the cost of
specializing the entire gradient graph for new batch shapes. It trains the same
seeded small network on synthetic batches of 87, 84, 95 and 87 positions, split
at 64, in separate processes on one L4:

| Gradient execution | First shape | New tail shape | Another new tail | Repeated shape |
| --- | ---: | ---: | ---: | ---: |
| Whole-gradient compilation | 169.13 s | 32.15 s | 31.65 s | 24 ms |
| Plain MLX autodiff | 1.87 s | 66 ms | 57 ms | 35 ms |

Compiled execution wins by about 11 ms once the graph is cached, but paying tens
of seconds for each new tail dominates these short fresh-generation runs.
Shape-independent compilation fails with `Pad cannot infer output shapes` on
MLX 0.32.3; it also fails on the repository's older MLX 0.21.1. The loss results
agree closely between compiled and plain autodiff. This probe uses synthetic
inputs only to isolate execution overhead; the training benchmarks use actual
complete self-play games.

PR #11 now reuses `nn.value_and_grad` without wrapping the entire gradient graph
in `mx.compile`. Device execution, position weighting, BatchNorm updates and the
single Adam step are preserved. The seven focused accumulation tests and all 87
integrated tests pass. The final GPU suite takes about 15–16 seconds versus 134
seconds initially. Long runs with repeatedly reused shapes may amortize compile
costs differently; this is not a claim that compilation is always slower.

## Rerun against the original baseline

Raw data: [l4-compilation-fix](results/l4-compilation-fix/). Baseline remains
`6a334ec3092fc9f0421ec7984157d1e997b47d5c`; candidate is
`ad0795704a40f2c5a2eaec358f9b8e00be7b63c9`, with the compilation fix.

The candidate's three learning passes now take **1.73, 0.32 and 0.13 seconds**,
processing 582, 656 and 175 positions respectively. All 87 GPU tests pass, both
models retain finite parameters, and the candidate applies exactly three Adam
updates. Default-network search remains 2.44× faster (320.26 versus 131.36 ms).

Total training wall time is still higher for the candidate: **116.37 versus
60.54 seconds**, with 1,413 versus 1,258 positions. Candidate self-play consumes
114.13 seconds; the learning phase is no longer the bottleneck. Main's policy
again collapses to effectively zero starting-position entropy after the large
learning-rate updates, while candidate entropy remains 8.41. Different game
lengths, search work and resulting policies make this an unsuitable basis for
claiming faster convergence. The updated-main comparison controls for the now
merged learning-rate change.

## Initial run (before the compilation fix)

Raw data: [l4-initial](results/l4-initial/). Both revisions ran serially on one
Modal NVIDIA L4 (23,034 MiB reported), with four allocated CPU cores and 16 GiB
container RAM. The large `host_memory_bytes` field describes the underlying host,
not the container allocation. Python 3.12.1, MLX 0.32.3/CUDA 12, driver 610.57.04;
the complete package list is included with the results.

- Baseline: main at experiment start, `6a334ec3092fc9f0421ec7984157d1e997b47d5c` (includes #4).
- Candidate: `618863b52318c6dcf2c82953be01ade2f3cfd180` (integrates #5–#14).
- All **87 tests passed on the GPU**, including native chess-rule tests checked
  against an independent test-only oracle, checkpoint/Adam restoration, and MCTS.

| Measurement | Main | Integrated | Observation |
| --- | ---: | ---: | --- |
| Starting-board copy, median | 177.15 µs | 1.20 µs | 147× faster CPU operation |
| Search, 2 blocks / 64 filters, median | 191.07 ms | 62.84 ms | 3.04× faster |
| Search, 19 blocks / 256 filters, median | 229.05 ms | 98.10 ms | 2.33× faster |
| Default-network search GPU peak, parent | 651.27 MB | 170.70 MB | 74% lower |
| Default-network materialized boards, median | 610 | 33 | Lazy materialization avoids most copies |
| Three fresh training generations, elapsed | 61.74 s | 133.32 s | Initial candidate is slower overall |
| Training positions / completed games | 1,253 / 6 | 522 / 6 | Different self-play trajectories |
| Adam updates over those generations | 21 | 3 | Candidate updates once per generation |

Search uses 32 requested simulations at three fixed openings, repeated three
times. Equal simulation settings do **not** imply equal useful work: main's
default-network median root/child visit counts were 54/74; the corrected search
recorded 32/32. Median network positions evaluated were 22 versus 33. Search
latency includes the combined fixes and cannot isolate one optimization's effect.

Training uses the smaller real ChessNet (10,261,377 parameters), two parallel
complete games per generation, three generations, and microbatches of 64. There
is no replay, random sampling, or artificial game-length cutoff. Adam uses each
branch's default learning rate: 0.2 versus 0.001. Evaluation is disabled in this
short throughput experiment; checkpoint/evaluation behavior is covered by tests.

The candidate's learning passes took 51.80, 22.65 and 22.16 seconds, despite only
215, 148 and 159 training positions. Its final microbatch sizes differed each
generation (23, 20, 31), suggesting shape-specific gradient compilation overhead.
The isolated compilation probe above confirms this overhead.

Main's policy loss reached 719.7 after the first update and its final starting
position policy entropy was effectively zero. The candidate's generation policy
losses were 8.45, 8.17 and 7.77, with final entropy 8.44; both retained finite
parameters. This is a useful instability signal, but different targets and update
frequencies prevent interpreting these losses as a controlled strength comparison.
The Adam and promotion PRs were subsequently merged into main during this work,
so this instability observation describes the original baseline, not current main.

Parent-process memory peaks exclude self-play workers. This is one short run with
one initial model seed and nondeterministic worker trajectories. Repeat with
multiple seeds and alternate execution order before relying on precise ratios.
NVIDIA measurements do not predict Apple Silicon/Metal performance.

## Full-model learning follow-up

The [longer learning experiment](LEARNING.md) used the original 19-block / 256-filter network. After 21 updates, its score was 29.69% against its initial model and 39.13% against random legal moves among completed games (nine censored). This did not demonstrate convergence. These learning results include the exact cache-key fix; the earlier performance comparisons above predate it.
