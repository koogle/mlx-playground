# Historical training regression audit

## State and scope

The user stopped `pr29-gated-rerun-20261016-060137`. Its call
`fc-01M51MTFB808C8C964QY5HB1AF` was cancelled with container termination, its
monitor disabled, and its last committed progress, logs, training histories and
sampled tensors preserved. The SDK subsequently reports that the call was
cancelled. A cancellation marker is persisted in the same Modal volume directory.
This was deliberate cancellation, not a runtime training failure.

At cancellation: **256 complete training games, 43,076 positions, 677 attempted
minibatch updates, eight rejected candidates, zero retained updates**. The
accepted model is still epoch 0. The eight candidate scores against it were
10%, 17.5%, 16.25%, 13.75%, 17.5%, 26.25%, 16.25%, 6.25%. These are eight independent
retries from the initial model, not checkpoints along a 677-step training path.

The source baseline is `fc51d26`, immediately before the October 13 changes.
It is the pre-change source, **not a verified copy of the user's successful
trained checkpoint**. No historical learned tensors are in that Git tree, the
two workspaces, or the available ChessZero convergence volume (created October 13).
The successful checkpoint, its actual saved settings and original runtime are
still needed for a direct model-to-model comparison. Checked-in defaults alone
must not be substituted for those settings.

No training jobs were launched during this audit. The reproductions below do
not apply optimizer updates or run search games. They compare actual source
functions and loss derivatives on saved data using MLX 0.21.1 CPU.

## What the latest evidence establishes

All 256 saved training histories replay legally in python-chess and their final
outcomes agree. All 2,042 recorded sample targets agree with the actual winner
and side to move; the 512 targets in the downloaded first/eighth-generation NPZ
files also agree. The original runtime additionally checked all 19 input planes
at those bounded sample locations. There were 198 checkmates, 43 insufficient-
material draws, 10 fifty-move draws, 4 stalemates and 1 repetition draw. This failed
rerun was **not** the previous long run's 96%-draw data collapse.

Thus the latest failure is not explained by flipped White/Black result labels,
incomplete games silently becoming draws, or no optimizer steps being applied.
Candidates really trained, then lost their promotion matches. This does not
exclude defects outside the checked paths or identify the weight-level cause
of those losses.

### 1. The loss change was a large change to the learning problem

PR #4 corrected two genuine mathematical bugs together:

- The old training loss fed `softmax(logits)` into a function expecting logits.
  The current loss feeds logits directly. With uniform logits and 64 actual saved
  policy targets, the measured logit-gradient norm changes from 0.0000093204 to
  0.0435414: approximately **4,672 times larger**, matching the output dimension.
  Adam rescales gradients; this is **not** a 4,672-fold parameter-update claim.
  Relative policy/value/shared-trunk gradients, epsilon sensitivity and the
  optimization problem nevertheless change. Calling it performance-neutral
  without validating the resulting recipe was unjustified.
- Old self-play inserted every identical history twice with opposite results.
  Running the actual old batcher on a Black win yields `[-1,+1,-1,+1]` and then
  `[+1,-1,+1,-1]`. The corrected batcher emits the first sequence once. For
  identical states, the mean old MSE objective is `v² + z²`, minimized at zero;
  the new one can learn nonzero outcomes. Reverting to contradictory labels is
  not a principled fix for the newly harmful value signal.

There is an important qualification: **the old loop also recycled candidate-
versus-best evaluation histories into subsequent training once, with correctly
converted White-relative results**. Those histories did not have the contradictory
duplication. The old complete system therefore did not universally train value
to zero. PR #20 removed that source of correctly signed opponent-game data as well
as changing update frequency. That changed the data mix and learning curriculum.

The earlier frozen interventions prove that learned nonterminal values harmed
the epoch 10 checkpoints on the tested banks. They do not identify why the
optimizer learned those values, prove all trained values are harmful, or explain
epoch 106's nearly-zero sampled values by the same mechanism.

### 2. Restoring minibatches did not restore the old training schedule

The previous long experiment's generation-wide averaging was not equivalent to
the old per-minibatch loop. The rerun did restore separate minibatch steps, but
introduced further differences:

| Behavior | Pre-change source | Stopped gated rerun |
| --- | --- | --- |
| First accepted checkpoint | First evaluation at generation 5 accepts current model without a match | Untrained epoch 0 accepted before any update |
| Candidate evaluation interval | Every5 generations | After every 32-game generation |
| Candidate between evaluations | Continues learning across generations | Reverted after every failed candidate |
| State on rejection | Revert weights, retain Adam moments/step | Restore accepted weights **and** Adam moments/step |
| Self-play teacher | Current candidate | Accepted checkpoint only |
| Promotion measure | Wins / decisive games | Wins plus half draws / all games |
| Evaluation data in next training | Yes | No |

In the rerun, every rejection returned Adam to step 0. Every candidate received
only 65–105 updates on one32-game dataset. The old process gave the candidate an
initial five-generation learning period, then longer periods between gates and
retained optimizer history. The rerun therefore does **not** establish that
sustained minibatch training necessarily follows the previous long run's decline.
It establishes severe early candidate weakness and failure to leave epoch 0 under
this new gate. Rejection protected the teacher; it did not fix candidate learning.

Uncorrected Adam is used by default in both versions. Repeated resets reproduce
its initial-moment transient, whereas retaining moments follows a different
update trajectory. The formerly cancelled first-step comparison did not prove
bias correction as a cause, and the shared default is not itself a new bug.

### 3. Search and inference changed substantially

The old network topology is still19 residual blocks /256 filters with the same
policy/value heads. Ordinary move indices and absolute-color input ordering are
unchanged. However, the function used to play games changed:

- Old MCTS ordinarily ran the network in **training mode**, using current-batch
  normalization; the new pipeline always searches in evaluation mode using
  running statistics. PR #10 also changed initial running variance from 0 to 1.
  Old and new play can differ with identical learned weights. Previous stats-
  only controls did not reliably fix the full-game regression, so this is a
  material difference, not an established sole cause.
- PR #19 fixes a real backup error: for a Black leaf with value+0.75 and White
  parent, old backup gives `[+0.75,-0.75]`; current backup correctly gives
  `[-0.75,+0.75]`. Do not flip the new labels to accommodate the old search bug.
- PR #8 replaced buffered traversals with sequential feedback after each simulation.
  The old batch selected paths before updating their visits, deduplicated leaves
  for evaluation, then reused those paths in backups. A numeric simulation count
  is not a comparable measure of search work across these implementations.
- The pre-change default was 1,600 simulations; main later defaulted to 256; our
  experiments explicitly used 64. Old source default LR was 0.2, then main used
  0.001 and the experiments 0.0001. The historical successful runtime values are
  unknown. These source/default differences are not a matched learning experiment.
- Old self-play actually sampled temperature 1 throughout. The newer worker uses
  0.5 after ply 30. Root noise changed from a transient first-selection perturbation
  to persistent noise throughout each search. Search subtrees and inherited visits
  are now retained between moves, which also changes generated policy targets.

The runtime changed too: the old lockfile pins MLX 0.21.1 for macOS; the experiment
uses MLX 0.32.3 CUDA. Existing CPU checks and successful serialization do not prove
cross-backend equality of full-network gradients. No backend gradient fault has
been demonstrated here; it remains an uncontrolled variable.

### 4. Old strength measurements are not directly interchangeable with the new ones

Promotion used wins divided by decisive games, then PR #11 changed it to wins
divided by all games, and the latest patch changed it to draw-half score. For
6W/30D/4L, those measures are 60%, 15%, 52.5%. The same model is accepted by the
original 55% rule and rejected by the latest one. This does not invalidate the
user's observation of improved play, but historical promotion numbers alone
cannot establish the same strength improvement measured by current matches.

I also reproduced old **cross-model tree sharing**: expanding the same board for
two models with deliberately different policy priors returns the identical
children dictionary in the old MCTS. Both expansions retain the first model's
preferred move. The current model-scoped caches produce distinct dictionaries
and the two intended moves. Candidate/best evaluation can visit overlapping
positions within a game, so the historical implementation could contaminate the
opponent's search. This is a reason to preserve the cache isolation fix and use
independent frozen-position comparisons when recovering the old checkpoint;
it is not proof the user's historical model never learned.

### 5. Game rules and input semantics changed

PR #13 added real castling/en-passant execution, full king-safety validation,
correct material adjudication, correct 100-ply fifty-move thresholds and counters.
PR #5 / PR #29 corrected repetition, including real false-draw hash collisions found
during evaluation. On the same eight-ply knight repetition, the old board does
not declare a draw and the current one correctly does.

Plane 18 changed from halfmoves / 50 to halfmoves / 100; plane 17 no longer clips at 1.
These retain tensor shapes but change input meaning/range. Any old learned
checkpoint comparison must account for them. This is **not** an explanation for
fresh random initialization failing by itself. Underpromotion output indices
also became distinct while ordinary/queen-promotion indices stayed unchanged.

## Complete production-change ledger

All 21 first-parent commits affecting production Python since `fc51d26` are listed
below, in order. Exact hashes and file lists are in [production-commits.json](production-commits.json).
Experiment-only commits are covered separately below; test/documentation-only
changes do not alter production inference or optimization.

| Commit / PR | What changed and why | Behavioral assessment |
| --- | --- | --- |
| `6a334ec` /4 | Fix contradictory targets and probability-as-logit CE | Correct mathematics; major objective/gradient change reproduced above |
| `56accd0` /17 | Replace target arithmetic with explicit conditional | Equivalent for the actual0/1 side-to-move plane; no new sign convention |
| `5dbfb06` /6 | Share encoder, separate underpromotions | Correctness fix; ordinary moves unchanged; changes rare promotion targets |
| `e754eec` /7 | Adam default 0.2→0.001 | Hyperparameter change, not a semantics-preserving fix |
| `a8acb25` /19 | Back up relative to actual leaf color | Corrects demonstrated old odd-depth sign bug |
| `272567e` /8 | Sequential search feedback; eight workers | Changes search trajectory and policy targets, plus concurrency |
| `a3d4205` /5 | Repetition identities/counts and history-aware draws | Corrects rule targets; changes when games end |
| `ddf3875` /9 | Bound/scoped caches and exact position keys | Prevents collisions and shared-tree/model contamination; important correctness change |
| `63ce398` /10 | Search in eval mode; initialize BN variance 1 | Changes neural inference function and dependence on normalization statistics |
| `565f631` /20 | Generation accumulation; remove evaluation-game training | Changes optimizer frequency, normalization/update schedule and data curriculum |
| `1797062` /11 | Full Adam/best snapshots, rollback, score/eval changes; MLX update | Useful persistence fixes bundled with algorithm, metric and runtime changes; legacy loader no longer supported |
| `2e5df7c` /13 | Special moves, king safety, draw rules/counters | Broad genuine rule corrections plus checkpoint input-semantics change |
| `f2015f3` /14 | Fast copies and lazy child boards | Performance intent; preserves position/history ownership through covered paths |
| `fa5539f` /21 | Reliable worker collection; persistent root noise; temperature schedule | Failure accounting fix plus exploration/data-distribution changes |
| `6ab1e6f` /25 | Reuse evaluations and played subtrees; local search noise | Performance benefit plus inherited-visit/search-target behavior changes |
| `525a988` /27 | Cap per-model CUDA workers to 1 | Resource/failure avoidance; no intended target or optimizer change |
| `042477a` /28 | Default simulations 1,600→256 | Search-budget/teacher change; experiments later use 64 |
| `19251a0` /29 prototype | Parent-owned batched pipe inference, NumPy handoff | Ownership/batching change; no explicit value sign or policy-index permutation |
| `c8b75a0` /29 completion | Both eval models in parent, exact repetition keys, streamed minibatches, robust CLI | Fixes observed false draws/worker failures; moves batch allocation without introducing label conversion |
| `d9aaa15` | Restore minibatches, gate every generation, initial accepted teacher, half-draw metric | Does not restore historical warmup, teacher, Adam rollback or promotion semantics |
| `6bad942` | Common promotion helper; persist training moves; gated experiment | Observability improvement; run uses reset-on-rejection protocol described above |

The experiment-only sequence added fresh initialization seeds and lower LR/search
settings, continued epoch 10 for 3,000 games without rejection, ran frozen value/
normalization interventions, then introduced the latest gated restart. Those
drivers determine the actual schedules; production defaults alone do not describe
them. No later source-only diagnostic commit changes these measured outcomes.

## What this does and does not explain

The evidence identifies **multiple algorithm changes bundled with bug fixes**,
a confirmed harmful learned-value contribution in earlier checkpoints, the
long run's draw collapse, and an overly early reset loop in the latest rerun.
It rules against the simple current label-flip hypothesis on the saved data.
It does **not** isolate a single changed commit as the cause of all candidate
weakness, establish a CUDA gradient bug, or compare the unavailable historical
learned tensors. The first eight rejected candidate weights were also not saved:
the harness retained pre-rejection candidates only at 500-game milestones. They
cannot now be examined as if they still existed. Saved training moves and bounded
samples do not retain every policy target needed to replay identical updates.

The earlier claim that update frequency was the leading explanation should be
treated as a hypothesis, not a diagnosis established by the rerun.

Before any further training, recover the actual successful checkpoint/config,
keep an isolated exact-source baseline, and compare identical frozen positions
under old/new encodings and inference modes. Preserve the genuine label, backup,
rule and cache fixes as individually testable changes. A subsequent learning
comparison must explicitly control the loss, normalization mode, runtime, data
mix, update count, warmup and rejection policy; save rejected candidates before
discarding them. No such new training comparison has been launched or queued.

## Reproducing these mechanics checks

The local evidence root is `/workspace/scratch/pr29-historical-audit`. It contains
`git archive fc51d26 chesszero` under `baseline`, the saved first-generation NPZ
targets, and cancelled-run data. Run `compare_mechanics.py EVIDENCE_ROOT` with
`PYTHONPATH=chesszero` using the existing MLX 0.21.1 CPU environment. It imports the
archived source and compares batchers, loss derivatives, backup, encoding,
normalization initialization and cache ownership. It performs no optimization.

[Mechanics measurements](mechanics-results.json) ·
[Training-history verification](training-evidence-audit.json) ·
[Deliberate cancellation](cancellation.json).
