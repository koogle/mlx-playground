# Audit of full-game learning regression

This audit compares PR29 with main (`042477a`) **and** the source immediately
before the recent sequence of changes (`fc51d26`). Main already contains most of
the algorithm changes; PR29 versus main alone is an insufficient comparison.
The successful historical run's actual checkpoint/configuration has not yet
been identified. Old source defaults are not evidence of its runtime settings.

The completed run is `pr29-3000-games-20261015-145659`. All 33 persisted Python
source hashes match the current production files. This audit uses read-only
Modal artifact downloads, frozen CPU inference, independent saved-game replay,
and 88 existing local tests. It launches no Modal jobs, generates no new games,
and performs no optimizer updates. [Measured evidence](evidence.json).

## Confirmed behavior

### We changed the learning algorithm, not just its implementation

Commit `565f631` (PR20, already on main) replaced one Adam update per minibatch
with one position-weighted accumulated update for a whole generation. The
completed continuation processed 515,113 positions with 96 updates. The same
game batches would give 8,094 updates with per-64-position minibatch updates,
or 555 with per-1,024-position updates (ignoring the historical duplicated
histories and evaluation data). These are counts, not claims that either
counterfactual would improve chess strength. Averaging gradients once is not
equivalent to sequential Adam updates; the model and moments change between
sequential updates.

The older checked-in defaults were LR 0.2 and 1,600 simulations; main defaults
are LR 0.001 and 256 simulations; the completed experiment used LR 0.0001 and
64 simulations. The old worker always sampled with temperature 1.0; the current
worker uses 0.5 after ply 30 and reuses search subtrees. The experiment also
bypassed `Trainer.train`'s accept/revert logic and used the latest checkpoint
for subsequent self-play. None of these changes should be described as a
controlled replication of the historical learning procedure.

The accumulation arithmetic passes the existing split-versus-full-batch test
without BatchNorm. With BatchNorm, each chronological 64-position microbatch
uses separate statistics and updates the running statistics; it is explicitly
not a single giant normalized batch. The running statistics can change roughly
80 times while learned weights change once. We have not established that this
schedule preserves useful learning.

Adam is also instantiated without an explicit `bias_correction` argument. The
installed API defaults it to false, as did the previous code. That is not a
new sign bug, but makes the changed step count especially relevant: the early
step scaling differs from conventional bias-corrected Adam. The previously
cancelled first-update experiment did not establish it as a cause; no such
claim follows from the default alone.

### The fresh self-play distribution collapsed into draws

| Additional games | Draws in preceding 500 | Mean plies | Mean value loss per update |
| --- | ---: | ---: | ---: |
| 500 | 50 (10.0%) | 162.7 | 0.7700 |
| 1,000 | 34 (6.8%) | 174.8 | 0.8466 |
| 1,500 | 34 (6.8%) | 200.0 | 0.8525 |
| 2,000 | 78 (15.6%) | 225.7 | 0.8115 |
| 2,500 | 354 (70.8%) | 191.5 | 0.3380 |
| 3,000 | 482 (96.4%) | 75.5 | 0.0350 |

The last two updates contain only draws; their value losses are approximately
0.0000195 and 0.0000109. Near-zero value loss then means fitting near-zero draw
targets. It is not evidence of stronger position evaluation.

Latest-model self-play, no replay, and one pass over fresh data allow these new
draw-heavy games to replace the earlier learning signal. Lack of rejection
allows the weaker model to remain the teacher. These establish the observed
feedback pattern; they do not isolate which change first caused it.

### Final losses and draws are real

All 128 saved final evaluation histories were independently replayed against
python-chess. Every move is legal and every recorded terminal result agrees.
The final direct match has 42 checkmate losses and 22 repetition draws. The
random-opponent evaluation has 63 terminal results and one censored game.
This is not an evaluation-side swap of wins and losses.

Frozen CPU probes at 28 identical original/mirrored positions show final values
near zero (mean 0.00114, standard deviation 0.00249). Mean policy entropy among
legal moves falls from 3.011 to 2.208 nats. At the initial board, policy mass on
legal moves rises from 0.43% to 73.08%. Thus the model has learned legality and
more concentrated move preferences while its playing strength regressed.
These selected positions are descriptive, not a representative value-accuracy
benchmark or a causal head intervention. Some positions repeat.
These local probes use MLX 0.21.1 CPU, while the saved training used MLX 0.32.3
CUDA; they are not a bit-for-bit replication of GPU inference.

The earlier epoch-10 frozen interventions established a harmful color-dependent
value contribution at that checkpoint. They do not establish that the same
mechanism explains epoch106, whose sampled values are nearly zero.

## Label and search audit

The current chain is consistent:

1. The worker stores the state **before** making the move, with a copied array.
2. It returns one terminal outcome from White's perspective.
3. Plane12 is one for White to move and zero for Black. The batcher produces
   `white_result` for White and `-white_result` for Black, once per position.
4. Network values and node values are side-to-move values. Backups use the leaf's
   actual color. Parent selection uses the negative of its child's value.
5. Search priors and policy targets use the same move encoder; no board rotation
   or policy permutation is inserted by the pipe handoff.

The local target, mirror, backup and accumulation tests passed: **88 tests**.
This covers forced wins for both colors and both move parities. It is evidence
against the suspected simple sign flip, not proof that all learning code is
correct. Self-play histories/targets were not persisted in the long run, so we
cannot retrospectively replay every training game or verify every sample.

The older label duplication was a real bug: it trained each identical state
with both `z` and `-z`. Their average MSE is `v² + z²`, minimized at `v=0`.
Removing that duplication (PR4) enables a nonzero learned value signal. PR4 also
replaced cross-entropy applied to probabilities with cross-entropy applied to
logits. PR19/PR8 corrected leaf-perspective backup and changed search execution.
These corrections alter optimization/search behavior; their individual unit
tests do not establish that the combined training recipe improves strength.

## What remains unproven and what should change next

We have **not** identified a newly introduced sign flip or established one commit
as the sole cause. The strongest regression candidate to isolate is the changed
optimizer granularity, alongside the changed data/search distribution. A clean
comparison needs the last successful run's actual settings and checkpoint.

Before another long run:

- Make per-minibatch updates versus generation accumulation an explicit choice,
  with optimizer-step counts reported separately from games and epochs. Restore
  the historical schedule as a comparison, not as an assumed cure.
- Compare changes individually with fixed data and initial weights, measuring
  raw value/policy learning and direct playing strength. Include the accepted
  teacher versus latest-model distinction explicitly.
- Persist a bounded sample of training histories, outcomes, targets, policy
  entropies and draw reasons, so scoring failures and draw collapse can be
  distinguished from the checkpoint alone.
- Track training- versus evaluation-mode value accuracy on held-out self-play
  positions before blaming BatchNorm or refreshing statistics. Prior stats-only
  controls did not reliably fix full-game strength.

There is also a separate scoring inconsistency: production best-model promotion
uses wins/all-games, giving draws zero credit, while experiment strength scores
give draws half. This path was bypassed in the completed run and therefore did
not cause this run's regression. Restoring promotion needs an explicit scoring
decision rather than silently assuming the two metrics agree.

## Repeating the frozen local inspection

`inspect_saved_run.py ARTIFACT_ROOT` reads the two downloaded model files and
their epoch0/epoch106 manifests, plus the final evaluation histories, under
their original run-directory names. It requires the project and tests on
`PYTHONPATH` (the test helper converts python-chess FENs to native tensors).
It writes `probe-results.json` beneath that artifact root. It performs no
network calls, games, gradient calculations or optimizer updates. The local
artifact root used here was `/workspace/scratch/pr29-regression-audit`.
