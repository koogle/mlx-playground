# Fresh-seed minibatch and rejection rerun

The original run was stopped deliberately. The current source additionally
recycles completed candidate-versus-best promotion games into the next
generation, including rejected candidates' games. Both players supply MCTS
visit-policy targets; random opening plies supply none. Outcomes are converted
from candidate-relative to White-relative before batching produces side-to-move
targets. Each position is used once. Pending games survive checkpoint/resume.
The CLI trainer and this driver share this behavior; no replacement run has
been launched as part of this change.

Start from the exact untrained model and Adam state at epoch0 of
`pr29-random-restart-20261015-085706`. Train for **3,000 total self-play games**,
rather than extending the collapsed checkpoint. Keep the original architecture,
seed20261015, 64 MCTS simulations, learning rate0.0001 and minibatch64. This is
a comparison of the combined minibatch-update and rejection changes, not an
ablation isolating either change individually.

Each generation has up to32 fresh complete games. Self-play uses the accepted
checkpoint. Apply one Adam update per minibatch and run the production
`Trainer.consider_candidate` rule after every generation: 40 balanced games
from paired seeded eight-ply openings, deterministic search without root noise,
strictly greater than55% score to promote, draws half. Rejection restores the
accepted model, BN statistics and Adam state. Match results are noisy estimates;
acceptance does not guarantee monotonic strength against random or other models.

Every500 games (fifteen32-game batches and one20-game batch), evaluate both the
latest candidate and the accepted model on the previous fixed64-game random
bank. Reports distinguish attempted updates from accepted optimizer steps,
promotions/rejections, and White/Black WDL and scores. Reuse the candidate result
when it was promoted. Random-opponent and final baseline-match data never train
the model; promotion-match games do. At completion,
play64 direct games against the frozen initial model on the same paired opening
bank used by the previous experiment, and verify saved model/teacher/Adam tensors.

Persist every training game's moves and independently replay them with
python-chess, validating legal moves, side to move, terminal outcome and policy
normalization. Verify all19 state planes at up to8 positions per game and save
their state/policy/value targets in compressed arrays. The production engine
and search choose moves; the oracle checks results after generation.

Checkpoints and progress are published after the gate. Retain epoch0, the latest
two checkpoints, and every500-game checkpoint. Also retain pre-gate candidate
weights and Adam at milestones so rejected candidates can be inspected. Resume
only after confirming no other worker is active. Incomplete-generation work is
not counted; missing post-checkpoint milestone reports are completed on resume.

One L40S,32 CPU cores,64–128GiB RAM. The worker has a22-hour workload limit and
23-hour Modal timeout; it has no automatic retry. Persisted checkpoints survive
interruption. Timing is uncertain because promotion matches add substantial work.

```sh
/workspace/setup/modal/.venv/bin/python /workspace/setup/modal/run.py run --detach experiments/pr29-gated-rerun/modal_train.py
/workspace/setup/modal/.venv/bin/python experiments/pr29-gated-rerun/status.py
```

The launcher records the run/call/volume/source revision in ignored `launch.json`.
Use `--resume-run RUN_ID` for an explicitly needed continuation of a stopped run.
Status reads only persisted Modal volume data and allocates no GPU.

Launched2026-10-16 at06:01UTC: `pr29-gated-rerun-20261016-060137`,
[Modal app](https://modal.com/apps/radiantai/main/ap-aXIFYbVH385FsITR1WsM7i),
call `fc-01M51MTFB808C8C964QY5HB1AF`, source `6bad942`.
Startup verified the restored epoch0/Adam step0 and began the64-game initial
evaluation. Those startup checks are not completed training or strength results.
