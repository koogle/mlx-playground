# Fresh-seed minibatch and rejection rerun

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
when it was promoted. Evaluation data never trains the model. At completion,
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
