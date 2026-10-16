# Controlled first-generation learning comparisons

One bounded L40S job generates 40 complete games from the same original epoch-0
checkpoint: 32 training games and 8 separate validation games. All full states,
visit distributions, outcomes and move histories are preserved. Legal moves and
results are independently checked. No match-game data enters training.

All training arms start with identical weights and step-zero Adam, use the same
positions exactly once and have the same update count. The reference is the
current joint policy/value objective, chronological batches of 64 and LR 1e-4.
Each alternative changes only one factor: shuffle positions, remove value loss,
remove policy loss, restore probabilities-as-logits policy loss, or enable Adam
bias correction. The legacy formula is an experimental control, not a production
rollback. Policy-only training still changes the shared trunk and can therefore
change value predictions.

Save every arm at updates 1, 8, 32 and its final update, including Adam state.
Probe the same validation positions at those points with batch statistics and
without mutating weights or stored normalization statistics. These probes are
descriptive; their losses are not chess-strength measurements.

Play 16 discovery games per arm against the initial model on the same eight
paired openings, using 64 simulations. Include an initial-versus-initial control
and a frozen zero-value intervention on the joint-trained model. All matches use
current-batch normalization; prediction caching is disabled. All model tensors
must remain identical before/after each match. Censored games are reported
separately. The discovery bank does not establish a confirmed improvement;
any proposed fix needs a separate held-out match and sustained learning check.

The remote workload is bounded to 4.5 hours, with a five-hour Modal timeout and
no automatic retry. No long continuation is automatically launched from the
discovery results. `launch.json` identifies the actual job.
