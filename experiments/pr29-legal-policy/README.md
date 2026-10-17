# Legal-action policy objective experiment

Hypothesis: global policy cross entropy may improve chiefly by suppressing illegal
outputs while worsening the relative legal preferences consumed by search. This is
an objective intervention, not a demonstrated encoding bug. MCTS already normalizes
its priors over legal moves; no production defaults change.

One new arm from the exact initial step0 checkpoint, using the same saved 32 games
(4935 positions), order, batch64, 78 Adam updates without bias correction at 1e-4,
ordinary training BatchNorm, value MSE and L2. Only policy CE changes: logits for
illegal moves are replaced by -1e9 before CE; the visit-distribution targets remain
unchanged. Exact legal masks come from python-chess replay of all saved UCI moves,
including underpromotions. Every position's 19 planes, value label, target support,
played move and final result must agree before training. Full masks are retained.

Reference is the saved batch_training step78 arm of pr29-training-normalization-
20261016-210929: same protocol and seed162016 discovery bank, 0W2D30L (3.125%).
Only the intervention is rerun; no duplicate control GPU workload. Checkpoints at
1,8,32,78 and optimizer state retained. Effective LR set AFTER optimizer restore
and asserted each batch. Search uses independent spatial BN, padded shape16,
64 simulations. Frozen candidate and initial fingerprints checked across 32 games.
This reused bank is discovery; any apparent strength gain needs fresh confirmation.

Local test verifies conditional legal CE and gradients against a two-action oracle,
zero illegal-logit gradients, and invariance to arbitrarily changed illegal logits.
On-device full replay is mandatory; failures must stop before training or matches.
