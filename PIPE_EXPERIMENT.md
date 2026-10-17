# Parent-process batched inference

Self-play and evaluation workers send NumPy positions over direct multiprocessing
pipes. The parent batches neural inference and keeps candidate and accepted-model
requests and prediction caches separate. CPU workers do not initialize MLX/CUDA.
Direct local MCTS calls preserve the caller's model mode.

Training streams minibatches and checkpoints model, optimizer and accepted-model
state. Self-play uses the accepted model; candidates are evaluated each generation
by default and promoted above a 55% match score, with draws worth half. Rejection
restores accepted weights, BatchNorm statistics and optimizer state. Completed
promotion games are recycled once into the next training epoch and survive resume;
random-opponent evaluation games are excluded from training. Incomplete matches
cannot promote a candidate or supply recycled training data.

Search uses current-batch BatchNorm by default. Opt-in independent spatial
normalization and fixed inference batch padding support reproducible comparisons.
Prediction caching is disabled when predictions depend on batch companions.
Repetition detection stores exact position keys to prevent false draws from hash
collisions. Regression tests cover these behaviors and scoring from both colors.

Run from the repository root:

```sh
python chesszero/train.py --workers 32 --games-per-iteration 32 \
  --epochs 10 --simulations 64 --batch-size 64 --learning-rate 0.0001 \
  --checkpoint-dir checkpoints --evaluate-random 64
```

This is an operational example, not a validated strength-improving recipe.
Controlled studies have **not established sustained improvement over the initial
model**, and color imbalance remains unresolved. This PR makes no convergence claim.
The experimental legal-move policy loss is not part of the implementation changes.

Experiment drivers, launch records and exact game histories are preserved on the
[research archive branch](https://github.com/koogle/mlx-playground/tree/archive/pr29-training-research-20261017/experiments).
They are excluded from this PR's implementation diff; local monitor files and Modal
artifacts remain available.
