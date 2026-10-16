# Parent-process batched inference

CPU game workers send NumPy positions over direct multiprocessing pipes. The parent owns the network, batches up to 64 unique ready cache misses, and performs learning after all games complete. Self-play and model-versus-model evaluation use this path; evaluation tags each request with its model index and keeps predictions from different networks separate. Evaluation has no search noise.

The CLI imports Trainer/MLX only inside main(), so spawned CPU workers can re-import the entrypoint with CUDA hidden. Direct interactive MCTS calls still convert NumPy inputs to MLX arrays and preserve the caller's training/eval mode. Training streams minibatches, applies an Adam update after each minibatch, and saves a resumable checkpoint after each epoch. `--accumulate-gradients` explicitly opts into the older one-update-per-generation experiment. A failed minibatch restores the generation's original model and Adam state.

Self-play uses the accepted checkpoint, initialized before the first training update. The candidate is evaluated after every generation by default (`--eval-interval 1`). Promotion requires a score strictly above 55%, with draws worth half, in balanced matches from paired seeded eight-ply legal openings. Search after the opening is deterministic and has no root noise. Rejection, including an incomplete evaluation, restores accepted weights, BatchNorm statistics and Adam moments/step. Changing the evaluation interval never makes an unaccepted candidate the self-play teacher. This is a finite-sample promotion rule, not a guarantee of the strongest possible checkpoint.

For the previous run's minibatch size and learning rate, use `--batch-size 64 --learning-rate 0.0001`. This changes the optimizer update count substantially; reported game counts, optimizer steps, and acceptance decisions should be tracked separately. Historical experiment drivers under `experiments/pr29-first-result`, `pr29-random-restart`, and `pr29-3000-games` bypass the production training loop and are not the entrypoints for this revised protocol.

Repetition history retains exact immutable position keys rather than reduced integer digests. A real Modal random-game regression collapsed three unrelated positions (plies 440, 462, and 500) into one digest and incorrectly declared a draw at ply 500. The 500-ply fixture and a forced hash-collision test cover this failure. The independent validator remains enabled in the Modal experiment.

The original 19-block/256-filter architecture and 256-simulation CLI default remain. Example bounded training and random-opponent evaluation, from the repository root:

```sh
python chesszero/train.py --workers 32 --games-per-iteration 32 \
  --epochs 10 --simulations 64 --batch-size 64 --learning-rate 0.0001 \
  --checkpoint-dir checkpoints --evaluate-random 64
```

`--epochs` is a total target, including completed epochs when using `--resume`. `--eval-interval` and `--evaluation-games` control existing best-model acceptance. Best-model promotion retains its existing wins/total threshold; the random-opponent report scores draws as half a point. Neither evaluation feeds training. Game counts and positive training settings are validated by argparse.

Game/inference counters are logged every 15 seconds, learning logs microbatch progress, and the Modal wrapper streams these logs into the app. Experiment details and launch/status commands are in `experiments/pr29-first-result/README.md`.

The earlier paired L40S benchmark measured 512 games at 64 simulations: 17.03 games/min for direct pipes versus 14.33 for the RPC implementation. It predates the integration and repetition corrections here; it is not a new throughput or convergence measurement.
