# Restored training experiment

Fresh run from the exact untrained weights and Adam state of
`pr29-random-restart-20261015-085706` epoch 0. Includes both requested changes:

- Recycle completed candidate-versus-best games once into the next generation,
  including rejected candidates; preserve queued games in checkpoints.
- Use current-batch BatchNorm statistics during search. Neural prediction caching
  is disabled and search preserves saved running statistics.

Keep visit-distribution targets and raw-logit cross-entropy. Use the accepted
checkpoint for self-play, 64 simulations, minibatches of 64, Adam learning rate
0.0001, and 40-game promotion matches after each up-to-32 fresh games. Promotion
requires a score above 55%, draws half; rejection restores model and Adam.

Target 3,000 fresh self-play games, with recycled promotion games counted
separately. Report candidate and accepted-model White/Black results every 500
fresh games; count censored games separately. Evaluate the initial model again
under the new normalization mode. Finish with checkpoint verification and a
64-game direct match against the same initial model using that mode.

One L40S, 32 CPUs, 22-hour workload limit, no automatic retry. Persist artifacts
in `chesszero-convergence`. `launch.json` identifies the actual run and source.
The previous cancelled run and its artifacts remain unchanged.
