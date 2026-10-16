# Frozen policy mechanism comparison

The completed independent-normalization study restored identical-model score to 50%, eight of eight paired move histories identical, and zero fixed-shape prediction error. All 128 games completed, fingerprints unchanged. Joint training still scored 9.375%, policy-only 21.875%, value-only 40.625%, legacy loss 12.5%, corrected Adam 9.375%, shuffled 6.25%, and joint with zero neural value 12.5% against the initial model.

These discovery results do not establish a successful training fix. They motivate three frozen interventions using the same eight opening pairs, 64 simulations, independent spatial normalization and padded batch size 16:

1. Joint trained value with original initial policy.
2. Joint trained policy with original initial value.
3. Policy-only checkpoint with policy probabilities raised to 1/4 and renormalized (temperature 4); its value unchanged.

All opponents remain the unchanged initial model. No optimizer updates, new training data, or checkpoint modification. Source fingerprints and inference immutability are asserted. Each arm is compared with its completed unchanged arm. Head substitution isolates output signals, not the cause of weight changes; flattening tests confidence at this search budget, not whether targets are correct. Full move histories, censored counts, and per-color WDL are retained. Any apparent strength gain requires independent confirmation before sustained self-play validation.
