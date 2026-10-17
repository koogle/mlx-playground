# Zero-value search-budget mechanism

Identical untrained weights at256simulation cap lost to64:6W8D18L31.25%,
White4W4D8L37.5%,Black2W4D10L25%. All32complete, weights frozen,
426585candidate simulations versus107136baseline. This is before training.

Repeat exact32game seed202016 bank with nonterminal neural values set to zero for
BOTH players. Keep policy probabilities, weights, exact terminal outcomes, cap256/64,
early stopping, independent spatialBN/fixedshape16 unchanged. Early-stop behavior
can change as a consequence of zero values; actual simulation counts recorded.
GPU preflight asserts exact policy equality, zero output values, unchanged weights.
No optimizer updates. This contrasts neural-value input, not a trained model fix.
Compare per-color results and censoring; reused bank is mechanism/discovery, and
any strength claim needs independent confirmation. Research branch only; PR29 untouched.
