# Frozen update trajectory

Training normalization test completed: ordinary batch0W2D30L3.125%, independent spatial0W3D29L4.6875%,64games complete. Matching normalization did not fix regression. Both arms reduced validation policy CE but playing strength collapsed.

Evaluate already-saved ordinary-training checkpoints1,8,32 against the initial model on the identical32-game seed162016 bank; step78 reference already complete. No optimizer updates or new training data. Same independent spatial inference, fixedshape16 and64simulations. Assert restored Adam steps and frozen fingerprints. Retain all moves/per-color outcomes. This locates onset of damage without changing losses, data, normalization, search or learningrate. It is discovery, not checkpoint selection or confirmedstrength. Use evidence to select the smallest next training intervention, then held-out and sustained-selfplay validation.
