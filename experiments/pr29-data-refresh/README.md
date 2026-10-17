# Data provenance comparison

Mirror augmentation did not fix regression (14.0625% vsinitial). All recent training
arms reused data from the original batch-dependent inference pipeline. Test whether
that provenance limits learning under current reproducible search.

Generate40new selfplay games from same frozen initial weights, IDs0..39, seed62016,
64simulations, but current independent spatialBN+fixedshape16. First32train,last8
validation by game ID, never completion order. Oracle-validate all40games and each
training/validation position, legality, labels and targets. Save full data/histories.

Two matched learning arms: old_data and fresh_data. Each samples4935positions WITH
replacement from its32game pool using seed62017, then shuffles62016. Retain exact
indices and distinct sample counts. Both same initialstep0, minibatch64,78Adamupdates
at1e-4 without bias correction, legalCE+valueMSE+L2, independent spatial trainingnorm.
Old resampled control is rerun because sampling changes its training distribution.
No mirror augmentation. Shared original validation probe is descriptive only; fresh
validation remains saved and excluded from training. Same32game seed162016 discovery
bank vsinitial per arm. No superiority claim without independent confirmation.

Audit: original training position-weighted White result is-.01824; validation is-.6
(6Black wins/2White wins). Thus training is nearly outcome-balanced, while the small
validation sample is skewed. This does not establish a label bug or explain collapse.
Keep PR29 unchanged; all work on research branch.
