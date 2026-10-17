# Early checkpoints after legal loss and matched normalization

Final step78 of pr29-legal-training-norm-20261017-060455 scored0W8D24L12.5%,
both colors0W4D12L12.5%; zero censoring. Matching normalization did not fix learning.
Validation value MSE .998 initial,1.002 step8,.971 step32,1.354 step78 motivates
checking whether earlier snapshots preserve or improve playing strength.

Evaluate already-saved checkpoints8 and32, each32games on same seed162016 discovery
openings versus initial. Zero new optimizer updates. Savedstep/effectiveLR1e-4 and
frozen model fingerprints checked. Same independent spatial searchBN, padding16,
64simulations. Preserve exact histories and checkpoints. These checkpoints are
selected using existing validation evidence, so discovery only; any apparent gain
needs an independent confirmation, then sustained selfplay validation. No automatic
promotion and no production changes. PR29 remains unchanged.
