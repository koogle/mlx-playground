# Independent confirmation of legal-action policy loss

Frozen step78 legal_policy checkpoint from pr29-legal-policy-20261017-032425
versus the matched batch_training checkpoint from pr29-training-normalization-
20261016-210929. Both started at identical epoch0, trained on identical data/order
for78 Adam updates at1e-4; only policy loss differs. No optimizer updates here.
Each plays128games versus initial on identical fresh seed192016 openings,
paired colors,64simulations, independent spatial searchBN and padded shape16.

Primary contrast: legal-minus-global score difference with paired16-opening-family
bootstrap95% interval. Both versus-initial comparisons use alpha.025 and family
bootstrap97.5%. Censored games separate, zero required for strength claims.
Verify saved optimizer step, effectiveLR, source fingerprint and inference immutability.
Preserve exact histories and both checkpoints; no selection/training on this bank.
Discovery: legal11W6D15L43.75% (White3W3D10L28.125%, Black8W3D5L59.375%),
global0W2D30L3.125%; all32 per arm complete. Avoiding collapse does not prove
improvement over the initial model or sustained self-play learning.
