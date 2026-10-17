# Frozen output substitutions after shuffled legal-CE training

Shuffled legal-CE training scored1W4D27L9.375% on32games, versus ordered43.75%.
Validation value MSE increased from0.998 to1.303. These observations motivate
localizing the damage before another training intervention.

Two frozen interventions on the same seed162016 discovery openings:
- initial policy output + trained value output;
- trained policy output + initial value output.

No weights replaced or optimizer updates. Same saved legal_policy step78 from
pr29-legal-shuffle-20261017-044626, each32games vsinitial,64total. Same64simulations,
independent spatial searchBN and fixed padded shape16. Verify savedAdamstep/LR,
sourcefingerprint, normalization invariance and frozen weights throughout matches.
Both source networks are evaluated in the same inference context. Only one returned
output changes in each arm. Comparison reference is retained unmodified9.375%.
Mechanism/discovery bank reused; no claim of confirmed model superiority.
