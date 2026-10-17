# Search budget as a policy-target quality check

Value anchoring held values close to initial but scored37.5% (8W8D16L), not a fix.
Before further training, test whether deeper search improves play with identical
initial network weights. Candidate cap256 versus baseline cap64, same existing early
stopping, paired16openings seed202016 (32games), same independent spatialBN/padding16.
Only search budget changes. Record actual simulation counts, exact moves and oracle
terminal results, censoring separately, frozen fingerprints. No optimizer updates.
Any win establishes search-budget benefit on this bank, not trained-model improvement.
A positive result motivates a matched training-data/search-target experiment.
No production changes; PR29 untouched.
