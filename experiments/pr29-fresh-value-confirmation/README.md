# Confirm learned-value damage on fresh openings

Frozen fresh_data step78 from pr29-data-refresh-20261017-080116. Two arms: unchanged full model, and only its nonterminal value zeroed. Initial opponent unchanged in both. Same trained policy/weights,64 simulations, independent spatial BN and padded shape16. No optimizer updates.

128 games per arm on identical fresh64 paired openings seed242016, disjoint from source discovery bank. Primary contrast paired16-family bootstrap zero-minus-unchanged score95%CI, with worst-case censor sensitivity. Each versus initial alpha.025; censored games separate, no superiority claim with censored results. Verify Adamstep78/effectiveLR1e-4/sourcefingerprint, zero policyidentity/valuezero and both rawmodels frozen afterplay. Retain allhistories/simulation counts.

Discovery zero13W6D13L50% versus original2W6D24L15.625%; both32 complete. Recovery is not proven improvement over initial, and does not establish a training fix. PR29 unchanged; research branch only.
