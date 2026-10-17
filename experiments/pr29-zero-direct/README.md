# Direct initial-value intervention

Frozen identical epoch0 weights; candidate nonterminal neural value zero, baseline unchanged. Both use simulation cap64 and unchanged earlystop, independent spatial BN and fixed inference shape16. Exact terminal outcomes and policies unchanged. Zero optimizer updates.

128 games (64 paired openings), fresh seed232016, disjoint from search-budget and search-confirmation banks. Primary score against 50%, alpha .05, bootstrap over16 opening families. Censored games reported separately; no superiority claim with censoring. Save full histories and actual simulation counts. Assert initial Adamstep0, identical fingerprints, exact candidate policy equality and zero value, final immutability of both underlying models.

Tests whether removing initial value improves play directly; previous studies only compared relative search-budget effects. This does not establish learned strength or a training fix. Research branch only; PR29 unchanged.
