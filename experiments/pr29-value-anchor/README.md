# Value-target anchoring intervention

Data-refresh did not fix collapse: old_data12.5%, fresh_data15.625% on32games each.
Earlier frozen substitutions localized value damage; fresh_data fit sampled outcome
targets closely while validation outcome MSE worsened. Test whether learning the
policy while constraining value drift improves play.

Use exactly the fresh_data4935 sampled positions (3590unique), same saved indices,
shuffle62016, initialweights/step0,78Adamupdates1e-4, legalCE/L2 and independent
spatial training normalization. ONLY value training targets change: replace terminal
outcomes with frozen initial-model values under fixedshape16 independent inference.
Original outcome labels retained in value-targets.npz; no sign corrections claimed.
Full trained model used for play, no output substitution. Value MSE weight unchanged.

Validate original data/masks first. Assert finite targets, unchanged initial teacher,
and initial model reproduces validation anchors. Track both terminal-outcome and
anchor MSE on fresh heldout8games (1766positions). This differs from data-refresh's
old validation probe; do not compare those outcome MSEs as identical banks.
Same32game seed162016 discovery evaluation, checkpoint stages retained. This is a
mechanism experiment; preserving initial values isn't successful outcome-value
learning, and discovery gains need confirmation then sustained selfplay improvement.
No production changes or additions to PR29.
