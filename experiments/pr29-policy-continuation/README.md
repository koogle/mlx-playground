# Three-generation continuation

Continue joint_ordered step78 with its full Adam state. Use frozen initial-policy outputs and the accepted trained network's value for self-play. The first frozen discovery comparison scored68.75%, but this is not confirmed superiority. Joint policy/value loss and training BatchNorm stay unchanged; inference uses independent spatial statistics and padded shape16.

Generate32 fresh games per generation for3 generations. Replay all games against python-chess, retain full tensors and histories, then train one ordered pass. Save every candidate and optimizer. Measure raw candidate and hybrid against initial on the existing16-game discovery bank. Promote the hybrid only if its40-game match against the accepted hybrid scores>55% with zero censored games. Self-play always uses accepted hybrid; rejected candidates do not replace it. The learner and optimizer continue across rejected proposals to test whether subsequent training recovers (not repeated cold resets).

This controlled continuation does not recycle evaluation games; it keeps evidence separate from training. No production recycling setting is changed. Updates track attempted and the accepted checkpoint's cumulative updates. The gate is an operational selection rule, not statistical confirmation.

After96 fresh games, evaluate starting and final accepted hybrid on the same new64-game held-out bank against initial. If no promotion, reuse the identical starting-checkpoint result instead of duplicating games. Keep censored counts separate. These results will determine whether to extend, change training, or investigate further. Runtime bounded at4.5h. All rejected checkpoints, data, revisions, exact moves, and per-color scores retained.
