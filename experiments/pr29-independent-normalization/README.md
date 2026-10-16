# Frozen normalization control

The batch-stat study's identical-weight control scored 4W/2D/10L (31.25%).
Current-batch normalization makes inference depend on other concurrently played
games. This control repeats the saved candidates and identical discovery
openings with independent per-position spatial normalization, equivalent to
evaluating each position alone with batch-size-one BatchNorm. Training BatchNorm
and all checkpoint tensors are unchanged. No learning or fresh self-play occurs.

Before each match, compare serial, mixed-batch, independently normalized and
reordered GPU predictions on 16 saved validation positions. Assert numerical
equivalence of independent and serial predictions and unchanged weights/statistics.
Record the original batch coupling magnitude. The initial-versus-initial control
also records whether paired games follow identical move sequences. Repeat all
saved arms, including the frozen zero-value intervention, for matched comparisons.

This tests a reproducibility defect in batch-stat search, not a proven explanation
of the earlier running-statistics training regression. Discovery scores remain
exploratory. The existing controlled-learning job continues to completion without
being duplicated or modified.
