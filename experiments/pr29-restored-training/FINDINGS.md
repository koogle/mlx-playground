# Findings at deliberate cancellation

Run `pr29-restored-training-20261016-105037`, source `0fdaa5d`, was stopped
at the user's request. The Modal call is confirmed cancelled and the monitor
disabled. Checkpoints and logs remain in `chesszero-convergence`.

The run restored current-batch BatchNorm during search and recycled completed
promotion-match histories. Probability targets and raw-logit cross-entropy
remained unchanged. Search preserved stored running statistics, so this was
not an exact reconstruction of all historical normalization behavior.

| Candidate | Fresh games | Fresh positions | Recycled games / positions | Adam updates | Gate W/D/L | Score |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| 1 | 32 | 5,154 | 0 / 0 | 81 | 4 / 2 / 34 | 12.5% |
| 2 | 32 | 4,552 | 40 / 3,458 | 126 | 0 / 5 / 35 | 6.25% |

Both candidates were rejected. Each started from the accepted initial model
and Adam step zero; these are separate short attempts, not a continuous
207-update trajectory. There were 64 committed fresh games and no promotions.
Generation 3 was interrupted and is not counted as a committed generation.

These results show that the two restorations together did not prevent severe
early candidate weakness in this setup. They do not measure either restoration
individually or prove that longer uninterrupted training would recover.

The initial model under this run's normalization mode scored 55W/8D/1L over
64 random-opponent games: 92.1875%, draws half. White scored 27W/4D/1L,
90.625%; Black 28W/4D/0L, 93.75%; no games were censored. No final trained-model
random evaluation or final direct-baseline report was completed. Gate scores
are against the accepted initial model, not against random moves.

The follow-up static review found that both historical and current batchers
traverse games in chronological order without shuffling. Lack of shuffling is
therefore not a newly introduced change. The current loss uses raw logits and
matching side-to-move value targets; this review did not establish a new defect
in those formulas. See the historical audit for the independently verified
labels, backup signs, altered loss gradients and changed rejection schedule.

The full regression remains unresolved. The earlier successful checkpoint and
its actual runtime settings are unavailable, and isolated learning comparisons
have not been performed for the remaining coupled changes. Blindly reverting
the demonstrated rule, label, backup and cache-isolation fixes would reintroduce
known defects without proving recovery. No further training or diagnostic jobs
were launched after cancellation; a causal training fix is not claimed.
