# Frozen color-imbalance diagnosis

The first ten-update run regressed primarily as White. This experiment changes inference outputs while holding every learned parameter and BatchNorm running statistic fixed, to distinguish a harmful value signal from a harmful policy signal.

Each intervention plays 32 games against random, using exactly the first 16 color pairs and random seeds of the original evaluation. The reference scores are restricted to those same pairs. All moves and outcomes are independently checked by python-chess. These are discovery comparisons; neither a baseline-superiority claim nor a general causal explanation follows from one small bank.

Interventions:

- Trained policy with zero value.
- Trained policy with the initial checkpoint's value.
- Initial policy with the trained checkpoint's value.
- Trained policy with the average of original and color/rank-mirrored side-to-move values.
- The average of original and color/rank-mirrored policies and values, mapping policy indices back to the original board.
- Trained policy with negated trained values, as a diagnostic control.

Every intervention verifies full parameter fingerprints before/after, including normalization statistics. There are zero optimizer updates and zero normalization updates. Evaluation positions never fit any model. Both models are in evaluation mode throughout. The color/rank transformation preserves side-to-move value rather than negating it; policy indices are permuted with explicit underpromotion handling.

```sh
modal run --detach experiments/pr29-color-diagnostic/modal_diagnostic.py
PYTHONPATH=chesszero python -m pytest experiments/pr29-color-diagnostic/test_ablate.py -q
```

Active run: `pr29-color-diagnostic-20261015-092800`, [Modal app](https://modal.com/apps/radiantai/main/ap-ftSat4qc9I0jRgy2B6aJCk), persisted in volume `chesszero-convergence`. `progress.json` contains completed interventions and color-specific references; `result.json` and `job.json` establish completion. A 90-minute workload limit bounds this diagnostic. The separate fresh-seed ten-generation repeat continues unchanged.

## Initial discovery evidence

The first two completed interventions isolate the value output as a cause of the original White regression on this bank. Each color has 16 games; there are no censored games in these comparisons.

| Frozen inference combination | White score | Black score |
| --- | ---: | ---: |
| Initial policy + initial value (reference) | 81.25% | 96.875% |
| Trained policy + trained value (reference) | 50.0% | 96.875% |
| Trained policy + zero neural value | 96.875% | 93.75% |
| Trained policy + initial value | 93.75% | 87.5% |

The zero intervention retains exact terminal rule results inside MCTS. It removes only the nonterminal neural value. These comparisons establish a harmful value contribution on the tested bank; they do not yet explain why value training fails or establish direct superiority against the initial model.

## Additional mechanism checks

`probe_value.py` reads intermediate checkpoints and evaluates identical original/mirrored position pairs without updating weights or statistics. The first run's mean absolute mirror-value discrepancy grows from approximately 0.000617 initially to 0.063919 at epoch 10. Its final Black-to-move predictions have standard deviation 0.00790 versus 0.03970 for White on the paired set, despite equal target distributions by construction. Hidden ReLU activity is about 18.5% for both colors; a wholly dead Black value head is not supported. The fresh seed at epoch 3 instead predicts approximately +0.38 for both colors on this same set, so a universal fixed Black-favoring sign error is not established. These are diagnostic observations on evaluation histories, never fitting data.

Activation run: `pr29-value-activations-20261015-093158`, [Modal](https://modal.com/apps/radiantai/main/ap-ri0COjGaKGWJipQEjC7O3g). It also evaluates the frozen fresh-seed epoch-3 checkpoint on the same 32-game subset.

`calibrate.py` generates 16 new complete self-play games with generation seed index 9000. It restores the identical trained checkpoint for each intervention, then refreshes only BatchNorm running statistics: original position order, shuffled positions, or shuffled mirror-balanced batches. It checks that all trainable parameters remain byte-identical and evaluates the same discovery bank. No evaluation position or outcome enters calibration. This distinguishes a normalization mechanism from harmful learned weights; pending results must not be described as a fix.

Calibration run: `pr29-normalization-diagnostic-20261015-093425`, [Modal](https://modal.com/apps/radiantai/main/ap-1Ra3zt0z5d3AAkFUh5zBE9). The workload is capped at one hour. Launch with `modal_calibrate.py`; read-only activation checks use `modal_probe.py`.

The reverse head swap (initial policy + trained value) also retains a White deficit: White 62.5%, Black 100%, on 16 games per color. Thus changing policy alone does not remove the deficit, whereas replacing the value does on this bank.

The fresh-seed epoch-3 evaluation completed with White 14W/2D/0L (93.75%) and Black 5W/9D/2L (59.375%). Its own epoch-zero reference on those identical first16 pairs was White 87.5% and Black 96.875%. This reverses the weak color relative to the first run; it is an intermediate checkpoint, not the final ten-update repeat. A fixed Black-favoring scoring error is not supported. A new frozen head intervention checks whether the value path also causes this reverse imbalance: `pr29-color-diagnostic-20261015-093849`, [Modal](https://modal.com/apps/radiantai/main/ap-lSKT1ayVA5qw5aP5rcN4UG). The fresh-seed value intervention is pending, so causality for that checkpoint is not yet established.

The diagnostic launcher accepts `--source-run`, `--epoch`, `--kinds` (comma separated), `--reference-path` (relative to the volume root), and `--launch-file` to repeat a controlled intervention on another saved checkpoint without changing any model. The default protocol remains the original epoch-10 six-intervention experiment.
