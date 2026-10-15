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

The fresh-seed epoch-3 zero-value intervention has also completed: White 15W/1D/0L and Black 15W/1D/0L (96.875% each), versus 93.75% and 59.375% with its unchanged trained value. Thus removing the neural value recovers the weak color for both seeds while preserving the trained policy and exact terminal scores. [Interim discovery evidence](results/value-intervention-discovery.json). This isolates the harmful value contribution in both checkpoints; the normalization mechanism and final fresh-seed epoch-10 results remain pending.

## Completed controls

All six frozen interventions completed with unchanged model/normalization fingerprints and no failed or censored games. In addition to the earlier table, mirrored-value inference scored White 96.875% / Black 100%; mirrored policy plus value scored 87.5% / 100%. Negating the value instead scored 87.5% / 50%, reversing the weak side rather than producing a general fix. [Full frozen results](results/frozen-interventions.json).

The fresh-seed epoch-3 reverse swap (initial policy + trained value) scored White 90.625% / Black 75%, retaining a Black deficit. Zeroing only its value scored 96.875% for both colors. The complete job verified frozen parameters. [Fresh-seed results](results/fresh-seed-epoch3-interventions.json).

Stats-only recalibration did not reliably solve the original deficit: ordered data scored White 71.875% / Black 93.75%; shuffled data 50% / 87.5%; mirror-balanced data 56.25% / 100%. All learned weights remained unchanged. These controls show sensitivity to statistics but do not establish normalization as the sole cause. [Normalization results](results/normalization.json).

The initial random weights with an ordered normalization warmup on the same fresh calibration dataset scored White 84.375% / Black 93.75%, versus 81.25% / 96.875% initially. This did not reproduce the severe White collapse. Zeroing the warmed random value gave 87.5% / 84.375%, so the trained model's large zero-value recovery is not a universal benefit of disabling any value head. [Warmup control](results/random-normalization-warmup.json).

## Confirmation and first-update mechanism

`confirm_value.py` compares the fixed epoch-10 model with mirrored-value inference on a new 64-game opponent-seed bank (`60000`), 32 games per color. It uses the same opening families but new random-opponent streams, and selects no further hyperparameters. This is confirmation of the color intervention, not a direct initial-model match. Run `pr29-value-confirmation-20261015-094902`, [Modal](https://modal.com/apps/radiantai/main/ap-dI4Aol9vxDCeyHXERNFoMW).

`first_step.py` reconstructs the model immediately before its first optimizer update: epoch-zero learned weights and epoch-one BatchNorm statistics. This reconstruction is exact for the generation-wide accumulation protocol, which holds learned weights fixed while processing every microbatch. It compares that state, the actual saved first update, and a bias-corrected first update reconstructed from the identical saved Adam moments and learning rate. The actual uncorrected update must match the saved checkpoint before any comparisons run. This isolates normalization from the first weight update and tests Adam's default bias-correction setting as a possible contributor; results are pending. Both update formulas were independently checked against real MLX optimizers locally, including epsilon-sensitive gradients. Run `pr29-first-step-diagnostic-20261015-095818`, [Modal](https://modal.com/apps/radiantai/main/ap-sxMDASFLU7xmyQiQ1D3Vp5).

The held-out opponent-seed confirmation completed successfully with all parameters/statistics frozen and no failed or censored games. The unchanged model scored White 7W/19D/6L (51.5625%) and Black 31W/1D/0L (98.4375%); mirrored-value inference scored **30W/2D/0L (96.875%) for each color**. The paired-bootstrap 95% interval for the reduction of the color gap is 34.375–57.8125 percentage points. [Confirmation evidence](results/mirror-value-confirmation.json). The opening families were unchanged; the random-opponent seeds were new. This confirms the value intervention for these banks, not superiority in direct matches against the initial model.

At the user's request, the remaining exploratory first-step job was cancelled with container termination after preserving its available evidence. It is deliberately incomplete; do not present it as a completed optimizer comparison or conclude that Adam bias correction is the cause. Its before-update reconstruction scored White 78.125% / Black 90.625%, which does not reproduce the original severe White collapse. [Cancellation and partial evidence](results/first-step-cancelled.json). No further exploratory tests are being launched. The fresh-seed ten-generation retraining run continues unchanged, and is the only remaining GPU workload from this investigation.
