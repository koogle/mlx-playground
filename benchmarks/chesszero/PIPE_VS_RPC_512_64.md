# Main-derived pipes versus PR #22 RPC, 512 games at 64 simulations

This is a throughput experiment, not a learning-convergence result. The pipe prototype starts from main after PR #27 (525a988), shares the parent GPU model through direct game pipes, and bypasses the per-model CUDA worker guard because game workers use NumPy only. PR #22 uses its dedicated manager process. Neither launches a GPU model per game. Both retain the merged cache and tree reuse changes.

Original model: 19 residual blocks, 256 filters. Same generation-4 checkpoint, game seeds 9000+game_id, 512 fresh terminal games, 64 simulations per move, 64-position inference batches, 32 CPU cores and one L40S per trial. CUDA convolution cache is explicitly 1024 in both runtimes. Completed games feed one sample-weighted accumulated Adam update, with finite-parameter and checkpoint-reload checks. The driver streams training microbatches rather than using Trainer.train's materialized batch list.

The pipe branch is a self-play-only prototype; evaluation integration remains unfinished. The repository default changes from 1600 to 64 in separate PR #28. Benchmark overrides the saved checkpoint's simulation budget explicitly for both trials.

CPU validation used a small network: duplicate-request cache reuse, crash and missing-result accounting, four terminal games totaling 1366 positions, finite Adam update and checkpoint save.

Pipe result: 512/512 games, zero failures, 70,175 positions; self-play 1,803.68 seconds (17.03 games/minute, 38.91 positions/second). Learning took 55.60 seconds, Adam step advanced from 4 to 5, parameters were finite, and checkpoint reload matched all weights and optimizer step. RPC also passed: 512 games, zero failures, 70,079 positions; self-play 2,143.96 seconds (14.33 games/minute, 32.69 positions/second). Learning took 44.41 seconds, Adam advanced from 4 to 5, parameters were finite, and checkpoint reload passed. Raw run metadata and results are stored with this report; Modal volume chesszero-convergence also preserves each trial's logs and GPU/host-memory samples.

## Comparison

| Approach | Parallel games | Games/minute | Positions/second | Self-play minutes | Learning seconds | GPU peak GiB | Host peak GiB | Sampled search GPU utilization |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Main-derived direct pipes | 512 | 17.03 | 38.91 | 30.06 | 55.60 | 12.36 | 201.10 | 35.9% |
| PR #22 manager RPC | 512 | 14.33 | 32.69 | 35.73 | 44.41 | 10.71 | 201.00 | 31.0% |

Direct pipes achieved 18.87% more games per minute and reduced generation time by 15.87%. Including learning, it reduced time for self-play plus one update from 2,188.37 to 1,859.28 seconds (15.04%). GPU peaks cover the whole trial, including learning, and utilization averages sample the self-play interval after startup. Both use one model for inference, CPU game processes, fixed model weights during search, the same chess rules/noise/tree logic, fresh data, and sample-weighted gradient accumulation. Direct pipes also avoid the dedicated GPU process's checkpoint reload and weight-snapshot update per batch, so this measures the complete execution approaches rather than transport overhead alone.

The cases produced matching game lengths for 509/512 seeds. Total positions differed by 96 (0.137%) and NN evaluations differed by 0.088%; batching can change floating-point results and a few subsequent choices. Game lengths alone do not prove identical trajectories. The mean NN batch was 42.89 for pipes and 42.53 for RPC; both reached the 64-position cap. This was one paired trial on two physical GPUs, not a statistical confidence interval or convergence test. Both were L40S with driver 580.95.05, 350 W limits and 2520 MHz maximum SM clocks; UUIDs are recorded in hardware.json.

512 simultaneous CPU games is a demonstrated safe configuration for this 32-core, high-memory L40S setup. It required about 201 GiB of host memory, even though GPU memory peaked below 13 GiB. 512 is a recommendation from this test, not a demonstrated maximum or optimum at 64 simulations. Do not infer that 1024 games would fit the same host-memory limit; cache/tree memory could roughly double. Native main with the PR #27 guard still uses one GPU game worker; the parallel pipe implementation is an experimental change on top of main.

Both trials completed successfully and the Modal benchmark app is stopped. Evaluation integration remains unfinished on the pipe branch, so it is not ready to replace the production training/evaluation flow. PR #22 remains open; no inference implementation has been merged by this experiment.
