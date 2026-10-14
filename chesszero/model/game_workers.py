"""Run a bounded set of independent game processes and account for every exit."""

import logging
import multiprocessing as mp
from multiprocessing.connection import wait

import mlx.core as mx


def run_game_workers(worker, jobs, max_workers):
    """Return successful results and a failure count after all jobs finish.

    Each worker owns a result pipe. EOF accounts for crashes even when no result
    was sent, so a missing game cannot leave the parent waiting indefinitely.
    """
    if max_workers < 1:
        raise ValueError("Worker count must be positive")
    logger = logging.getLogger(__name__)
    cuda = getattr(mx, "cuda", None)
    if (
        max_workers > 1
        and cuda is not None
        and cuda.is_available()
        and mx.default_device() == mx.gpu
    ):
        # Each spawned game owns its model, CUDA context, and allocator. Unlike
        # a shared inference server, these allocations multiply with workers.
        logger.warning(
            "Limiting CUDA game workers from %d to 1: this scheduler uses "
            "a separate GPU model per game. Use shared inference for parallel "
            "GPU self-play.",
            max_workers,
        )
        max_workers = 1
    ctx = mp.get_context("spawn")
    pending = iter(enumerate(jobs))
    active = {}
    results = {}
    failures = 0
    exhausted = False
    try:
        while active or not exhausted:
            while not exhausted and len(active) < max_workers:
                try:
                    job_id, args = next(pending)
                except StopIteration:
                    exhausted = True
                    break
                receiver, sender = ctx.Pipe(duplex=False)
                process = ctx.Process(
                    target=worker, args=(*args, sender, max_workers)
                )
                try:
                    process.start()
                except Exception:
                    failures += 1
                    receiver.close()
                    process.close()
                    logger.exception("Failed to start game job %d", job_id)
                else:
                    active[receiver] = (process, job_id)
                finally:
                    sender.close()

            for receiver in wait(active) if active else ():
                process, job_id = active[receiver]
                try:
                    result = receiver.recv()
                except (EOFError, OSError):
                    result = None
                finally:
                    receiver.close()
                    process.join()
                if result is None or process.exitcode != 0:
                    failures += 1
                    logger.error(
                        "Game job %d failed (exit code %s)", job_id, process.exitcode
                    )
                else:
                    results[job_id] = result
                process.close()
                del active[receiver]
    finally:
        for receiver, (process, _) in active.items():
            if process.is_alive():
                process.terminate()
            process.join()
            process.close()
            receiver.close()
    return [results[key] for key in sorted(results)], failures
