"""Run a bounded set of independent game processes and account for every exit."""

import logging
import multiprocessing as mp
import os
from multiprocessing.connection import wait


def run_game_workers(worker, jobs, max_workers, clients):
    """Return successful results and a failure count after all jobs finish.

    Each worker owns a result pipe. EOF accounts for crashes even when no result
    was sent, so a missing game cannot leave the parent waiting indefinitely.
    """
    if max_workers < 1:
        raise ValueError("Worker count must be positive")
    ctx = mp.get_context("spawn")
    pending = iter(enumerate(jobs))
    active = {}
    results = {}
    failures = 0
    exhausted = False
    logger = logging.getLogger(__name__)
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
                    target=worker, args=(*clients, *args, sender, max_workers)
                )
                # Hide CUDA before spawn imports MLX in a game worker. Selecting
                # mx.cpu after import can still leave an unnecessary GPU context.
                gpu_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
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
                    if gpu_devices is None:
                        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                    else:
                        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
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
