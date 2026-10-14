"""CPU games use direct pipes; the calling process owns GPU inference."""

import logging
import multiprocessing as mp
from multiprocessing.connection import wait
import os

import numpy as np
from cachetools import LRUCache


class PipeInferenceClient:
    """One game has one outstanding prediction; responses need no request IDs."""

    def __init__(self, connection):
        self.connection = connection

    def __call__(self, states):
        self.connection.send(("predict", np.asarray(states[0], dtype=np.float32)))
        return self.connection.recv()

    def close(self):
        self.connection.close()


def run_pipe_games(model, config, worker, jobs, max_workers):
    """Run CPU game processes and service their predictions in the parent.

    The same pipe carries prediction requests and the final game result. Only
    this process imports MLX for inference; training can follow after it returns.
    We batch currently ready unique cache misses without a batching delay.
    """
    import mlx.core as mx

    if max_workers < 1 or config.inference_batch_size < 1:
        raise ValueError("Worker count and inference batch size must be positive")
    model.eval()
    ctx = mp.get_context("spawn")
    pending = iter(enumerate(jobs))
    active, results = {}, {}
    exhausted, failures = False, 0
    predictions = LRUCache(maxsize=config.prediction_cache_size)
    statistics = dict(cache_hits=0, forwards=0, positions=0, largest_batch=0)
    logger = logging.getLogger(__name__)

    try:
        while active or not exhausted:
            while not exhausted and len(active) < max_workers:
                try:
                    job_id, args = next(pending)
                except StopIteration:
                    exhausted = True
                    break
                parent, child = ctx.Pipe()
                process = ctx.Process(
                    target=worker,
                    args=(PipeInferenceClient(child), *args, child, max_workers),
                )
                gpu_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                try:
                    process.start()
                except Exception:
                    failures += 1
                    parent.close()
                    process.close()
                    logger.exception("Failed to start game %d", job_id)
                else:
                    active[parent] = process, job_id
                finally:
                    child.close()
                    if gpu_devices is None:
                        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                    else:
                        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices

            batch = {}
            ready = wait(active) if active else ()
            for connection in ready:
                try:
                    message = connection.recv()
                except (EOFError, OSError):
                    message = None
                if isinstance(message, tuple) and message[0] == "predict":
                    # Rotate serviced games so fast workers cannot repeatedly
                    # occupy the front of a large ready list.
                    active[connection] = active.pop(connection)
                    position = message[1]
                    key = (position.shape, position.tobytes())
                    if key in predictions:
                        statistics["cache_hits"] += 1
                        try:
                            connection.send(predictions[key])
                        except (BrokenPipeError, OSError):
                            pass
                    elif key in batch:
                        statistics["cache_hits"] += 1
                        batch[key][1].append(connection)
                    else:
                        batch[key] = position, [connection]
                    if len(batch) >= config.inference_batch_size:
                        break
                else:
                    process, job_id = active.pop(connection)
                    connection.close()
                    process.join()
                    if message is None or process.exitcode != 0:
                        failures += 1
                        logger.error(
                            "Game %d failed (exit %s)", job_id, process.exitcode
                        )
                    else:
                        results[job_id] = message
                    process.close()

            if batch:
                states = mx.array(
                    np.stack([position for position, _ in batch.values()])
                )
                policies, values = model(states)
                mx.eval(policies, values)
                policies, values = np.array(policies), np.array(values)
                statistics["forwards"] += 1
                statistics["positions"] += len(batch)
                statistics["largest_batch"] = max(
                    statistics["largest_batch"], len(batch)
                )
                for index, (key, (_, connections)) in enumerate(batch.items()):
                    prediction = (
                        policies[index : index + 1].copy(),
                        values[index : index + 1].copy(),
                    )
                    predictions[key] = prediction
                    for connection in connections:
                        try:
                            connection.send(prediction)
                        except (BrokenPipeError, OSError):
                            # EOF/exit handling on the next wait accounts for this game.
                            pass
    finally:
        for connection, (process, _) in active.items():
            if process.is_alive():
                process.terminate()
            process.join()
            process.close()
            connection.close()
    return [results[key] for key in sorted(results)], failures, statistics
