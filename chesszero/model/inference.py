"""One model owner batches inference requests from CPU game processes."""

import logging
from multiprocessing.connection import Connection, wait
from collections.abc import Sequence

from config.model_config import ModelConfig
from model.network import ChessNet
from cachetools import LRUCache
import threading
import time

import mlx.core as mx
import numpy as np


class InferenceClient:
    """Picklable model interface; only a pipe and model identifier cross processes."""

    def __init__(self, connection: Connection, model_id: int, batch_size: int):
        self.connection = connection
        self.model_id = model_id
        self.batch_size = batch_size
        self.training = False

    def train(self, mode: bool = True):
        self.training = mode

    def eval(self):
        self.train(False)

    def __call__(self, states: mx.array) -> tuple[mx.array, mx.array]:
        states = np.asarray(states, dtype=np.float32)
        predictions = []
        for start in range(0, len(states), self.batch_size):
            try:
                self.connection.send(
                    (self.model_id, states[start : start + self.batch_size])
                )
                predictions.append(self.connection.recv())
            except (EOFError, OSError) as error:
                raise RuntimeError("Shared inference service disconnected") from error
        policies, values = zip(*predictions)
        return mx.array(np.concatenate(policies)), mx.array(np.concatenate(values))


class BatchedInference:
    """Serve immutable model snapshots in the parent while spawned games use CPU.

    A single thread owns all forward calls during generation. It stops before
    training resumes, so inference never overlaps mutations to the model/BN state.
    Evaluation can register two snapshots; their requests are batched separately.
    """

    def __init__(self, models: Sequence[ChessNet], config: ModelConfig):
        if (
            config.inference_batch_size < 1
            or config.inference_wait_ms < 0
            or config.prediction_cache_size < 1
        ):
            raise ValueError(
                "Inference/cache sizes must be positive and wait nonnegative"
            )
        self.models = list({id(model): model for model in models}.values())
        self.model_ids = {id(model): index for index, model in enumerate(self.models)}
        self.batch_size = config.inference_batch_size
        self.wait_seconds = config.inference_wait_ms / 1000
        # This owner exists only while its model snapshots are immutable.
        self.predictions = LRUCache(maxsize=config.prediction_cache_size)
        self.cache_hits = 0
        self.device = mx.default_device()
        self.connections = set()
        self.lock = threading.Lock()
        self.wakeup = threading.Event()
        self.stop = threading.Event()
        self.error = None
        self.forward_passes = self.positions = self.max_batch_size = 0

    def __enter__(self):
        self.original_modes = [model.training for model in self.models]
        for model in self.models:
            model.eval()
            mx.eval(model.parameters())
        self.thread = threading.Thread(target=self._serve, name="chess-inference")
        self.thread.start()
        return self

    def prepare_job(self, args, context):
        """Replace model arguments with clients, opening pipes only for live jobs."""
        with self.lock:
            if self.error is not None:
                raise RuntimeError("Shared inference failed") from self.error
            receiver, sender = context.Pipe()
            self.connections.add(receiver)
        self.wakeup.set()
        args = tuple(
            InferenceClient(sender, self.model_ids[id(arg)], self.batch_size)
            if id(arg) in self.model_ids
            else arg
            for arg in args
        )
        return args, sender

    def _disconnect(self, connection):
        with self.lock:
            self.connections.discard(connection)
        connection.close()

    def _receive(self, timeout):
        with self.lock:
            connections = tuple(self.connections)
        if not connections:
            self.wakeup.wait(timeout)
            self.wakeup.clear()
            return []
        requests = []
        for connection in wait(connections, timeout=timeout):
            try:
                model_id, states = connection.recv()
            except (EOFError, OSError):
                self._disconnect(connection)
                continue
            requests.append((connection, model_id, states))
        return requests

    def _predict(self, requests):
        for model_id, model in enumerate(self.models):
            group = [request for request in requests if request[1] == model_id]
            # Limit each GPU forward even if many clients became ready together.
            for start in range(0, len(group), self.batch_size):
                batch = group[start : start + self.batch_size]
                states = np.concatenate([request[2] for request in batch])
                # Exact NN inputs, rather than rule/history keys, identify predictions.
                keys = [(model_id, row.shape, row.tobytes()) for row in states]
                predictions, missing = {}, {}
                for key, row in zip(keys, states):
                    if key in predictions or key in missing:
                        self.cache_hits += 1
                    elif key in self.predictions:
                        predictions[key] = self.predictions[key]
                        self.cache_hits += 1
                    else:
                        missing[key] = row
                # Deduplicate within a batch as well as across games and requests.
                pending = list(missing)
                for offset in range(0, len(pending), self.batch_size):
                    batch_keys = pending[offset : offset + self.batch_size]
                    inputs = mx.array(np.stack([missing[key] for key in batch_keys]))
                    policies, values = model(inputs)
                    mx.eval(policies, values)
                    policies, values = np.array(policies), np.array(values)
                    self.forward_passes += 1
                    self.positions += len(inputs)
                    self.max_batch_size = max(self.max_batch_size, len(inputs))
                    for key, policy, value in zip(batch_keys, policies, values):
                        # Copies let LRU eviction release each entry independently.
                        policy, value = policy.copy(), value.copy()
                        policy.setflags(write=False)
                        value.setflags(write=False)
                        predictions[key] = policy, value
                        self.predictions[key] = policy, value
                # Local references remain valid even if this batch exceeds the LRU.
                policies = np.stack([predictions[key][0] for key in keys])
                values = np.stack([predictions[key][1] for key in keys])
                offset = 0
                for connection, _, inputs in batch:
                    end = offset + len(inputs)
                    try:
                        connection.send((policies[offset:end], values[offset:end]))
                    except (EOFError, OSError):
                        self._disconnect(connection)
                    offset = end

    def _serve(self):
        try:
            mx.set_default_device(self.device)
            while not self.stop.is_set():
                requests = self._receive(0.05)
                if not requests:
                    continue
                deadline = time.monotonic() + self.wait_seconds
                while sum(len(request[2]) for request in requests) < self.batch_size:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    requests.extend(self._receive(remaining))
                self._predict(requests)
        except Exception as error:
            with self.lock:
                self.error = error
        finally:
            with self.lock:
                connections = tuple(self.connections)
            for connection in connections:
                self._disconnect(connection)

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop.set()
        self.wakeup.set()
        self.thread.join()
        self.predictions.clear()
        for model, training in zip(self.models, self.original_modes):
            model.train(training)
        logging.getLogger(__name__).info(
            "Shared inference: %d forwards, %d positions, largest batch %d, %d cache hits",
            self.forward_passes,
            self.positions,
            self.max_batch_size,
            self.cache_hits,
        )
        if exc_type is None and self.error is not None:
            raise RuntimeError("Shared inference failed") from self.error
