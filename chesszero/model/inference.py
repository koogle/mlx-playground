"""Queued model RPC in one dedicated process, using local manager IPC."""

from __future__ import annotations

from concurrent.futures import Future
import logging
import multiprocessing as mp
from multiprocessing.managers import BaseManager
from pathlib import Path
from queue import Queue, Empty
from tempfile import TemporaryDirectory
import threading
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np
from cachetools import LRUCache
from config.model_config import ModelConfig

if TYPE_CHECKING:
    from model.network import ChessNet


class InferenceClient:
    """Inference-only RPC handle; game workers receive no model or GPU tensors."""

    def __init__(self, rpc, reference=False):
        self.rpc, self.reference = rpc, reference

    def __call__(self, states: mx.array) -> tuple[mx.array, mx.array]:
        position = np.asarray(states[0], dtype=np.float32)
        predict = self.rpc.predict_reference if self.reference else self.rpc.predict
        policy, value = predict(position)
        return mx.array(policy), mx.array(value)

    def close(self):
        # The manager owns the backend; releasing a client must not close it.
        self.rpc = None


class _InferenceBackend:
    """The manager constructs this object inside its dedicated GPU process."""

    def __init__(self, config, current_path, reference_path, device):
        from model.network import ChessNet

        self.device = mx.gpu if device == "gpu" else mx.cpu
        mx.set_default_device(self.device)
        self.model = ChessNet(config)
        self.model.eval()
        self.snapshots = {}
        for name, path in (("current", current_path), ("reference", reference_path)):
            if path is not None:
                self.model.load_weights(path)
                self.snapshots[name] = self.model.parameters()
                mx.eval(self.snapshots[name])
        self.batch_size = config.inference_batch_size
        self.predictions = LRUCache(maxsize=config.prediction_cache_size)
        self.cache_hits = 0
        self.requests = Queue()
        self.error = None
        self.forward_passes = self.positions = self.max_batch_size = 0
        self.executor = threading.Thread(target=self._serve)
        self.executor.start()

    def predict(self, position):
        return self._request("current", position)

    def predict_reference(self, position):
        return self._request("reference", position)

    def _request(self, snapshot, position):
        response = Future()
        self.requests.put((snapshot, position, response))
        return response.result()

    def _predict(self, batch):
        # Keys describe the exact network input under a fixed weight snapshot.
        # Rules/history and root exploration noise never enter this cache.
        pending, predictions = {}, {}
        for snapshot, position, _ in batch:
            key = (snapshot, position.shape, position.tobytes())
            if key in self.predictions:
                predictions[key] = self.predictions[key]
                self.cache_hits += 1
            elif key in pending:
                self.cache_hits += 1
            else:
                pending[key] = position
        for snapshot in {key[0] for key in pending}:
            keys = [key for key in pending if key[0] == snapshot]
            self.model.update(self.snapshots[snapshot])
            states = mx.array(np.stack([pending[key] for key in keys]))
            policies, values = self.model(states)
            mx.eval(policies, values)
            policies, values = np.array(policies), np.array(values)
            self.forward_passes += 1
            self.positions += len(keys)
            self.max_batch_size = max(self.max_batch_size, len(keys))
            for index, key in enumerate(keys):
                prediction = (policies[index : index + 1], values[index : index + 1])
                self.predictions[key] = predictions[key] = prediction
        # Keep this batch's results even if its size exceeds the LRU capacity.
        for snapshot, position, response in batch:
            key = (snapshot, position.shape, position.tobytes())
            response.set_result(predictions[key])

    def _serve(self):
        mx.set_default_device(self.device)
        while True:
            first = self.requests.get()  # Block only when the queue has no work.
            if first is None:
                return
            batch = [first]
            while len(batch) < self.batch_size:
                try:
                    request = self.requests.get_nowait()
                except Empty:
                    break
                if request is None:
                    self.requests.put(None)
                    break
                batch.append(request)
            try:
                if self.error is not None:
                    raise RuntimeError(self.error)
                self._predict(batch)
            except Exception as error:
                self.error = str(error)
                for _, _, response in batch:
                    if not response.done():
                        response.set_exception(RuntimeError(self.error))

    def statistics(self):
        return dict(
            cache_hits=self.cache_hits,
            forwards=self.forward_passes,
            positions=self.positions,
            largest_batch=self.max_batch_size,
            error=self.error,
        )

    def close(self):
        self.requests.put(None)
        self.executor.join()


class _InferenceManager(BaseManager):
    pass


_InferenceManager.register(
    "backend",
    _InferenceBackend,
    exposed=("predict", "predict_reference", "statistics", "close"),
)


class BatchedInference:
    """One process owns inference; all clients use the same local RPC endpoint.

    Model weights are fixed for the session. Evaluation's accepted checkpoint
    uses the same executor and model architecture, with separate weight snapshots.
    """

    def __init__(self, model: ChessNet, config: ModelConfig, reference_model=None):
        if config.inference_batch_size < 1:
            raise ValueError("Inference batch size must be positive")
        self.model, self.config, self.reference_model = model, config, reference_model
        self.device = "gpu" if mx.default_device() == mx.gpu else "cpu"

    def __enter__(self):
        self.files = TemporaryDirectory(prefix="chess-inference-")
        current = str(Path(self.files.name) / "current.safetensors")
        self.model.save_weights(current)
        reference = None
        if self.reference_model is not None:
            reference = str(Path(self.files.name) / "reference.safetensors")
            self.reference_model.save_weights(reference)
        self.manager = _InferenceManager(ctx=mp.get_context("spawn"))
        self.manager.start()  # Standard library RPC over a local Unix-domain socket.
        try:
            self.rpc = self.manager.backend(
                self.config, current, reference, self.device
            )
        except Exception:
            self.manager.shutdown()
            self.files.cleanup()
            raise
        return self

    def client(self) -> InferenceClient:
        return InferenceClient(self.rpc)

    def reference_client(self) -> InferenceClient:
        return InferenceClient(self.rpc, reference=True)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.rpc.close()
            self.statistics = self.rpc.statistics()
        finally:
            self.manager.shutdown()
            self.files.cleanup()
        logging.getLogger(__name__).info("Shared inference: %s", self.statistics)
        if exc_type is None and self.statistics["error"] is not None:
            raise RuntimeError(f"Shared inference failed: {self.statistics['error']}")
