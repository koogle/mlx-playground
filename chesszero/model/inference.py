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

import numpy as np
from cachetools import LRUCache
from config.model_config import ModelConfig

if TYPE_CHECKING:
    from model.network import ChessNet


class InferenceClient:
    """Inference-only RPC handle; game workers receive no model or GPU tensors."""

    def __init__(self, rpc, reference=False):
        self.rpc, self.reference = rpc, reference

    def __call__(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        position = np.asarray(states[0], dtype=np.float32)
        predict = self.rpc.predict_reference if self.reference else self.rpc.predict
        policy, value = predict(position)
        return policy, value

    def close(self):
        # The manager owns the backend; releasing a client must not close it.
        self.rpc = None


class _InferenceBackend:
    """The manager constructs this object inside its dedicated GPU process."""

    def __init__(self, config, current_path, reference_path, device):
        # CPU workers import this module for RPC handles; load MLX only here.
        import mlx.core as mx
        from model.network import ChessNet

        self.mx = mx

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

    def _predict(self, pending):
        mx = self.mx
        for snapshot in {key[0] for key in pending}:
            keys = [key for key in pending if key[0] == snapshot]
            self.model.update(self.snapshots[snapshot])
            states = mx.array(np.stack([pending[key][0] for key in keys]))
            policies, values = self.model(states)
            mx.eval(policies, values)
            policies, values = np.array(policies), np.array(values)
            self.forward_passes += 1
            self.positions += len(keys)
            self.max_batch_size = max(self.max_batch_size, len(keys))
            for index, key in enumerate(keys):
                prediction = (policies[index : index + 1], values[index : index + 1])
                self.predictions[key] = prediction
                # Resolve every duplicate directly, even if later entries evict
                # this prediction from a cache smaller than the GPU batch.
                for response in pending[key][1]:
                    response.set_result(prediction)

    def _serve(self):
        self.mx.set_default_device(self.device)
        while True:
            request = self.requests.get()  # Block only when the queue has no work.
            if request is None:
                return
            pending = {}
            try:
                while True:
                    snapshot, position, response = request
                    if self.error is not None:
                        raise RuntimeError(self.error)
                    # The limit counts unique GPU inputs, not cache hits or
                    # duplicate requests. Only this executor accesses the LRU.
                    key = (snapshot, position.shape, position.tobytes())
                    if key in self.predictions:
                        self.cache_hits += 1
                        response.set_result(self.predictions[key])
                    elif key in pending:
                        self.cache_hits += 1
                        pending[key][1].append(response)
                    else:
                        pending[key] = (position, [response])
                    if len(pending) == self.batch_size:
                        break
                    try:
                        request = self.requests.get_nowait()
                    except Empty:
                        break  # Dispatch available misses without a timer.
                    if request is None:
                        self.requests.put(None)
                        break
                if pending:
                    self._predict(pending)
            except Exception as error:
                self.error = str(error)
                responses = [r for _, waiters in pending.values() for r in waiters]
                if request is not None:
                    responses.append(request[2])
                for response in responses:
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
        import mlx.core as mx

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
