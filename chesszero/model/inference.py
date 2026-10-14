"""Queued model RPC in one dedicated process, using local manager IPC."""

from __future__ import annotations

import logging
import multiprocessing as mp
from multiprocessing.managers import BaseManager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import numpy as np
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


def _create_backend(config, current_path, reference_path, device):
    # This factory runs in the server process. Keeping the backend import here
    # lets CPU workers import the RPC interface without initializing MLX.
    from model.inference_backend import _InferenceBackend

    return _InferenceBackend(config, current_path, reference_path, device)


class _InferenceManager(BaseManager):
    pass


_InferenceManager.register(
    "backend",
    _create_backend,
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
