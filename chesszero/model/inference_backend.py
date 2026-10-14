"""GPU inference implementation, imported only when starting the backend."""

from concurrent.futures import Future
from queue import Queue, Empty
import threading

import mlx.core as mx
import numpy as np
from cachetools import LRUCache
from model.network import ChessNet


class _InferenceBackend:
    """The manager constructs this object inside its dedicated GPU process."""

    def __init__(self, config, current_path, reference_path, device):
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
        mx.set_default_device(self.device)
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
