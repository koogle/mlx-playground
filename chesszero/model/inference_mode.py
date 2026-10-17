"""Select search normalization without changing checkpoint statistics."""

from contextlib import contextmanager


def prediction_cache_safe(config):
    return not config.search_batch_norm or (
        config.search_batch_norm_per_position and config.inference_pad_batches
    )


def predict_search_batch(network, states, config):
    """Keep CUDA kernel shapes fixed when independent normalization is selected."""
    if not config.inference_pad_batches:
        return network(states)
    if not config.search_batch_norm_per_position:
        raise ValueError("Padded inference requires independent position statistics")
    import mlx.core as mx

    count = len(states)
    if not 0 < count <= config.inference_batch_size:
        raise ValueError("Prediction batch exceeds configured padded shape")
    if count < config.inference_batch_size:
        states = mx.concatenate(
            (
                states,
                mx.zeros(
                    (config.inference_batch_size - count, *states.shape[1:]),
                    dtype=states.dtype,
                ),
            )
        )
    policies, values = network(states)
    return policies[:count], values[:count]


@contextmanager
def search_inference_mode(network, batch_statistics, per_position=False):
    # Import only in the inference owner; CPU game workers must not load MLX.
    import mlx.nn as nn

    was_training = network.training
    batch_norms = (
        [layer for layer in network.modules() if isinstance(layer, nn.BatchNorm)]
        if hasattr(network, "modules")
        else []
    )
    tracking = [layer.track_running_stats for layer in batch_norms]
    position_modes = [
        getattr(layer, "_search_per_position", None) for layer in batch_norms
    ]
    try:
        network.eval()
        if batch_statistics:
            # MLX uses current batch/spatial statistics when tracking is off,
            # even in eval mode. This keeps search from mutating the teacher.
            for layer in batch_norms:
                layer.track_running_stats = False
                if per_position:
                    if not hasattr(layer, "_search_per_position"):
                        raise TypeError(
                            "Per-position search needs SearchBatchNorm layers"
                        )
                    layer._search_per_position = True
        yield
    finally:
        for layer, original in zip(batch_norms, tracking):
            layer.track_running_stats = original
        for layer, original in zip(batch_norms, position_modes):
            if original is not None:
                layer._search_per_position = original
        network.train(was_training)
