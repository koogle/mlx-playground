"""Select search normalization without changing checkpoint statistics."""

from contextlib import contextmanager


@contextmanager
def search_inference_mode(network, batch_statistics):
    # Import only in the inference owner; CPU game workers must not load MLX.
    import mlx.nn as nn

    was_training = network.training
    batch_norms = (
        [layer for layer in network.modules() if isinstance(layer, nn.BatchNorm)]
        if hasattr(network, "modules")
        else []
    )
    tracking = [layer.track_running_stats for layer in batch_norms]
    try:
        network.eval()
        if batch_statistics:
            # MLX uses current batch/spatial statistics when tracking is off,
            # even in eval mode. This keeps search from mutating the teacher.
            for layer in batch_norms:
                layer.track_running_stats = False
        yield
    finally:
        for layer, original in zip(batch_norms, tracking):
            layer.track_running_stats = original
        network.train(was_training)
