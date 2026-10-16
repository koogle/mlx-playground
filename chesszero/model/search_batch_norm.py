"""BatchNorm with optional independent spatial statistics for search positions."""

import mlx.core as mx
import mlx.nn as nn


class SearchBatchNorm(nn.BatchNorm):
    """Training is ordinary BN; search can reproduce batch-size-one BN in parallel.

    This preserves existing checkpoint tensor names and learned affine weights.
    Independent spatial reductions prevent other games in an inference batch
    from changing the policy/value assigned to a position.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._search_per_position = False

    def __call__(self, x):
        if not self._search_per_position:
            return super().__call__(x)
        if x.ndim not in (3, 4):
            raise ValueError("Per-position normalization needs spatial dimensions")
        axes = tuple(range(1, x.ndim - 1))
        mean = mx.mean(x, axis=axes, keepdims=True)
        variance = mx.var(x, axis=axes, keepdims=True)
        normalized = (x - mean) * mx.rsqrt(variance + self.eps)
        return self.weight * normalized + self.bias if "weight" in self else normalized
