"""Test higher CPU-game concurrency with one L40S inference process."""

import modal
from modal_batched_throughput import image, volume, experiment

app = modal.App("chesszero-rpc-throughput-scale")


@app.function(image=image, gpu="L40S", cpu=32, memory=196608,
              timeout=3900, scaledown_window=2, volumes={"/results": volume})
def games512(run_id):
    return experiment("L40S", run_id, 512, limit_seconds=3600)


@app.function(image=image, gpu="L40S", cpu=32, memory=393216,
              timeout=7500, scaledown_window=2, volumes={"/results": volume})
def games1024(run_id):
    return experiment("L40S", run_id, 1024, limit_seconds=7200)


@app.function(image=image, gpu="L40S", cpu=32, memory=786432,
              timeout=14700, scaledown_window=2, volumes={"/results": volume})
def games2048(run_id):
    return experiment("L40S", run_id, 2048, limit_seconds=14400)
