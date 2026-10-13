"""Test higher CPU-game concurrency with one L40S inference process."""

from pathlib import Path
import modal
from modal_batched_throughput import image, volume, experiment

image = image.add_local_python_source("modal_batched_throughput", copy=True)

app = modal.App("chesszero-rpc-throughput-scale")


@app.function(image=image, gpu="L40S", cpu=32, memory=(65536, 196608),
              timeout=3900, scaledown_window=2, volumes={"/results": volume})
def games512(run_id):
    return experiment("L40S", run_id, 512, limit_seconds=3600)


@app.function(image=image, gpu="L40S", cpu=32, memory=(65536, 393216),
              timeout=7500, scaledown_window=2, volumes={"/results": volume})
def games1024(run_id):
    return experiment("L40S", run_id, 1024, limit_seconds=7200)


@app.function(image=image, gpu="L40S", cpu=32, memory=(65536, 786432),
              timeout=14700, scaledown_window=2, volumes={"/results": volume})
def games2048(run_id):
    return experiment("L40S", run_id, 2048, limit_seconds=14400)


@app.function(image=image, gpu="L40S", cpu=32, memory=65536,
              timeout=86400, scaledown_window=2, volumes={"/results": volume})
def probe(run_id, workers, limit_seconds):
    """Allow the controller to keep doubling beyond the prepared stages."""
    return experiment("L40S", run_id, workers, limit_seconds)


controller_image = image.add_local_file(
    str(Path(__file__).with_name("capacity_sweep.py")),
    "/opt/capacity_sweep.py", copy=True,
)


@app.function(image=controller_image, cpu=0.25, memory=512,
              timeout=86400, scaledown_window=2, volumes={"/results": volume})
def sweep(rows, directory):
    import json
    import os
    import runpy
    from pathlib import Path

    root = Path("/results") / directory
    root.mkdir(parents=True, exist_ok=True)
    (root / "capacity-runs.json").write_text(json.dumps(rows, indent=2))
    os.environ["CHESSZERO_CAPACITY_DIR"] = str(root)
    os.environ["CHESSZERO_CAPACITY_COMMIT"] = "1"
    volume.commit()
    runpy.run_path("/opt/capacity_sweep.py", run_name="__main__")
    return json.loads((root / "capacity-runs.json").read_text())
