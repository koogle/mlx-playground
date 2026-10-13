"""Compare complete fresh self-play generations on L40S and H100."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import modal

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("libgomp1")
    .pip_install(
        "mlx[cuda12]==0.32.3",
        "numpy==2.2.6",
        "psutil==7.0.0",
        "tqdm==4.67.1",
        "cachetools==7.2.1",
    )
    .add_local_dir(
        os.environ.get("CHESSZERO_EXPORT_DIR", "/workspace/chesszero-batched-export")
        + "/candidate",
        "/opt/chesszero",
        copy=True,
    )
    .add_local_file(
        str(Path(__file__).with_name("batched_gpu_throughput.py")),
        "/opt/throughput.py",
        copy=True,
    )
)
volume = modal.Volume.from_name("chesszero-convergence")
app = modal.App("chesszero-batched-throughput")


def experiment(label, run_id):
    out = Path("/results") / run_id / label
    out.mkdir(parents=True, exist_ok=False)
    samples = []
    start = time.monotonic()
    with (out / "run.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", "/opt/throughput.py", str(out)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=dict(os.environ, TQDM_DISABLE="1"),
            start_new_session=True,
        )
        while process.poll() is None:
            gpu = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()
            samples.append({"seconds": time.monotonic() - start, "gpu": gpu})
            (out / "gpu-samples.json").write_text(json.dumps(samples))
            volume.commit()
            print(label, round(time.monotonic() - start), gpu, flush=True)
            if time.monotonic() - start > 1800:
                import signal

                os.killpg(process.pid, signal.SIGTERM)
                process.wait()
                break
            time.sleep(15)
    (out / "job.json").write_text(
        json.dumps(
            {
                "exit_code": process.returncode,
                "elapsed_seconds": time.monotonic() - start,
            }
        )
    )
    volume.commit()
    if process.returncode:
        raise RuntimeError((out / "run.log").read_text()[-4000:])
    return json.loads((out / "result.json").read_text())


@app.function(
    image=image,
    gpu="L40S",
    cpu=32,
    memory=65536,
    timeout=2100,
    scaledown_window=2,
    volumes={"/results": volume},
)
def l40s(run_id):
    return experiment("L40S", run_id)


@app.function(
    image=image,
    gpu="H100",
    cpu=32,
    memory=65536,
    timeout=2100,
    scaledown_window=2,
    volumes={"/results": volume},
)
def h100(run_id):
    return experiment("H100", run_id)
