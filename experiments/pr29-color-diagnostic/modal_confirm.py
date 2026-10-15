"""Frozen color-imbalance interventions on an independent L40S."""

from pathlib import Path
import sys
import modal

ROOT = (
    Path(__file__).resolve().parents[2] if modal.is_local() else Path("/opt/chesszero")
)
volume = modal.Volume.from_name("chesszero-convergence", create_if_missing=True)
image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("libgomp1")
    .pip_install(
        "mlx[cuda12]==0.32.3",
        "numpy==2.2.6",
        "psutil==7.0.0",
        "tqdm==4.67.1",
        "cachetools==7.2.1",
        "python-chess==1.999",
        "pytest==8.4.2",
    )
    .env(
        {
            "MLX_CUDA_CONV_CACHE_SIZE": "1024",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "TQDM_DISABLE": "1",
        }
    )
    .add_local_dir(
        str(ROOT / "chesszero"),
        "/opt/chesszero/chesszero",
        copy=True,
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
    .add_local_file(
        str(ROOT / "experiments/pr29-first-result/run_training.py"),
        "/opt/run_training.py",
        copy=True,
    )
    .add_local_file(
        str(ROOT / "experiments/pr29-first-result/preflight.py"),
        "/opt/preflight.py",
        copy=True,
    )
)

image = image.add_local_file(
    str(ROOT / "experiments/pr29-random-restart/restart_training.py"),
    "/opt/restart_training.py",
    copy=True,
)
image = image.add_local_file(
    str(Path(__file__).with_name("ablate.py")), "/opt/ablate.py", copy=True
)
image = image.add_local_file(
    str(Path(__file__).with_name("confirm_value.py")),
    "/opt/confirm_value.py",
    copy=True,
)
app = modal.App("chesszero-pr29-value-confirmation")


@app.function(
    image=image,
    gpu="L40S",
    cpu=32,
    memory=65536,
    timeout=1800,
    retries=0,
    volumes={"/results": volume},
)
def probe(run_id):
    import json
    import subprocess
    import time

    out = Path("/results") / run_id
    out.mkdir()
    start = time.monotonic()
    with (out / "run.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", "/opt/confirm_value.py", str(out)],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        while process.poll() is None:
            (out / "heartbeat.json").write_text(
                json.dumps(
                    dict(elapsed_seconds=time.monotonic() - start, process_running=True)
                )
            )
            volume.commit()
            if time.monotonic() - start > 1740:
                process.kill()
                process.wait()
                break
            time.sleep(15)
    (out / "job.json").write_text(
        json.dumps(
            dict(exit_code=process.returncode, elapsed_seconds=time.monotonic() - start)
        )
    )
    volume.commit()
    if process.returncode:
        raise RuntimeError((out / "run.log").read_text()[-5000:])
    return json.loads((out / "result.json").read_text())


@app.local_entrypoint()
def main():
    import datetime
    import json

    run_id = "pr29-value-confirmation-" + datetime.datetime.now(
        datetime.timezone.utc
    ).strftime("%Y%m%d-%H%M%S")
    call = probe.spawn(run_id)
    record = dict(
        run_id=run_id, function_call_id=call.object_id, volume="chesszero-convergence"
    )
    Path(__file__).with_name("confirmation-launch.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print(json.dumps(record), flush=True)
