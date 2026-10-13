"""Persist a bounded learning experiment and its resumable checkpoints on Modal."""

import io
import os
from pathlib import Path
import zipfile

import modal

export = Path(
    os.environ.get("CHESSZERO_EXPORT_DIR", "/workspace/chesszero-benchmark-export")
)
image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("libgomp1")
    .pip_install(
        "mlx[cuda12]==0.32.3",
        "numpy==2.2.6",
        "psutil==7.0.0",
        "tqdm==4.67.1",
        "python-chess==1.999",
        "cachetools==7.2.1",
    )
    .add_local_dir(export / "candidate", "/opt/chesszero", copy=True)
)
volume = modal.Volume.from_name("chesszero-convergence", create_if_missing=True)
app = modal.App("chesszero-convergence")


@app.function(
    image=image,
    gpu="L40S",
    cpu=16,
    memory=65536,
    timeout=43200,
    max_containers=1,
    scaledown_window=2,
    volumes={"/results": volume},
)
def run(
    run_id: str,
    resume_run: str = "",
    training_seconds: int = 39600,
    limit_seconds: int = 42900,
    resume_generation: int = 10,
    workers: int = 16,
    games_per_generation: int = 16,
):
    import json
    import signal
    import subprocess
    import sys
    import time

    if not 0 <= training_seconds < limit_seconds <= 42900:
        raise ValueError("Run must fit within the bounded 12-hour allocation")
    out = Path("/results") / run_id
    out.mkdir(exist_ok=False)
    (out / "packages.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    )
    (out / "gpu.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
    command = [
        sys.executable,
        "-u",
        "/opt/chesszero/benchmarks/chesszero/convergence.py",
        "--repo",
        "/opt/chesszero",
        "--output",
        str(out),
        "--training-deadline-seconds",
        str(training_seconds),
        "--generations",
        "500",
        "--games-per-generation",
        str(games_per_generation),
        "--workers",
        str(workers),
        "--simulations",
        "32",
    ]
    if resume_run:
        command.extend(
            [
                "--resume-run",
                str(Path("/results") / resume_run),
                "--resume-generation",
                str(resume_generation),
            ]
        )
    start = time.monotonic()
    timed_out = False
    with (out / "run.log").open("w") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=dict(os.environ, TQDM_DISABLE="1"),
        )
        while process.poll() is None:
            if time.monotonic() - start > limit_seconds:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                break
            time.sleep(30)
            volume.commit()
            path = out / "generations.json"
            try:
                rows = json.loads(path.read_text()) if path.exists() else []
            except (OSError, json.JSONDecodeError):
                rows = []
            print(
                json.dumps(
                    {
                        "elapsed_seconds": round(time.monotonic() - start),
                        "completed_generations": len(rows),
                        "latest": rows[-1] if rows else None,
                    }
                ),
                flush=True,
            )
    (out / "job.json").write_text(
        json.dumps(
            {"exit_code": process.returncode, "timed_out": timed_out, "run_id": run_id}
        )
    )
    volume.commit()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in out.rglob("*"):
            if path.is_file() and path.suffix != ".safetensors":
                archive.write(path, path.relative_to(out))
    return buffer.getvalue()


@app.local_entrypoint()
def main(
    run_id: str,
    output: str,
    resume_run: str = "",
    training_seconds: int = 39600,
    limit_seconds: int = 42900,
    resume_generation: int = 10,
    workers: int = 16,
    games_per_generation: int = 16,
):
    result = run.remote(
        run_id, resume_run, training_seconds, limit_seconds, resume_generation,
        workers, games_per_generation
    )
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(result)) as archive:
        archive.extractall(destination)
    print(
        "Results:",
        destination,
        "; checkpoints: Modal volume chesszero-convergence /",
        run_id,
    )
