"""Detached bounded repeat using the original experiment CUDA configuration."""

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
    str(ROOT / "experiments/pr29-restored-training/train_gated.py"),
    "/opt/train_gated.py",
    copy=True,
)
image = image.add_local_file(
    str(Path(__file__).with_name("matches.py")), "/opt/matches.py", copy=True
)
image = image.add_local_file(
    str(Path(__file__).with_name("study.py")), "/opt/study.py", copy=True
)
app = modal.App("chesszero-pr29-zero-direct")


@app.function(
    image=image,
    gpu="L40S",
    cpu=32,
    memory=(65536, 131072),
    timeout=18000,
    retries=0,
    scaledown_window=2,
    volumes={"/results": volume},
)
def restart(run_id, source_revision):
    import json
    import os
    import signal
    import subprocess
    import time

    out = Path("/results") / run_id
    out.mkdir(exist_ok=True)
    began = time.monotonic()
    position = 0
    timed_out = False
    with (out / "run.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", "/opt/study.py", str(out)],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=dict(os.environ, CHESSZERO_SOURCE_REVISION=source_revision),
        )
        try:
            while process.poll() is None:
                with (out / "run.log").open() as reader:
                    reader.seek(position)
                    data = reader.read()
                    position = reader.tell()
                if data:
                    print(data, end="", flush=True)
                elapsed = time.monotonic() - began
                (out / "heartbeat.json").write_text(
                    json.dumps(
                        dict(
                            elapsed_seconds=elapsed,
                            process_running=True,
                            latest_log=data.splitlines()[-1] if data else None,
                        )
                    )
                )
                volume.commit()
                if elapsed >= 16200:
                    timed_out = True
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=30)
                    break
                time.sleep(15)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            job = dict(
                exit_code=process.returncode,
                timed_out=timed_out,
                elapsed_seconds=time.monotonic() - began,
            )
            (out / "job.json").write_text(json.dumps(job, indent=2))
            (out / "heartbeat.json").write_text(
                json.dumps(dict(job, process_running=False))
            )
            volume.commit()
            print("JOB_EXIT", json.dumps(job), flush=True)
    if process.returncode:
        raise RuntimeError((out / "run.log").read_text()[-6000:])
    return json.loads((out / "result.json").read_text())


@app.local_entrypoint()
def main(resume_run: str = ""):
    import datetime
    import json
    import subprocess

    run_id = resume_run or "pr29-zero-direct-" + datetime.datetime.now(
        datetime.timezone.utc
    ).strftime("%Y%m%d-%H%M%S")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    call = restart.spawn(run_id, revision)
    record = dict(
        run_id=run_id,
        function_call_id=call.object_id,
        volume="chesszero-convergence",
        remote_path="/" + run_id,
        source_revision=revision,
        source_run="pr29-random-restart-20261015-085706",
        training_games=0,
        source_study="pr29-random-restart-20261015-085706",
        source_epochs=[0],
        confirmation_match_games=128,
        search_batch_norm=True,
        search_batch_norm_per_position=True,
        inference_pad_batches=True,
        inference_batch_size=16,
        recycle_promotion_games=False,
        policy_target="visit_distribution",
    )
    Path(__file__).with_name("launch.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print(json.dumps(record), flush=True)
