"""Launch: python /workspace/setup/modal/run.py run --detach modal_train.py"""

from pathlib import Path
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
        str(Path(__file__).with_name("run_training.py")),
        "/opt/run_training.py",
        copy=True,
    )
    .add_local_file(
        str(Path(__file__).with_name("preflight.py")), "/opt/preflight.py", copy=True
    )
)
app = modal.App("chesszero-pr29-first-result")


@app.function(
    image=image,
    gpu="L40S",
    cpu=32,
    memory=(65536, 131072),
    timeout=14700,
    scaledown_window=2,
    retries=0,
    volumes={"/results": volume},
)
def train(run_id, baseline_run="", preflight_run=""):
    import json
    import os
    import signal
    import subprocess
    import sys
    import time

    out = Path("/results") / run_id
    out.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    timed_out = False
    log_position = 0
    with (out / "run.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", "/opt/run_training.py", str(out)],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=dict(
                os.environ,
                CHESSZERO_BASELINE_RUN=baseline_run,
                CHESSZERO_PREFLIGHT_RUN=preflight_run,
            ),
        )
        try:
            while process.poll() is None:
                elapsed = time.monotonic() - began
                with (out / "run.log").open() as reader:
                    reader.seek(log_position)
                    new_output = reader.read()
                    log_position = reader.tell()
                if new_output:
                    print(new_output, end="", flush=True)
                print(f"RUNNING {run_id}: {elapsed:.0f}s elapsed", flush=True)
                (out / "heartbeat.json").write_text(
                    json.dumps(
                        dict(
                            elapsed_seconds=elapsed,
                            process_running=True,
                            latest_log=new_output.splitlines()[-1]
                            if new_output
                            else None,
                        )
                    )
                )
                volume.commit()
                if elapsed >= 14400:
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
            (out / "heartbeat.json").write_text(
                json.dumps(dict(job, process_running=False))
            )
            print("JOB_EXIT", json.dumps(job), flush=True)
            (out / "job.json").write_text(json.dumps(job, indent=2))
            volume.commit()
    if process.returncode:
        raise RuntimeError((out / "run.log").read_text()[-6000:])
    return json.loads((out / "result.json").read_text())


@app.local_entrypoint()
def main(baseline_run: str = "", preflight_run: str = ""):
    import datetime
    import json

    run_id = "pr29-first-result-" + datetime.datetime.now(
        datetime.timezone.utc
    ).strftime("%Y%m%d-%H%M%S")
    call = train.spawn(run_id, baseline_run, preflight_run)
    record = dict(
        run_id=run_id,
        function_call_id=call.object_id,
        volume="chesszero-convergence",
        remote_path="/" + run_id,
        baseline_run=baseline_run,
        preflight_run=preflight_run,
    )
    Path(__file__).with_name("launch.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print(json.dumps(record), flush=True)


@app.function(
    image=image,
    gpu="L40S",
    cpu=8,
    memory=32768,
    timeout=1200,
    scaledown_window=2,
    volumes={"/results": volume},
)
def preflight():
    import datetime, os, subprocess, sys, json

    run_id = "pr29-preflight-" + datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%d-%H%M%S"
    )
    out = Path("/results") / run_id
    out.mkdir()
    with (out / "run.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", "/opt/preflight.py", str(out)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=dict(os.environ, PYTHONPATH="/opt/chesszero/chesszero"),
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        process.wait()
    volume.commit()
    print("PREFLIGHT_ARTIFACTS", run_id, flush=True)
    if process.returncode:
        raise RuntimeError(
            f"Preflight failed with exit {process.returncode}; see {run_id}/run.log"
        )
    return json.loads((out / "result.json").read_text())
