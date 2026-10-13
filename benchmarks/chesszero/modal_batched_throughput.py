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
app = modal.App("chesszero-rpc-throughput-final")


def experiment(label, run_id, workers, limit_seconds=1800):
    import psutil

    out = Path("/results") / run_id / label
    out.mkdir(parents=True, exist_ok=False)
    samples = []
    observed_processes = {}
    start = time.monotonic()
    with (out / "run.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", "/opt/throughput.py", str(out)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=dict(os.environ, TQDM_DISABLE="1", CHESSZERO_WORKERS=str(workers)),
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
            sample = {"seconds": time.monotonic() - start, "gpu": gpu}
            sample["gpu_processes"] = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
                text=True,
            ).strip().splitlines()
            # Container cgroup memory includes the server and every CPU worker.
            memory_file = Path("/sys/fs/cgroup/memory.current")
            if not memory_file.exists():
                memory_file = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")
            if memory_file.exists():
                sample["container_memory_bytes"] = int(memory_file.read_text())
            cpu_file = Path("/sys/fs/cgroup/cpu.stat")
            if cpu_file.exists():
                sample["cpu_usage_usec"] = int(dict(
                    line.split() for line in cpu_file.read_text().splitlines()
                )["usage_usec"])
            for child in [psutil.Process(process.pid), *psutil.Process(process.pid).children(recursive=True)]:
                observed_processes.setdefault(child.pid, child)
            rss, cpu_percent = 0, 0.0
            for child in observed_processes.values():
                try:
                    rss += child.memory_info().rss
                    cpu_percent += child.cpu_percent()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            # RSS sums count shared pages more than once; retain this distinction.
            sample["process_tree_rss_bytes"] = rss
            sample["process_tree_cpu_cores"] = cpu_percent / 100
            samples.append(sample)
            (out / "gpu-samples.json").write_text(json.dumps(samples))
            volume.commit()
            print(label, round(time.monotonic() - start), gpu, flush=True)
            if time.monotonic() - start > limit_seconds:
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
def l40s(run_id, workers=16):
    return experiment("L40S", run_id, workers)


@app.function(
    image=image,
    gpu="H100",
    cpu=32,
    memory=65536,
    timeout=2100,
    scaledown_window=2,
    volumes={"/results": volume},
)
def h100(run_id, workers=16):
    return experiment("H100", run_id, workers)
