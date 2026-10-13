"""Run the controlled full-size chess learning diagnostic on one Modal L4."""

import io
from pathlib import Path
import zipfile
import modal
import os

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
    )
    .add_local_dir(export / "candidate", "/opt/chesszero", copy=True)
)
volume = modal.Volume.from_name("chesszero-convergence", create_if_missing=True)

app = modal.App("chesszero-tactical-learning")


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=32768,
    timeout=960,
    max_containers=1,
    scaledown_window=2,
    volumes={"/results": volume},
)
def run(run_id: str, bn_check: bool = False):
    import json
    import subprocess
    import sys
    import time
    import signal
    import os

    out = Path("/results") / run_id
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    with (out / "run.log").open("w") as stream:
        p = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "/opt/chesszero/benchmarks/chesszero/tactical_learning.py",
                "--repo",
                "/opt/chesszero",
                "--output",
                str(out),
            ]
            + (["--bn-check", "--steps", "20"] if bn_check else []),
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        timed_out = False
        while p.poll() is None:
            if time.monotonic() - started > 900:
                timed_out = True
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
                break
            time.sleep(20)
            volume.commit()
            progress = {}
            for f in out.glob("202*-*.json"):
                try:
                    r = json.loads(f.read_text())
                    last = r["rows"][-1]
                    progress[f.stem] = {
                        "step": last["step"],
                        "eval": last["probe"]["eval"],
                    }
                except (ValueError, IndexError):
                    pass
            print(
                json.dumps(
                    {"elapsed": round(time.monotonic() - started), "progress": progress}
                ),
                flush=True,
            )
    (out / "job.json").write_text(
        json.dumps(
            {
                "exit_code": p.returncode,
                "timed_out": timed_out,
                "elapsed_seconds": time.monotonic() - started,
            }
        )
    )
    volume.commit()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        for f in out.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(out))
    return buffer.getvalue()


@app.local_entrypoint()
def main(run_id: str, output: str, bn_check: bool = False):
    data = run.remote(run_id, bn_check)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(out)
    print("Results:", out)
