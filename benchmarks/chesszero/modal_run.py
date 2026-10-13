"""Modal GPU comparison; upload git-archive exports, never workspace credentials.

Set CHESSZERO_EXPORT_DIR to a directory containing baseline/ and candidate/,
each with a REVISION file, then run `modal run modal_run.py`.
"""

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
    )
    .add_local_dir(export, "/opt/chesszero", copy=True)
)
app = modal.App("chesszero-comparison")


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=16384,
    timeout=1800,
    max_containers=1,
    scaledown_window=2,
)
def compare(probe_only: bool = False):
    import subprocess
    import sys
    import time

    root = Path("/opt/chesszero")
    out = Path("/tmp/chesszero-results")
    out.mkdir()
    start = time.time()
    # Verify actual CUDA convolution/autodiff and the integrated regression suite
    # before spending time on full self-play runs.
    with (out / "tests.log").open("w") as handle:
        check = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"],
            cwd=root / "candidate/chesszero",
            stdout=handle,
            stderr=subprocess.STDOUT,
            timeout=180,
        )
    print("GPU regression exit:", check.returncode, flush=True)
    print((out / "tests.log").read_text()[-6000:], flush=True)
    (out / "packages.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    )
    if check.returncode == 0 and not probe_only:
        script = str(root / "candidate/benchmarks/chesszero/compare.py")
        common = [
            sys.executable,
            script,
            "--backend",
            "cuda",
            "--baseline",
            str(root / "baseline"),
            "--candidate",
            str(root / "candidate"),
        ]
        for name, options in (
            (
                "smoke",
                [
                    "--profile",
                    "smoke",
                    "--simulations",
                    "32",
                    "--games",
                    "2",
                    "--generations",
                    "3",
                    "--workers",
                    "2",
                    "--batch-size",
                    "64",
                    "--timeout",
                    "300",
                ],
            ),
            (
                "default-search",
                [
                    "--profile",
                    "default",
                    "--task",
                    "search",
                    "--simulations",
                    "32",
                    "--timeout",
                    "180",
                ],
            ),
        ):
            print("Starting", name, "elapsed", round(time.time() - start), flush=True)
            subprocess.run(
                common + ["--output", str(out / name)] + options,
                check=False,
                timeout=1100,
            )
            for path in sorted((out / name).glob("*.json")):
                print(path.name, path.read_text()[-5000:], flush=True)
            for path in sorted((out / name).glob("*.log")):
                print(path.name, "log tail:", path.read_text()[-1500:], flush=True)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in out.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(out))
    return buffer.getvalue()


@app.local_entrypoint()
def main(output: str = "/workspace/chesszero-modal-results", probe_only: bool = False):
    payload = compare.remote(probe_only)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(destination)
    print("Results saved to", destination)
