"""Resume one queued probe, then double L40S game concurrency until failure."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import modal


ROOT = Path(os.environ.get("CHESSZERO_CAPACITY_DIR", Path(__file__).parent))
STATE = ROOT / "capacity-runs.json"
OUTPUT = ROOT / "results" / "capacity"
APP = "chesszero-rpc-throughput-scale"


def save(rows):
    STATE.write_text(json.dumps(rows, indent=2) + "\n")
    if os.environ.get("CHESSZERO_CAPACITY_COMMIT"):
        modal.Volume.from_name("chesszero-convergence").commit()


def main():
    rows = json.loads(STATE.read_text())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    volume = modal.Volume.from_name("chesszero-convergence")
    while True:
        run = rows[-1]
        workers = run["workers"]
        call = modal.FunctionCall.from_id(run["function_call_id"])
        try:
            while True:
                try:
                    result = call.get(timeout=45)
                    break
                except TimeoutError:
                    try:
                        samples = json.loads(b"".join(volume.read_file(
                            run["run_id"] + "/L40S/gpu-samples.json"
                        )))
                        run["latest_sample"] = samples[-1]
                        run["status"] = "running"
                    except FileNotFoundError:
                        run["status"] = "queued or provisioning"
                    save(rows)
                    print(workers, run["status"], run.get("latest_sample", ""), flush=True)
            if result["games"] != workers or result["failures"] or not result["finite_parameters"]:
                raise RuntimeError("Probe did not complete every game and a finite update")
            if not result["source_revision"].startswith("96cb5f6"):
                raise RuntimeError("Probe source changed; results are not comparable")
            (OUTPUT / f"result-{workers}.json").write_text(json.dumps(result, indent=2) + "\n")
            samples = b"".join(volume.read_file(run["run_id"] + "/L40S/gpu-samples.json"))
            (OUTPUT / f"gpu-samples-{workers}.json").write_bytes(samples)
            run["status"] = "completed"
            run["games_per_minute"] = result["games_per_minute"]
            save(rows)
            print(workers, "COMPLETED", result["games_per_minute"], "games/min", flush=True)
        except Exception as error:
            run["status"] = "failed or blocked"
            run["error"] = str(error)[-6000:]
            save(rows)
            try:
                (OUTPUT / f"failure-{workers}.log").write_bytes(b"".join(
                    volume.read_file(run["run_id"] + "/L40S/run.log")
                ))
            except FileNotFoundError:
                pass
            print(workers, "STOPPED", run["error"], flush=True)
            return

        workers *= 2
        limit = min(86100, 3600 * workers // 512)
        run = dict(workers=workers, source_revision="96cb5f6", cpu=32, gpu="L40S",
                   memory_request_mb=65536, memory_limit_mb=196608 * workers // 512,
                   limit_seconds=limit)
        run["run_id"] = f"scale-rpc-22-{workers}-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        rows.append(run)
        try:
            function = modal.Function.from_name(APP, "probe").with_options(
                memory=(run["memory_request_mb"], run["memory_limit_mb"]),
                timeout=limit + 300, volumes={"/results": volume},
            )
            call = function.spawn(run["run_id"], workers, limit)
            run["function_call_id"] = call.object_id
            run["status"] = "submitted"
        except Exception as error:
            run["status"] = "resource request rejected"
            run["error"] = str(error)
            save(rows)
            print(workers, "RESOURCE REQUEST REJECTED", run["error"], flush=True)
            return
        save(rows)
        print(workers, "SUBMITTED", call.object_id, flush=True)


if __name__ == "__main__":
    main()
