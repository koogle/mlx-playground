"""Read only: live progress and completed 500-game reports."""

import json
from pathlib import Path

import modal
import truststore

truststore.inject_into_ssl()
record = json.loads(Path(__file__).with_name("launch.json").read_text())
volume = modal.Volume.from_name(record["volume"])
root = record["run_id"]
print("RUN", root)
try:
    cancellation = json.loads(b"".join(volume.read_file(root + "/cancellation.json")))
except FileNotFoundError:
    cancellation = None
if cancellation and cancellation.get("cancelled_deliberately"):
    print("CANCELLED_BY_USER", cancellation["reason"])
    progress = cancellation["progress"]
    print(
        "last_committed_progress",
        json.dumps({k: v for k, v in progress.items() if k != "generations"}),
    )
    raise SystemExit(0)
for name in (
    "heartbeat.json",
    "progress.json",
    "protocol.json",
    "job.json",
    "result.json",
):
    try:
        obj = json.loads(b"".join(volume.read_file(root + "/" + name)))
        if name == "progress.json":
            obj = {k: v for k, v in obj.items() if k != "generations"}
        print(name, json.dumps(obj))
    except FileNotFoundError:
        pass
try:
    print("LOG_TAIL", b"".join(volume.read_file(root + "/run.log")).decode()[-1800:])
except FileNotFoundError:
    pass
