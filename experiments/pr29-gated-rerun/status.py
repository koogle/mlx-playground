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
for name in ("heartbeat.json", "progress.json", "job.json", "result.json"):
    try:
        obj = json.loads(b"".join(volume.read_file(root + "/" + name)))
        if name == "progress.json":
            obj = {k: v for k, v in obj.items() if k != "generations"}
        print(name, json.dumps(obj))
    except FileNotFoundError:
        pass
for target in range(500, 3001, 500):
    try:
        print(
            "MILESTONE",
            b"".join(volume.read_file(root + f"/milestone-{target:04d}.json")).decode(),
        )
    except FileNotFoundError:
        pass
try:
    print("LOG_TAIL", b"".join(volume.read_file(root + "/run.log")).decode()[-1800:])
except FileNotFoundError:
    pass
