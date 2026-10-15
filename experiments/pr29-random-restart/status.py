"""Read persisted status without attaching to or interrupting the training job."""

import json
from pathlib import Path
import modal
import truststore

truststore.inject_into_ssl()
launch = json.loads(Path(__file__).with_name("launch.json").read_text())
volume = modal.Volume.from_name(launch["volume"])
for name in (
    "heartbeat.json",
    "generations.json",
    "initial-vs-random.json",
    "trained-vs-random.json",
    "job.json",
    "result.json",
):
    try:
        data = b"".join(volume.read_file(launch["remote_path"] + "/" + name)).decode()
        print(name, data[-5000:])
    except FileNotFoundError:
        pass
try:
    print(
        "run.log tail:",
        b"".join(volume.read_file(launch["remote_path"] + "/run.log")).decode()[-6000:],
    )
except FileNotFoundError:
    pass
