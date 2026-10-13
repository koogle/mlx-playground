"""Read persisted progress without starting another GPU allocation.

Usage: python benchmarks/chesszero/learning_status.py RUN_ID [--output DIRECTORY]
Requires the Modal SDK and configured Modal credentials.
"""

import argparse
import json
from pathlib import Path

import modal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    volume = modal.Volume.from_name("chesszero-convergence")
    records = {}
    names = ["experiment.json", "generations.json", "job.json", "completion.json"]
    for entry in volume.listdir(args.run_id):
        name = Path(entry.path).name
        if name.startswith("scores-") and name.endswith(".json"):
            names.append(name)
    for name in names:
        try:
            data = b"".join(volume.read_file(f"{args.run_id}/{name}"))
        except FileNotFoundError:
            continue
        records[name] = json.loads(data)
        if args.output:
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / name).write_bytes(data)
    rows = records.get("generations.json", [])
    summary = {
        "run_id": args.run_id,
        "completed_updates": len(rows),
        "completed_training_games": sum(len(row["game_plies"]) for row in rows),
        "training_positions": sum(row["positions"] for row in rows),
        "latest": rows[-1] if rows else None,
        "evaluations": {k: v for k, v in records.items() if k.startswith("scores-")},
        "job": records.get("job.json"),
        "completion": records.get("completion.json"),
        "note": "Snapshots update about every 30 seconds; absent completion is not proof of liveness.",
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
