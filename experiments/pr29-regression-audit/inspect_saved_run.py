import json
import time
import collections
import sys
from pathlib import Path
import chess
import numpy as np
import mlx.core as mx
from model.network import ChessNet
from config.model_config import ModelConfig
from model.move_encoding import encode_move
from test_board import from_fen, coords

root = Path(sys.argv[1])
run = root / "pr29-3000-games-20261015-145659"
report = {}
for filename in ["vs-random-3000-games.json", "final-vs-initial-games.json"]:
    rows = json.loads((run / filename).read_text())
    term = collections.Counter()
    for r in rows:
        b = chess.Board()
        for u in r["moves"]:
            m = chess.Move.from_uci(u)
            assert m in b.legal_moves
            b.push(m)
        o = b.outcome(claim_draw=True)
        expected = (
            None
            if o is None
            else 0
            if o.winner is None
            else 1
            if o.winner == (r["color"] == 0)
            else -1
        )
        assert r["outcome"] == expected
        term[o.termination.name if o else "censored"] += 1
    report[filename] = {
        "games_verified": len(rows),
        "terminations": dict(term),
        "mean_plies": float(np.mean([len(r["moves"]) for r in rows])),
    }
    print(filename, report[filename], flush=True)
# Evaluate frozen networks at identical positions, including forced mates of both colors.
fens = [
    chess.STARTING_FEN,
    "7k/8/5K2/6Q1/8/8/8/8 b - - 0 1",
    "8/7k/5K2/6Q1/8/8/8/8 w - - 1 2",
]
rows = json.loads((run / "final-vs-initial-games.json").read_text())
for r in rows[:4]:
    b = chess.Board()
    for i, u in enumerate(r["moves"]):
        if i in (8, 16, 32, 48):
            fens.append(b.fen())
        b.push_uci(u)
positions = []
for fen in fens:
    b = chess.Board(fen)
    for mirror in (False, True):
        ref = b.mirror() if mirror else b
        native, _ = from_fen(ref.fen())
        if mirror:
            native.state[17] = positions[-1][1].state[17]
        positions.append((ref, native))
states = mx.array(np.stack([n.state for _, n in positions]))
for label, source, epoch in [
    ("initial", root / "pr29-random-restart-20261015-085706", 0),
    ("final", run, 106),
]:
    manifest = json.loads(
        (source / f"checkpoints/state_epoch_{epoch}.json").read_text()
    )
    model = ChessNet(ModelConfig(**manifest["config"]))
    model.load_weights(str(source / "checkpoints" / manifest["model_file"]))
    model.eval()
    start = time.monotonic()
    ps, vs = model(states)
    mx.eval(ps, vs)
    ps, vs = np.asarray(ps), np.asarray(vs).ravel()
    data = []
    for (b, n), p, v in zip(positions, ps, vs):
        legal = list(b.legal_moves)
        ids = [encode_move(*coords(m)) for m in legal]
        q = p[ids]
        q = q / q.sum()
        data.append(
            {
                "fen": b.fen(),
                "value": float(v),
                "legal_mass": float(p[ids].sum()),
                "legal_entropy": float(-np.sum(q * np.log(q + 1e-30))),
                "top_legal": legal[int(q.argmax())].uci(),
                "top_probability": float(q.max()),
            }
        )
    report[label] = {
        "seconds": time.monotonic() - start,
        "mirror_discrepancy_mean": float(np.mean(np.abs(vs[::2] - vs[1::2]))),
        "positions": data,
        "value_mean": float(vs.mean()),
        "value_std": float(vs.std()),
    }
    print(label, json.dumps(report[label]), flush=True)
    del model
    getattr(mx, "clear_cache", lambda: None)()
(root / "probe-results.json").write_text(json.dumps(report, indent=2) + "\n")
