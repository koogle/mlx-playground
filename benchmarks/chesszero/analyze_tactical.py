"""Independently replay the controlled games and plot raw-network learning."""

import argparse
import json
from pathlib import Path
import chess
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tactical_learning import FEN


def analyze(root):
    runs = json.loads((root / "results.json").read_text())
    checked = 0
    summaries = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for run in runs:
        for generation in run["games"]:
            for game in generation["games"]:
                board = chess.Board(FEN)
                if game["mirror"]:
                    board = board.mirror()
                for uci in game["moves"]:
                    move = chess.Move.from_uci(uci)
                    assert move in board.legal_moves
                    board.push(move)
                outcome = board.outcome(claim_draw=True)
                assert outcome is not None
                assert game["white_result"] == (
                    0 if outcome.winner is None else 1 if outcome.winner else -1
                )
                checked += 1
        first, last = run["rows"][0], run["rows"][-1]
        summaries.append(
            {
                "seed": run["seed"],
                "mode": run["mode"],
                "seconds": last["elapsed_seconds"],
                "steps": last["step"],
                "initial": first["probe"]["eval"],
                "final": last["probe"]["eval"],
                "finite_parameters": run["finite_parameters"],
                "bn_recalibration": run.get("bn_recalibration"),
                "first_probe_greedy_both": next(
                    (
                        r["step"]
                        for r in run["rows"]
                        if all(f["greedy_mate"] for f in r["probe"]["eval"]["finishes"])
                    ),
                    None,
                ),
                "games": sum(len(g["games"]) for g in run["games"]),
                "positions": sum(
                    len(g["moves"])
                    for generation in run["games"]
                    for g in generation["games"]
                ),
                "all_two_plies": all(
                    len(g["moves"]) == 2
                    for generation in run["games"]
                    for g in generation["games"]
                ),
            }
        )
        style = "-" if run["mode"] == "self_play" else "--"
        color = "#2361bd" if run["seed"] == 2026 else "#bd6423"
        label = f"{run['mode'].replace('_',' ')} · seed {run['seed']}"
        xs = [r["step"] for r in run["rows"]]
        ys = [
            min(f["mate_probability_legal"] for f in r["probe"]["eval"]["finishes"])
            for r in run["rows"]
        ]
        axes[0].plot(xs, ys, style, color=color, label=label, marker=".", alpha=0.85)
        axes[1].semilogy(
            xs,
            [max(r["probe"]["eval"]["value_mse"], 1e-12) for r in run["rows"]],
            style,
            color=color,
            marker=".",
            alpha=0.85,
        )
    axes[0].axhline(0.04, color="gray", linestyle=":", label="uniform legal move: 4%")
    axes[0].set(
        ylabel="Mating-move probability among legal moves",
        xlabel="Adam updates",
        ylim=(-0.03, 1.05),
        title="Raw network, no MCTS; minimum across both colors",
    )
    axes[1].set(
        ylabel="Value MSE (evaluation mode)",
        xlabel="Adam updates",
        title="Four states: losing −1 / winning +1 for both colors",
    )
    axes[0].legend(fontsize=8, loc="lower right")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle("Controlled chess learning · original 19-block / 256-filter model")
    fig.savefig(root / "tactical-learning.png", dpi=160)
    fig.savefig(root / "tactical-learning.svg")
    result = {"verified_games": checked, "runs": summaries}
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    analyze(parser.parse_args().results)
