"""Independently verify evaluation games and plot a saved learning experiment."""

import argparse
import json
from pathlib import Path

import chess  # Analysis oracle only. Never used to choose moves or train.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def verify_games(root):
    checked, errors = 0, []
    for path in sorted(root.glob("evaluation-*-*.json")):
        for game_id, row in enumerate(json.loads(path.read_text())):
            board = chess.Board()
            try:
                for source, target in row["moves"]:
                    move = chess.Move(
                        chess.square(source[1], source[0]),
                        chess.square(target[1], target[0]),
                        promotion=target[2] + 1 if len(target) == 3 else None,
                    )
                    if move not in board.legal_moves:
                        raise ValueError(f"Illegal move {move.uci()} at {board.fen()}")
                    board.push(move)
                if row["score"] is not None:
                    outcome = board.outcome(claim_draw=True)
                    if outcome is None:
                        raise ValueError("Native engine ended a nonterminal position")
                    expected = (
                        0.5
                        if outcome.winner is None
                        else float(outcome.winner == (row["candidate_color"] == 0))
                    )
                    if row["score"] != expected:
                        raise ValueError(
                            f"Native score {row['score']} != oracle {expected}"
                        )
                checked += 1
            except ValueError as error:
                errors.append({"file": path.name, "game": game_id, "error": str(error)})
    return {"games_verified": checked, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    root = args.results
    verification = verify_games(root)
    (root / "oracle-verification.json").write_text(
        json.dumps(verification, indent=2) + "\n"
    )
    if verification["errors"]:
        raise RuntimeError(json.dumps(verification))
    rows = json.loads((root / "generations.json").read_text())
    evaluations = {
        int(p.stem.split("-")[1]): json.loads(p.read_text())
        for p in sorted(root.glob("scores-*.json"))
    }
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)
    config = json.loads((root / "experiment.json").read_text())["config"]
    fig.suptitle(
        f"ChessZero: {config['n_residual_blocks']} residual blocks / "
        f"{config['n_filters']} filters, {config['n_simulations']} MCTS simulations",
        fontsize=14,
    )
    for opponent, color in [("initial", "#355cbb"), ("random", "#ce741b")]:
        points = [(g, s[opponent]) for g, s in evaluations.items() if opponent in s]
        x = [g for g, _ in points]
        y = [s["score"] for _, s in points]
        axes[0].plot(x, y, "o-", color=color, label=f"vs {opponent}")
        for g, score in points:
            if score["censored"]:
                lower, upper = score["censored_score_bounds"]
                axes[0].vlines(
                    g, lower, upper, color=color, linewidth=2, linestyles="--"
                )
                continue
            interval = score["pair_bootstrap_95_percent"]
            if interval is not None:
                axes[0].vlines(g, interval[0], interval[1], color=color, linewidth=2)
    axes[0].axhline(0.5, color="gray", linestyle="--", linewidth=1)
    axes[0].set(
        ylim=(-0.03, 1.03),
        ylabel="Match score (win + ½ draw)",
        title=(
            "Points: completed-game scores; solid bars: paired bootstrap 95% intervals\n"
            "Dashed bars: all-game score bounds if unfinished games lose/win"
        ),
    )
    axes[0].legend()
    x = [r["generation"] for r in rows]
    axes[1].plot(x, [r["losses"][1] for r in rows], color="#355cbb")
    axes[1].set(
        ylabel="Policy cross-entropy",
        title="Fresh training targets change each generation",
    )
    axes[2].plot(x, [r["losses"][2] for r in rows], color="#ce741b")
    axes[2].set(ylabel="Value MSE", xlabel="Generation / Adam update")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(root / "learning-curve.png", dpi=160)
    fig.savefig(root / "learning-curve.svg")
    summary = {
        "generations": len(rows),
        "positions": sum(r["positions"] for r in rows),
        "games": sum(len(r["game_plies"]) for r in rows),
        "all_parameters_finite": all(r["finite_parameters"] for r in rows),
        "one_adam_update_per_generation": all(
            r["optimizer_step"] == r["generation"] for r in rows
        ),
        "self_play_seconds": sum(r["self_play_seconds"] for r in rows),
        "learning_seconds": sum(r["learning_seconds"] for r in rows),
        "self_play_draws": sum(
            result == 0 for r in rows for result in r["outcomes_white"]
        ),
        "draw_position_fraction": (
            sum(
                plies
                for r in rows
                for plies, result in zip(r["game_plies"], r["outcomes_white"])
                if result == 0
            )
            / sum(r["positions"] for r in rows)
        ),
        "first_ten_mean_losses": np.mean(
            [r["losses"] for r in rows[:10]], axis=0
        ).tolist(),
        "last_ten_mean_losses": np.mean(
            [r["losses"] for r in rows[-10:]], axis=0
        ).tolist(),
        "evaluations": evaluations,
        "oracle": verification,
    }
    (root / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
