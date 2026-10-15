"""Read-only activation and mirror probes; never fit evaluation positions."""

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, "/opt/chesszero/chesszero")
from run_training import evaluate, write_json
from ablate import mirror_states
from restart_training import color_summary

SOURCE = Path("/results/pr29-first-result-20261014-202746")
REPEAT = Path("/results/pr29-random-restart-20261015-085706")


def positions():
    import chess
    from chess_engine.bitboard import BitBoard

    states, targets = [], []
    for label in ("initial", "trained"):
        for row in json.loads((SOURCE / f"{label}-vs-random-games.json").read_text()):
            board = BitBoard()
            selected = set(np.linspace(4, len(row["moves"]) - 1, 8, dtype=int))
            white_result = row["outcome"] * (1 if row["color"] == 0 else -1)
            for ply, uci in enumerate(row["moves"]):
                if ply in selected:
                    states.append(board.state.copy())
                    targets.append(
                        white_result * (1 if board.get_current_turn() == 0 else -1)
                    )
                move = chess.Move.from_uci(uci)
                source, target = divmod(move.from_square, 8), divmod(move.to_square, 8)
                if move.promotion:
                    target += (move.promotion - 1,)
                assert board.make_move(source, target)
    original = np.stack(states)
    return np.concatenate((original, mirror_states(original))), np.tile(targets, 2)


def inspect(model, states, targets):
    import mlx.core as mx

    values, alive, hidden_norm, bn_input_means, bn_input_std = [], [], [], [], []
    for start in range(0, len(states), 64):
        batch = mx.array(states[start : start + 64])
        features = model.relu(
            model.bn_input(model.conv_input(mx.transpose(batch, (0, 2, 3, 1))))
        )
        features = model.residual_tower(features)
        conv = model.value_conv(features)
        value_features = model.relu(model.value_bn(conv))
        pre_hidden = model.value_fc1(value_features.reshape(-1, 32 * 8 * 8))
        hidden = model.relu(pre_hidden)
        predictions = mx.tanh(model.value_fc2(hidden))
        _, direct = model(batch)
        assert bool(mx.allclose(predictions, direct).item())
        values.extend(np.array(predictions).reshape(-1))
        alive.extend(np.array(mx.mean((pre_hidden > 0).astype(mx.float32), axis=1)))
        hidden_norm.extend(np.array(mx.sqrt(mx.sum(hidden * hidden, axis=1))))
        bn_input_means.extend(np.array(mx.mean(conv, axis=(1, 2, 3))))
        bn_input_std.extend(np.array(mx.std(conv, axis=(1, 2, 3))))
    values = np.array(values)
    n_original = len(states) // 2
    # Both members of every mirror pair have the same side-to-move target.
    result = dict(
        mirror_value_absolute_difference=float(
            np.abs(values[:n_original] - values[n_original:]).mean()
        ),
        value_fc2_bias=float(np.array(model.value_fc2.bias).item()),
    )
    white_material = np.sum(states[:, :5], axis=(2, 3)) @ np.array([1, 3, 3, 5, 9])
    black_material = np.sum(states[:, 6:11], axis=(2, 3)) @ np.array([1, 3, 3, 5, 9])
    material = (white_material - black_material) * (2 * states[:, 12, 0, 0] - 1)
    for color, name in ((1, "white"), (0, "black")):
        mask = states[:, 12, 0, 0] == color
        result[name] = dict(
            positions=int(mask.sum()),
            mean_value=float(values[mask].mean()),
            std_value=float(values[mask].std()),
            target_mean=float(targets[mask].mean()),
            outcome_mse=float(np.mean((values[mask] - targets[mask]) ** 2)),
            material_correlation=float(np.corrcoef(values[mask], material[mask])[0, 1]),
            positive_hidden_fraction=float(np.array(alive)[mask].mean()),
            hidden_norm=float(np.array(hidden_norm)[mask].mean()),
            value_conv_mean=float(np.array(bn_input_means)[mask].mean()),
            value_conv_std=float(np.array(bn_input_std)[mask].mean()),
        )
    return result


def main():
    import logging
    import mlx.core as mx
    from model.network import ChessNet

    logging.basicConfig(level=logging.INFO)
    mx.set_default_device(mx.gpu)
    out = Path(sys.argv[1])
    states, targets = positions()
    result = {}
    for root, epochs in ((SOURCE, (0, 1, 3, 5, 10)), (REPEAT, (0, 3))):
        for epoch in epochs:
            model, checkpoint = ChessNet.load_checkpoint(root / "checkpoints", epoch)
            del checkpoint
            model.eval()
            key = root.name + ":" + str(epoch)
            result[key] = inspect(model, states, targets)
            write_json(out / "activations.json", result)
            print("ACTIVATIONS", key, json.dumps(result[key]), flush=True)
            if root == REPEAT and epoch == 3:
                evaluate(model, model.config, out, "repeat-epoch3-vs-random", 16, 10000)
                rows = json.loads(
                    (out / "repeat-epoch3-vs-random-games.json").read_text()
                )
                result["repeat_epoch3_colors"] = color_summary(rows)
            del model
            mx.clear_cache()
    write_json(
        out / "result.json",
        dict(
            status="completed",
            optimizer_updates=0,
            normalization_updates=0,
            report=result,
        ),
    )


if __name__ == "__main__":
    main()
