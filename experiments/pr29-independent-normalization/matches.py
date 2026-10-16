"""Paired candidate-versus-frozen-baseline matches and untouched confirmation banks."""

import math
import time
import numpy as np
from model.mcts import MCTS
from model.pipe_inference import run_pipe_games


def play_match(client, config, pair_id, color, opening, connection, workers):
    import chess
    from chess_engine.bitboard import BitBoard

    board, oracle, moves = BitBoard(), chess.Board(), []
    searches = {
        color: MCTS(client, config, exploration_noise=False),
        1 - color: MCTS(client.for_model(1), config, exploration_noise=False),
    }
    started = time.monotonic()

    def apply(move):
        source, target = move
        ref = chess.Move(
            chess.square(source[1], source[0]),
            chess.square(target[1], target[0]),
            promotion=target[2] + 1 if len(target) == 3 else None,
        )
        if ref not in oracle.legal_moves or not board.make_move(*move):
            raise RuntimeError(
                f"Illegal baseline match move {ref.uci()}: {oracle.fen()}"
            )
        oracle.push(ref)
        moves.append(ref.uci())
        for search in searches.values():
            search.advance_root(move, board)

    try:
        for u in opening.split():
            apply(((int(u[1]) - 1, ord(u[0]) - 97), (int(u[3]) - 1, ord(u[2]) - 97)))
        while (
            not board.is_game_over()
            and len(moves) < 1024
            and time.monotonic() - started < 900
        ):
            move = searches[board.get_current_turn()].get_move(board, temperature=0)
            if move is None:
                raise RuntimeError("Missing move in an ongoing match")
            apply(move)
        outcome = board.get_game_result(color)
        if outcome is not None:
            ref = oracle.outcome(claim_draw=True)
            expected = (
                None
                if ref is None
                else 0
                if ref.winner is None
                else 1
                if ref.winner == (color == 0)
                else -1
            )
            if outcome != expected:
                raise RuntimeError("Baseline match outcome disagrees with oracle")
        connection.send(
            dict(
                pair_id=pair_id,
                color=color,
                outcome=outcome,
                score=None if outcome is None else (outcome + 1) / 2,
                moves=moves,
                opening=opening,
                seconds=time.monotonic() - started,
                oracle_verified=True,
            )
        )
    finally:
        client.close()


def summarize_matches(rows, alpha=0.05, cluster_families=False):
    if not rows:
        raise ValueError("No matches to score")
    scores = [r["score"] for r in rows]
    wins, draws, losses = (sum(s == v for s in scores) for v in (1, 0.5, 0))
    censored = sum(s is None for s in scores)
    decisive = wins + losses
    p = (
        sum(math.comb(decisive, k) for k in range(wins, decisive + 1)) / 2**decisive
        if decisive
        else 1.0
    )
    units = {}
    for row in rows:
        key = row["pair_id"] % 16 if cluster_families else row["pair_id"]
        units.setdefault(key, []).append(row["score"] or 0)
    means = np.array([np.mean(v) for v in units.values()])
    samples = (
        np.random.default_rng(901)
        .choice(means, (20000, len(means)), replace=True)
        .mean(1)
    )
    interval = np.quantile(samples, [alpha / 2, 1 - alpha / 2]).tolist()
    return dict(
        games=len(rows),
        wins=wins,
        draws=draws,
        losses=losses,
        censored=censored,
        score=sum(s or 0 for s in scores) / len(scores),
        one_sided_p=p,
        bootstrap_interval=interval,
        alpha=alpha,
        baseline_beaten=not censored and p < alpha and interval[0] > 0.5,
        by_color={
            str(c): dict(
                wins=sum(r["outcome"] == 1 for r in rows if r["color"] == c),
                draws=sum(r["outcome"] == 0 for r in rows if r["color"] == c),
                losses=sum(r["outcome"] == -1 for r in rows if r["color"] == c),
            )
            for c in (0, 1)
        },
    )


def confirmation_openings(base_openings, seed, pairs=64):
    """Native legal continuations, held apart from all discovery start positions."""
    import chess
    from chess_engine.bitboard import BitBoard

    rng = np.random.default_rng(seed)
    result = []
    seen = set()
    while len(result) < pairs:
        pair_id = len(result)
        base = base_openings[pair_id % len(base_openings)]
        native, oracle = BitBoard(), chess.Board()
        sequence = base.split()
        for u in sequence:
            assert native.make_move(
                (int(u[1]) - 1, ord(u[0]) - 97), (int(u[3]) - 1, ord(u[2]) - 97)
            )
            oracle.push_uci(u)
        for _ in range(4):
            if native.is_game_over():
                break
            legal = sorted(
                (s, d)
                for s, _ in native.get_all_pieces(native.get_current_turn())
                for d in native.get_valid_moves(s)
            )
            source, target = legal[int(rng.integers(len(legal)))]
            move = chess.Move(
                chess.square(source[1], source[0]),
                chess.square(target[1], target[0]),
                promotion=target[2] + 1 if len(target) == 3 else None,
            )
            assert move in oracle.legal_moves and native.make_move(source, target)
            oracle.push(move)
            sequence.append(move.uci())
        identity = " ".join(oracle.fen().split()[:4])
        if len(sequence) == 8 and not native.is_game_over() and identity not in seen:
            result.append(" ".join(sequence))
            seen.add(identity)
    return result


def evaluate(
    candidate,
    baseline,
    config,
    openings,
    write_json,
    out,
    name,
    workers=32,
    alpha=0.05,
    confirmation=False,
):
    print("MATCH_START", name, 2 * len(openings), flush=True)
    rows, failures, statistics = run_pipe_games(
        [candidate, baseline],
        config,
        play_match,
        [(config, p, c, opening) for p, opening in enumerate(openings) for c in (0, 1)],
        workers,
    )
    write_json(out / (name + "-games.json"), rows)
    if failures or len(rows) != 2 * len(openings):
        raise RuntimeError(f"Incomplete match bank {name}")
    result = summarize_matches(rows, alpha, cluster_families=confirmation)
    result["inference"] = statistics
    write_json(out / (name + ".json"), result)
    print("MATCH_COMPLETE", name, result, flush=True)
    return result
