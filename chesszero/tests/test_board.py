"""Rule regressions against python-chess (a test-only dependency)."""

import random
import unittest

import chess
import numpy as np

from chess_engine.bitboard import BitBoard


def from_fen(fen):
    reference = chess.Board(fen)
    board = BitBoard()
    board.state.fill(0)
    for square, piece in reference.piece_map().items():
        color = 0 if piece.color else 1
        row, col = divmod(square, 8)
        board.state[piece.piece_type - 1 + 6 * color, row, col] = 1
        if piece.piece_type == chess.KING:
            board.king_positions[color] = (row, col)
    board.state[12].fill(reference.turn)
    for color in (0, 1):
        board.state[13 + 2 * color].fill(
            reference.has_kingside_castling_rights(not color)
        )
        board.state[14 + 2 * color].fill(
            reference.has_queenside_castling_rights(not color)
        )
    board.state[17].fill(
        (2 * (reference.fullmove_number - 1) + (not reference.turn)) / 200
    )
    board.state[18].fill(reference.halfmove_clock / 100)
    board.en_passant_target = (
        divmod(reference.ep_square, 8) if reference.ep_square is not None else None
    )
    board.position_history = {board.get_repetition_hash(): 1}
    return board, reference


def coords(move):
    target = divmod(move.to_square, 8)
    if move.promotion:
        target += (move.promotion - 1,)
    return divmod(move.from_square, 8), target


def legal_moves(board):
    return {
        (source, target)
        for source, _ in board.get_all_pieces(board.get_current_turn())
        for target in board.get_valid_moves(source)
    }


class TestBoard(unittest.TestCase):
    def test_legal_moves_and_state_match_reference_during_seeded_games(self):
        rng = random.Random(2026)
        for _ in range(3):
            board, reference = from_fen(chess.STARTING_FEN)
            for ply in range(80):
                moves = list(reference.legal_moves)
                self.assertEqual(
                    legal_moves(board), {coords(m) for m in moves}, reference.fen()
                )
                self.assertEqual(
                    board.is_in_check(board.get_current_turn()), reference.is_check()
                )
                if not moves:
                    break
                move = rng.choice(moves)
                self.assertTrue(board.make_move(*coords(move)))
                reference.push(move)
                expected, _ = from_fen(reference.fen(en_passant="fen"))
                np.testing.assert_array_equal(board.state, expected.state)
                self.assertEqual(board.en_passant_target, expected.en_passant_target)

    def test_castling_both_sides_and_colors_moves_rook_and_clears_rights(self):
        for color in ("w", "b"):
            for uci in ("e1g1", "e1c1") if color == "w" else ("e8g8", "e8c8"):
                board, ref = from_fen(f"r3k2r/8/8/8/8/8/8/R3K2R {color} KQkq - 0 1")
                move = chess.Move.from_uci(uci)
                self.assertIn(coords(move), legal_moves(board))
                self.assertTrue(board.make_move(*coords(move)))
                ref.push(move)
                expected, _ = from_fen(ref.fen())
                np.testing.assert_array_equal(board.state, expected.state)

    def test_special_positions_match_reference(self):
        for fen in (
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "4kr1r/8/8/8/8/8/8/R3K2R w KQ - 0 1",  # through check
            "4k3/8/8/8/8/8/8/R3K2R w Q - 0 1",  # distinct rights
            "4k3/8/8/r4pPK/8/8/8/8 w - f6 0 1",  # EP exposes rook
            "4k3/8/8/3pP3/4K3/8/8/8 w - d6 0 1",  # EP evades check
            "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1",  # black EP
            "4k3/P7/8/8/8/8/7p/4K3 w - - 0 1",  # promotions
        ):
            board, ref = from_fen(fen)
            self.assertEqual(
                legal_moves(board), {coords(m) for m in ref.legal_moves}, fen
            )
            for move in list(ref.legal_moves):
                child = board.copy()
                self.assertTrue(child.make_move(*coords(move)))
                ref.push(move)
                expected, _ = from_fen(ref.fen(en_passant="fen"))
                np.testing.assert_array_equal(child.state, expected.state)
                ref.pop()

    def test_en_passant_expires_and_changes_repetition_only_when_legal(self):
        for fen, differs in (
            ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", True),
            ("4k3/8/8/3p4/8/8/8/4K3 w - d6 0 1", False),
            ("4k3/8/8/r4pPK/8/8/8/8 w - f6 0 1", False),
        ):
            board, _ = from_fen(fen)
            clone = board.copy()
            clone.en_passant_target = None
            self.assertEqual(
                board.get_repetition_hash() != clone.get_repetition_hash(), differs
            )
            self.assertNotEqual(board.get_hash(), clone.get_hash())
        board, _ = from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
        self.assertTrue(board.make_move((0, 4), (0, 3)))
        self.assertIsNone(board.en_passant_target)

    def test_rejected_moves_do_not_mutate_state(self):
        board = BitBoard()
        before = board.copy()
        for move in (
            ((1, 4), (4, 4)),
            ((0, 4), (0, 6)),
            ((1, 0), (2, 0, 5)),
            ((6, 0), (5, 0)),
            ((-1, 0), (0, 0)),
        ):
            self.assertFalse(board.make_move(*move))
            np.testing.assert_array_equal(board.state, before.state)
            self.assertEqual(board.get_hash(), before.get_hash())

    def test_fifty_move_boundary_and_resets(self):
        board, _ = from_fen("4k3/8/8/8/8/8/P7/R3K3 w - - 99 151")
        self.assertFalse(board.is_draw())
        self.assertEqual(board.get_move_count(), 300)
        pawn = board.copy()
        self.assertTrue(pawn.make_move((1, 0), (2, 0)))
        self.assertEqual(pawn.get_moves_without_progress(), 0)
        self.assertTrue(board.make_move((0, 0), (0, 1)))
        self.assertEqual(board.get_moves_without_progress(), 100)
        self.assertEqual(board.get_move_count(), 301)
        self.assertTrue(board.is_draw())

    def test_claimable_and_automatic_draw_thresholds(self):
        board, _ = from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 99 1")
        self.assertTrue(board.can_claim_draw())  # May name a quiet next move.
        self.assertFalse(board.is_draw())
        board.state[18].fill(1.0)
        self.assertTrue(board.is_fifty_moves())
        self.assertTrue(board.is_draw())
        self.assertFalse(board.is_draw(claim_draw=False))
        board.state[18].fill(1.5)
        self.assertTrue(board.is_seventyfive_moves())
        self.assertTrue(board.is_draw(claim_draw=False))
        board.state[18].fill(0)
        board.position_history[board.get_repetition_hash()] = 3
        self.assertTrue(board.is_draw())
        self.assertFalse(board.is_draw(claim_draw=False))
        board.position_history[board.get_repetition_hash()] = 5
        self.assertTrue(board.is_draw(claim_draw=False))

    def test_terminal_results_and_material_match_reference(self):
        for fen in (
            "7k/6Q1/5K2/8/8/8/8/8 b - - 100 90",  # mate beats 50 moves
            "k7/2Q5/6K1/8/8/8/8/8 b - - 0 1",  # stalemate
            "4k3/8/8/8/8/8/8/1N2K1N1 w - - 0 1",  # two knights can mate
            "4k1n1/8/8/8/8/8/8/1N2K3 w - - 0 1",
            "4kb2/8/8/8/8/8/8/2B1K3 w - - 0 1",
            "4k3/8/8/8/8/8/8/2B1K3 w - - 0 1",
        ):
            board, ref = from_fen(fen)
            self.assertEqual(
                board._has_insufficient_material(), ref.is_insufficient_material()
            )
            outcome = ref.outcome()
            for color in (0, 1):
                expected = (
                    None
                    if outcome is None
                    else (
                        0.0
                        if outcome.winner is None
                        else (1.0 if outcome.winner == (color == 0) else -1.0)
                    )
                )
                self.assertEqual(board.get_game_result(color), expected, fen)

    def test_rook_capture_revokes_the_correct_castling_right(self):
        board, ref = from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        move = chess.Move.from_uci("a1a8")
        self.assertTrue(board.make_move(*coords(move)))
        ref.push(move)
        expected, _ = from_fen(ref.fen())
        np.testing.assert_array_equal(board.state, expected.state)
        self.assertEqual(board.get_moves_without_progress(), 0)

    def test_kiwipete_perft_depth_two(self):
        board, _ = from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
        )
        moves = legal_moves(board)
        self.assertEqual(len(moves), 48)
        total = 0
        for move in moves:
            child = board.copy()
            self.assertTrue(child.make_move(*move))
            total += len(legal_moves(child))
        self.assertEqual(total, 2039)

    def test_initial_perft_depth_two(self):
        board = BitBoard()
        total = 0
        for move in legal_moves(board):
            child = board.copy()
            child.make_move(*move)
            total += len(legal_moves(child))
        self.assertEqual(total, 400)
