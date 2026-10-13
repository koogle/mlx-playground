"""Checks that evaluation measures outcomes without turning censored games into draws."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "chesszero"))
import convergence


class EvaluationTests(unittest.TestCase):
    def test_terminal_result_is_from_candidate_color(self):
        with patch.object(convergence, "OPENINGS", ["f2f3 e7e5 g2g4 d8h4"]):
            self.assertEqual(convergence.evaluation_game((0, 0, "initial"))["score"], 0)
            self.assertEqual(convergence.evaluation_game((0, 1, "initial"))["score"], 1)

    def test_incomplete_game_is_not_a_draw_or_a_completed_pair(self):
        rows = [{"opening_id": 0, "score": 1}, {"opening_id": 0, "score": None}]
        result = convergence.summarize(rows)
        self.assertEqual(result["censored"], 1)
        self.assertEqual(result["draws"], 0)
        self.assertEqual(result["complete_pairs"], 0)
        self.assertEqual(result["censored_score_bounds"], [0.5, 1.0])

    def test_color_swapped_pairs_have_expected_score(self):
        rows = [{"opening_id": i, "score": score} for i in range(4) for score in (0, 1)]
        result = convergence.summarize(rows)
        self.assertEqual(result["score"], 0.5)
        self.assertEqual(result["pair_bootstrap_95_percent"], [0.5, 0.5])

    def test_every_opening_is_legal(self):
        import chess  # Independent test oracle, never the production engine.
        from chess_engine.bitboard import BitBoard

        for opening in convergence.OPENINGS:
            board, oracle = BitBoard(), chess.Board()
            for uci in opening.split():
                move = chess.Move.from_uci(uci)
                self.assertIn(move, oracle.legal_moves)
                oracle.push(move)
                self.assertTrue(
                    board.make_move(
                        (int(uci[1]) - 1, ord(uci[0]) - 97),
                        (int(uci[3]) - 1, ord(uci[2]) - 97),
                    )
                )


if __name__ == "__main__":
    unittest.main()
