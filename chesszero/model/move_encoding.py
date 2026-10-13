"""Shared policy indices for search priors and self-play targets.

Ordinary moves and queen promotions retain the existing 64-by-64 from/to
encoding. Knight, bishop and rook promotions use previously unused outputs
4096..4479, within the existing 4672-output network. This is a custom encoding,
not AlphaZero's 73-plane encoding; checkpoint tensor shapes remain unchanged.
"""

from typing import Tuple, Union


MoveDestination = Union[Tuple[int, int], Tuple[int, int, int]]


def encode_move(from_pos: Tuple[int, int], to_pos: MoveDestination) -> int:
    from_row, from_col = from_pos
    to_row, to_col = to_pos[:2]
    ordinary_index = (from_row * 8 + from_col) * 64 + to_row * 8 + to_col
    if len(to_pos) == 2:
        return ordinary_index

    promotion_piece = to_pos[2]
    if (
        (from_row, to_row) not in ((6, 7), (1, 0))
        or abs(from_col - to_col) > 1
        or promotion_piece not in (1, 2, 3, 4)
    ):
        raise ValueError(f"Invalid promotion move: {from_pos} -> {to_pos}")
    if promotion_piece == 4:
        return ordinary_index

    # Reserve 8 source files * 8 destination files * 3 pieces for each color.
    color = 0 if from_row == 6 else 1
    return 4096 + color * 192 + from_col * 24 + to_col * 3 + promotion_piece - 1
