from .board import Board, Color, Piece, PieceType
from typing import Tuple, List, Optional, Set


class ChessGame:
    def __init__(self):
        self.board = Board()
        self.move_history: List[str] = []
        self.moves_without_progress = 0  # Counter for 50/75 move rule
        self.DEBUG = False

    def _get_piece_symbol(self, piece: Piece) -> str:
        """Get the algebraic notation symbol for a piece."""
        if piece.piece_type == PieceType.PAWN:
            return ""
        piece_symbols = {
            PieceType.KING: "K",
            PieceType.QUEEN: "Q",
            PieceType.ROOK: "R",
            PieceType.BISHOP: "B",
            PieceType.KNIGHT: "N",
        }
        return piece_symbols[piece.piece_type]

    def _find_similar_pieces(
        self, piece_type: PieceType, target_square: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Find all pieces of the same type that could move to the target square."""
        similar_pieces = []
        for row in range(8):
            for col in range(8):
                piece = self.board.squares[row][col]
                if (
                    piece
                    and piece.piece_type == piece_type
                    and piece.color == self.board.current_turn
                ):
                    # Check if the piece can move to the target square
                    if self._can_move(piece, (row, col), target_square):
                        similar_pieces.append((row, col))
        return similar_pieces

    def _can_move(
        self, piece: Piece, from_square: Tuple[int, int], to_square: Tuple[int, int]
    ) -> bool:
        """Check if a piece can move from from_square to to_square."""
        attack_info = self.board.get_attack_info(piece.color)
        valid_moves = self.board.get_valid_moves(from_square, attack_info)
        return to_square in valid_moves

    def get_all_valid_moves(self) -> List[str]:
        """Get all valid moves in algebraic notation."""
        valid_moves = []
        pieces = (
            self.board.white_pieces
            if self.board.current_turn == Color.WHITE
            else self.board.black_pieces
        )

        # Get attack info once for all pieces
        attack_info = self.board.get_attack_info(self.board.current_turn)

        for piece, pos in pieces:
            moves = self.board.get_valid_moves(pos, attack_info)
            for move in moves:
                move_str = self._move_to_algebraic(pos, move, piece)
                valid_moves.append(move_str)

        return valid_moves

    def make_move(self, move: str) -> bool:
        """Make a move in algebraic notation (e.g. 'e2e4' or 'g1f3')"""
        from_pos, to_pos = self.parse_move(move)
        if not from_pos or not to_pos:
            return False

        # Track moves without captures or pawn moves for 50/75 move rule
        piece = self.board.squares[from_pos[0]][from_pos[1]]
        target = self.board.squares[to_pos[0]][to_pos[1]]
        if piece.piece_type == PieceType.PAWN or target:
            self.moves_without_progress = 0
        else:
            self.moves_without_progress += 1

        # Make the move
        if self.board.move_piece(from_pos, to_pos):
            self.move_history.append(move)
            return True
        return False

    def make_move_coords(
        self, from_pos: Tuple[int, int], to_pos: Tuple[int, int], move_str: str
    ) -> bool:
        """Make a move using board coordinates"""
        # Track moves without captures or pawn moves for 50/75 move rule
        piece = self.board.squares[from_pos[0]][from_pos[1]]
        target = self.board.squares[to_pos[0]][to_pos[1]]
        if piece.piece_type == PieceType.PAWN or target:
            self.moves_without_progress = 0
        else:
            self.moves_without_progress += 1

        # Make the move
        if self.board.move_piece(from_pos, to_pos):
            self.move_history.append(move_str)
            return True
        return False

    def get_valid_moves(self, pos: Tuple[int, int]) -> Set[Tuple[int, int]]:
        """Get valid moves for a piece"""
        return self.board.get_valid_moves(pos)

    def get_game_state(self) -> str:
        """Get the current game state as a string"""
        if self.board.is_checkmate(self.board.current_turn):
            winner = "Black" if self.board.current_turn == Color.WHITE else "White"
            return f"Checkmate - {winner} wins"
        elif self.board.is_draw():
            return "Draw"
        elif self.moves_without_progress >= 75:
            return "Draw (75-move rule)"
        elif len(self.move_history) >= 200:
            return "Draw (maximum moves)"
        elif self.board.is_in_check(self.board.current_turn):
            return "Check"
        return "Ongoing"

    def get_result(self) -> float:
        """Get the game result from current player's perspective"""
        if self.board.is_game_over():
            return self.board.get_game_result()
        elif self.moves_without_progress >= 75 or len(self.move_history) >= 200:
            return 0.0  # Draw by move limit or 75-move rule
        return 0.0  # Game not over

    def can_claim_draw(self) -> bool:
        """Check if a player can claim a draw (50-move rule)"""
        return self.moves_without_progress >= 50

    def __str__(self):
        return str(self.board)

    def _move_to_algebraic(
        self, from_pos: Tuple[int, int], to_pos: Tuple[int, int], piece: Piece
    ) -> str:
        """Convert a move to algebraic notation."""
        files = "abcdefgh"
        ranks = "12345678"

        # Special case for castling
        if piece.piece_type == PieceType.KING and abs(to_pos[1] - from_pos[1]) == 2:
            return "O-O" if to_pos[1] > from_pos[1] else "O-O-O"

        piece_symbol = ""
        if piece.piece_type != PieceType.PAWN:
            piece_symbol = piece.piece_type.name[0]
            if piece.piece_type == PieceType.KNIGHT:
                piece_symbol = "N"

        # Add capture symbol if needed
        target = self.board.squares[to_pos[0]][to_pos[1]]
        capture = ""
        if target:
            # For pawns, include the file of origin
            if piece.piece_type == PieceType.PAWN:
                capture = files[from_pos[1]] + "x"
            else:
                capture = "x"

        # Add destination square
        destination = files[to_pos[1]] + ranks[to_pos[0]]

        # Add promotion indicator
        promotion = ""
        if piece.piece_type == PieceType.PAWN:
            if (piece.color == Color.WHITE and to_pos[0] == 7) or (
                piece.color == Color.BLACK and to_pos[0] == 0
            ):
                promotion = "=Q"  # Default to queen promotion

        return f"{piece_symbol}{capture}{destination}{promotion}"

    def load_game_history(self, history_str: str) -> bool:
        """Load and replay a game from a history string."""
        # Reset the game
        self.__init__()

        try:
            # Parse moves from history string
            moves = []
            for line in history_str.strip().split("\n"):
                if not line or line.startswith("Move history:"):
                    continue
                # Extract move number and moves
                parts = line.split(".")
                if len(parts) != 2:
                    continue
                # Split moves, handling potential single move at end
                move_parts = parts[1].strip().split()
                moves.extend(move_parts)

            # Replay moves
            print("\nReplaying game history:")
            for i, move in enumerate(moves):
                if not self.make_move(move):
                    print(f"Failed to replay move: {move}")
                    return False

                # Print board state after each move
                print(f"\nMove {i+1}: {move}")
                print(self.board)

            return True

        except Exception as e:
            print(f"Error loading game history: {e}")
            self.__init__()  # Reset on error
            return False

    def get_game_history_str(self) -> str:
        """Get the game history in a readable format."""
        if not self.move_history:
            return "No moves played"

        history = []
        for i in range(0, len(self.move_history), 2):
            move_num = i // 2 + 1
            if i + 1 < len(self.move_history):
                history.append(
                    f"{move_num}. {self.move_history[i]} {self.move_history[i+1]}"
                )
            else:
                history.append(f"{move_num}. {self.move_history[i]}")

        return "\n".join(history)

    def is_over(self) -> bool:
        """Check if the game is over (checkmate, stalemate, or draw)"""
        return (
            self.board.is_checkmate(self.board.current_turn)
            or self.board.is_stalemate(self.board.current_turn)
            or len(self.move_history) >= 200  # Maximum game length
            or self.moves_without_progress >= 75  # 75-move rule
            or self.board.is_draw()  # Other draw conditions
        )

    def is_draw(self) -> bool:
        """Check if the game is a draw"""
        return (
            self.board.is_draw()  # Original draw conditions (stalemate, insufficient material)
            or self.moves_without_progress >= 75  # 75-move rule
            or len(self.move_history) >= 200  # Maximum game length
        )

    def get_current_turn(self) -> Color:
        return self.board.current_turn

    def parse_move(
        self, move_str: str
    ) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """Parse algebraic notation into board coordinates."""
        if not move_str:
            return None, None

        move = move_str.strip()

        # Handle castling
        if move in ["O-O", "0-0"]:  # Kingside castling
            row = 0 if self.board.current_turn == Color.WHITE else 7
            return (row, 4), (row, 6)
        elif move in ["O-O-O", "0-0-0"]:  # Queenside castling
            row = 0 if self.board.current_turn == Color.WHITE else 7
            return (row, 4), (row, 2)

        # Handle pawn captures with promotion (e.g. "cxb1=Q")
        if "=" in move and "x" in move:
            try:
                # Extract source file, target square and promotion piece
                source_file = ord(move[0]) - ord("a")
                target_file = ord(move[2]) - ord("a")
                target_rank = int(move[3]) - 1

                if (
                    0 <= source_file < 8
                    and 0 <= target_file < 8
                    and 0 <= target_rank < 8
                ):
                    # Find pawn that can make this capture
                    pawns = self.board.get_pieces_by_type(
                        self.board.current_turn, PieceType.PAWN
                    )
                    for piece, from_pos in pawns:
                        if from_pos[1] == source_file and (
                            target_rank,
                            target_file,
                        ) in self.board.get_valid_moves(from_pos):
                            return from_pos, (target_rank, target_file)
            except (ValueError, IndexError):
                pass

        # Handle standard moves (e.g., "e2e4" or "e7e8")
        if len(move) >= 4:
            try:
                from_file = ord(move[0]) - ord("a")
                from_rank = int(move[1]) - 1
                to_file = ord(move[2]) - ord("a")
                to_rank = int(move[3]) - 1

                # Validate coordinates
                if (
                    0 <= from_file < 8
                    and 0 <= from_rank < 8
                    and 0 <= to_file < 8
                    and 0 <= to_rank < 8
                ):
                    # Handle promotion if present (e.g., "e7e8Q" or "f1=Q")
                    if len(move) >= 5:
                        promotion_piece = move[4] if move[4] != "=" else move[5]
                        if promotion_piece in ["Q", "R", "B", "N"]:
                            return (from_rank, from_file), (to_rank, to_file)
                    elif len(move) == 4:
                        return (from_rank, from_file), (to_rank, to_file)
            except (ValueError, IndexError):
                pass

        # Remove capture and check symbols
        move = move.replace("x", "").rstrip("+#")

        # Handle pawn promotion notation like "f1=Q"
        if "=" in move:
            try:
                parts = move.split("=")
                if (
                    len(parts) == 2
                    and len(parts[0]) == 2
                    and parts[1] in ["Q", "R", "B", "N"]
                ):
                    to_file = ord(parts[0][0]) - ord("a")
                    to_rank = int(parts[0][1]) - 1
                    if 0 <= to_file < 8 and 0 <= to_rank < 8:
                        # Find pawn that can move to this square
                        to_pos = (to_rank, to_file)
                        pawns = self.board.get_pieces_by_type(
                            self.board.current_turn, PieceType.PAWN
                        )
                        for piece, from_pos in pawns:
                            if to_pos in self.board.get_valid_moves(from_pos):
                                return from_pos, to_pos
            except (ValueError, IndexError):
                pass

        # Get destination square
        if len(move) < 2:
            return None, None

        dest_square = move[-2:]
        try:
            to_col = ord(dest_square[0]) - ord("a")
            to_row = int(dest_square[1]) - 1
            if not (0 <= to_row < 8 and 0 <= to_col < 8):
                return None, None
            to_pos = (to_row, to_col)
        except (ValueError, IndexError):
            return None, None

        # Determine piece type
        piece_type = {
            "K": PieceType.KING,
            "Q": PieceType.QUEEN,
            "R": PieceType.ROOK,
            "B": PieceType.BISHOP,
            "N": PieceType.KNIGHT,
        }.get(move[0], PieceType.PAWN)

        # For pawn captures, use the source file
        source_file = None
        if piece_type == PieceType.PAWN and len(move) == 3 and move[0] in "abcdefgh":
            source_file = ord(move[0]) - ord("a")

        # Get all pieces of this type
        pieces = self.board.get_pieces_by_type(self.board.current_turn, piece_type)

        # Find piece that can make this move
        for piece, from_pos in pieces:
            if to_pos in self.board.get_valid_moves(from_pos):
                # For pawns with source file, check column matches
                if source_file is not None and from_pos[1] != source_file:
                    continue
                return from_pos, to_pos

        return None, None
