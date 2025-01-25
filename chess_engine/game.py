from board import Board, Color, Piece, PieceType
from typing import Tuple, List, Optional


class ChessGame:
    def __init__(self):
        self.board = Board()
        self.current_turn = Color.WHITE
        self.move_history: List[str] = []
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
                    and piece.color == self.current_turn
                ):
                    # Check if the piece can move to the target square
                    if self._can_move(piece, (row, col), target_square):
                        similar_pieces.append((row, col))
        return similar_pieces

    def _can_move(
        self, piece: Piece, from_square: Tuple[int, int], to_square: Tuple[int, int]
    ) -> bool:
        """Check if a piece can move from from_square to to_square."""
        return self.board.is_valid_move(from_square, to_square)

    def get_all_valid_moves(self) -> List[str]:
        """Get all valid moves in algebraic notation."""
        valid_moves = []
        pieces = (
            self.board.white_pieces
            if self.current_turn == Color.WHITE
            else self.board.black_pieces
        )

        for piece, pos in pieces:
            moves = self.board.get_valid_moves(pos)
            for move in moves:
                move_str = self._move_to_algebraic(pos, move, piece)
                valid_moves.append(move_str)

        return valid_moves

    def make_move(self, move_str: str) -> bool:
        """Make a move using standard algebraic notation."""
        # Parse the move
        from_pos, to_pos = self._parse_move(move_str)
        if not from_pos or not to_pos:
            if self.DEBUG:
                print(f"\nDEBUG: Failed to parse move: {move_str}")
            return False

        # Verify it's the correct player's turn
        piece = self.board.squares[from_pos[0]][from_pos[1]]
        if not piece or piece.color != self.current_turn:
            if self.DEBUG:
                print(f"\nDEBUG: Wrong player's turn or no piece at {from_pos}")
            return False

        # Try to make the move
        if self._try_move(from_pos, to_pos):
            self.move_history.append(move_str)
            return True

        if self.DEBUG:
            print(f"\nDEBUG: Invalid move {move_str}")
            print(f"From: {from_pos}, To: {to_pos}")
            print(f"Valid moves: {self.board.get_valid_moves(from_pos)}")
        return False

    def _try_move(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
        """Attempt to make a move and validate it."""
        piece = self.board.squares[from_pos[0]][from_pos[1]]
        if not piece or piece.color != self.current_turn:
            return False

        # Check if move is valid
        valid_moves = self.board.get_valid_moves(from_pos)
        if to_pos not in valid_moves:
            return False

        # Make the move
        self.board.move_piece(from_pos, to_pos)
        self.current_turn = (
            Color.BLACK if self.current_turn == Color.WHITE else Color.WHITE
        )
        return True

    def _parse_move(
        self, move_str: str
    ) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """Parse algebraic notation into board coordinates."""
        move = move_str.strip()

        # Handle castling
        if move in ["O-O", "0-0"]:  # Kingside castling
            row = 0 if self.current_turn == Color.WHITE else 7
            return (row, 4), (row, 6)
        elif move in ["O-O-O", "0-0-0"]:  # Queenside castling
            row = 0 if self.current_turn == Color.WHITE else 7
            return (row, 4), (row, 2)

        # Remove capture and check symbols
        move = move.replace("x", "").rstrip("+#")

        # Get destination square
        dest_square = move[-2:]
        to_col = ord(dest_square[0]) - ord("a")
        to_row = int(dest_square[1]) - 1
        if not (0 <= to_row < 8 and 0 <= to_col < 8):
            return None, None
        to_pos = (to_row, to_col)

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
        pieces = self.board.get_pieces_by_type(self.current_turn, piece_type)

        # Find piece that can make this move
        for piece, from_pos in pieces:
            if to_pos in self.board.get_valid_moves(from_pos):
                # For pawns with source file, check column matches
                if source_file is not None and from_pos[1] != source_file:
                    continue
                return from_pos, to_pos

        return None, None

    def get_current_turn(self) -> Color:
        return self.current_turn

    def get_game_state(self) -> str:
        """Get the current state of the game."""
        if self.board.is_checkmate(self.current_turn):
            winner = "Black" if self.current_turn == Color.WHITE else "White"
            return f"Checkmate! {winner} wins!"
        elif self.board.is_in_check(self.current_turn):
            player = "White" if self.current_turn == Color.WHITE else "Black"
            return f"Check! {player} is in check!"
        elif self.board.is_draw():
            return "Draw!"
        elif self.board.is_stalemate(self.current_turn):
            return "Stalemate! Game is a draw!"
        return "Normal"

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
        capture = "x" if target else ""

        # Add destination square
        destination = files[to_pos[1]] + ranks[to_pos[0]]

        return f"{piece_symbol}{capture}{destination}"

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
