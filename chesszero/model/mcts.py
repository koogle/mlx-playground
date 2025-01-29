import math
import numpy as np
from tqdm import tqdm
from chess_engine.board import Board, Color
from config.model_config import ModelConfig
from utils.board_utils import encode_board, decode_policy
from typing import List, Tuple


class Node:
    """Tree node storing state, prior probabilities, visit counts, and Q-values"""

    def __init__(self, board: Board, parent=None, prior=0.0):
        self.board = board
        self.parent = parent
        self.prior = prior
        self.children = {}  # (from_pos, to_pos) -> Node
        self.visit_count = 0
        self.value_sum = 0
        self.is_expanded = False

    def value(self):
        """Get the average value of this node"""
        if self.visit_count == 0:
            return 0.0  # Return neutral value if unvisited
        return self.value_sum / self.visit_count

    def select_child(self, c_puct):
        """Select child with highest UCB score"""
        if not self.children:  # No children available
            print("no children")
            return None, None

        best_score = float("-inf")
        best_move = None
        best_child = None

        total_visits = max(1, self.visit_count)  # Ensure at least 1 visit

        for move, child in self.children.items():
            # Debug the PUCT calculation
            q_value = -child.value() if child.visit_count > 0 else 0
            u_value = (
                c_puct * child.prior * math.sqrt(total_visits) / (1 + child.visit_count)
            )
            score = q_value + u_value

            print(
                f"Move {move}: Q={q_value:.3f}, U={u_value:.3f}, Score={score:.3f}, "
                f"Prior={child.prior:.3f}, Visits={child.visit_count}"
            )

            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        if best_move is None:  # This should never happen if we have children
            print("WARNING: No best move found despite having children!")
            return None, None

        print(f"Selected move {best_move} with score {best_score}")
        return best_move, best_child


class MCTS:
    """
    MCTS implementation with:
    - Selection using PUCT algorithm
    - Expansion using network predictions
    - Backup using value estimates
    - Tree reuse between moves
    """

    def __init__(self, model, config: ModelConfig):
        self.model = model
        self.config = config

    def get_move(self, board: Board):
        """Get the best move for the current position after running MCTS"""
        # Set model to eval mode during search
        self.model.eval()
        try:
            root = Node(board)

            # First expand root node
            value = self.expand_and_evaluate(root)
            if not root.children:  # No valid moves at root
                return None

            selected_move = None
            # Run simulations
            # pbar = tqdm(range(self.config.n_simulations), desc="Running simulations")
            for idx in range(self.config.n_simulations):
                try:
                    # Early stopping checks
                    if idx > 20:  # Only check after some initial simulations
                        best_child = None
                        total_visits = sum(
                            child.visit_count for child in root.children.values()
                        )

                        # Skip early stopping check if no visits yet
                        if total_visits == 0:
                            continue
                        # Check if one move is dominating
                        for move, child in root.children.items():
                            visit_ratio = child.visit_count / total_visits
                            if visit_ratio > 0.9:  # 90% of visits
                                best_child = child
                                if child.visit_count > 50:  # Ensure enough exploration
                                    selected_move = move
                                    # pbar.close()
                                    break

                        # Check if best line is clearly better
                        if (
                            best_child and best_child.value() > 0.9
                        ):  # Clear winning position
                            selected_move = move
                            # pbar.close()
                            break

                    # Selection
                    node = root
                    search_path = [node]

                    while node.is_expanded:
                        move, child = node.select_child(self.config.c_puct)
                        print("move", move, child)

                        if not move:  # No valid moves
                            break
                        node = child
                        search_path.append(node)

                    # Expansion and evaluation
                    if not node.is_expanded:
                        try:
                            value = self.expand_and_evaluate(node)
                        except Exception as e:
                            print(f"Error during expansion: {e}")
                            # Use parent's value as fallback
                            value = -node.parent.value() if node.parent else 0

                        if not node.children:  # No valid moves
                            value = -1  # Loss position

                    # Backup
                    self.backup(search_path, value)

                    # Check for single legal move
                    if len(root.children) == 1:
                        selected_move = next(iter(root.children.keys()))
                        # pbar.close()
                        break

                except Exception as e:
                    import traceback

                    print(f"Error during simulation {idx}: {e}")
                    print(traceback.format_exc())
                    continue  # Skip this simulation if there's an error

            # If we didn't select a move through early stopping, use visit counts
            if not selected_move and root.children:
                # Find the most visited move
                max_visits = -1
                for move, child in root.children.items():
                    if child.visit_count > max_visits:
                        max_visits = child.visit_count
                        selected_move = move

            if not selected_move:
                # Fallback to random legal move if everything else fails
                moves = list(root.children.keys())
                if moves:
                    selected_move = moves[0]
                else:
                    return None

            # Print move info
            best_child = root.children[selected_move]
            total_visits = sum(child.visit_count for child in root.children.values())

            # print(f"\nPosition evaluation: {float(root.value()):.3f}")
            # print(f"Selected move: {selected_move}")
            # print(f"Move visits: {best_child.visit_count}")
            # print(f"Move confidence: {best_child.visit_count/total_visits:.1%}")

            return selected_move
        finally:
            # Restore training mode
            self.model.train()

    def expand_and_evaluate(self, node: Node):
        """Expand node and return value estimate"""
        try:
            # Get model predictions
            encoded_board = encode_board(node.board)
            policy, value = self.model(encoded_board[None, ...])

            # Debug output to check valid moves
            print("\nExpanding node:")
            print(node.board)
            print(f"Current turn: {node.board.current_turn}")

            # Add children for all legal moves
            pieces = (
                node.board.white_pieces
                if node.board.current_turn == Color.WHITE
                else node.board.black_pieces
            )

            # Debug output
            print("Available pieces:", len(pieces))
            for piece, pos in pieces:
                valid_moves = node.board.get_valid_moves(pos)
                print(f"{piece.piece_type} at {pos} has moves: {valid_moves}")

                for to_pos in valid_moves:
                    # Get policy probability for this move
                    move_idx = self.encode_move(pos, to_pos)
                    prior = policy[0, move_idx] if move_idx < len(policy[0]) else 0.0

                    # Create child node
                    child_board = node.board.copy()
                    child_board.move_piece(pos, to_pos)
                    node.children[(pos, to_pos)] = Node(
                        board=child_board, parent=node, prior=prior
                    )

            node.is_expanded = True
            if not node.children:
                print("WARNING: No valid moves found during expansion!")

            return value[0]

        except Exception as e:
            print(f"Error in expand_and_evaluate: {e}")
            import traceback

            print(traceback.format_exc())
            return 0.0

    def backup(self, search_path: List[Node], value: float):
        """Update statistics of all nodes in search path"""
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Value flips for opponent

    def encode_move(self, from_pos, to_pos):
        """Convert move to policy index"""
        # This is a simplified version - we'll need to handle all move types properly
        from_square = from_pos[0] * 8 + from_pos[1]
        to_square = to_pos[0] * 8 + to_pos[1]
        return from_square * 64 + to_square
