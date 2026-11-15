
import chess
from engine import evaluate_board

board = chess.Board()

print("Starting position:", evaluate_board(board))

# Remove Black's queen
board.remove_piece_at(chess.D8)

print("After removing Black's queen:", evaluate_board(board))

# Expected score: +900
assert evaluate_board(board) == 900

print("\nEvaluation tests passed!")
