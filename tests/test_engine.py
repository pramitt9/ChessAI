import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess
from engine import evaluate_board, get_best_move


def test_material_evaluation():
    board = chess.Board()
    assert evaluate_board(board) == 0
    board.remove_piece_at(chess.D8)
    assert evaluate_board(board) > 0


def test_engine_returns_legal_move():
    board = chess.Board()
    move = get_best_move(board, depth=2)
    assert move is not None
    assert move in board.legal_moves


def test_terminal_position():
    board = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
    assert board.is_checkmate()
    assert get_best_move(board, depth=2) is None


if __name__ == "__main__":
    test_material_evaluation()
    test_engine_returns_legal_move()
    test_terminal_position()
    print("All engine tests passed.")
