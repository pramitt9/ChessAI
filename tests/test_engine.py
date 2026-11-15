import sys
from pathlib import Path
from engine import evaluate_board, get_best_move, _quiescence, INF

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

def test_quiescence_handles_check_at_depth_limit():
    board = chess.Board("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1")

    assert board.is_check()
    assert board.legal_moves.count() > 0

    original_fen = board.fen()

    score = _quiescence(
        board,
        alpha=-INF,
        beta=INF,
        ply=0,
        qdepth=0
    )

    assert score > -INF
    assert board.fen() == original_fen

def test_quiescence_handles_quiet_position():
    board = chess.Board()
    original_fen = board.fen()

    score = _quiescence(
        board,
        alpha=-INF,
        beta=INF,
        ply=0,
        qdepth=0
    )

    assert score == 0
    assert board.fen() == original_fen


def test_quiescence_preserves_board_after_search():
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    original_fen = board.fen()

    _quiescence(
        board,
        alpha=-INF,
        beta=INF,
        ply=0,
        qdepth=2
    )

    assert board.fen() == original_fen


if __name__ == "__main__":
    test_material_evaluation()
    test_engine_returns_legal_move()
    test_terminal_position()
    print("All engine tests passed.")
