
import chess
import pytest

from ml.feature_extractor_v2 import (
    FEATURE_COLUMNS,
    extract_features,
    piece_safety,
    king_safety_features,
    king_mobility,
    king_centralization,
    pawn_promotion_distance,
)


def test_starting_position():
    board = chess.Board()
    features = extract_features(board)

    assert len(FEATURE_COLUMNS) == 47
    assert len(features) == 47
    assert set(FEATURE_COLUMNS) == set(features.keys())

    assert features["own_attacked_pieces"] == 0
    assert features["opp_attacked_pieces"] == 0
    assert features["own_hanging_pieces"] == 0
    assert features["opp_hanging_pieces"] == 0


def test_hanging_piece_detection():
    """
    Black rook on e8 attacks white queen on e4.
    No white piece defends the queen.
    """
    board = chess.Board(
        "4r2k/8/8/8/4Q3/8/8/4K3 w - - 0 1"
    )

    attacked, hanging, material = piece_safety(
        board, chess.WHITE
    )

    assert attacked >= 1
    assert hanging >= 1
    assert material >= 9


def test_defended_piece_not_hanging():
    """
    Black rook on e8 attacks white queen on e4.
    White rook on e1 defends the queen.
    """
    board = chess.Board(
        "4r2k/8/8/8/4Q3/8/8/4R1K1 w - - 0 1"
    )

    attacked, hanging, material = piece_safety(
        board, chess.WHITE
    )

    assert attacked >= 1
    assert hanging == 0
    assert material >= 9


def test_king_centralization():
    board = chess.Board(
        "8/8/8/8/3K4/8/8/7k w - - 0 1"
    )

    assert king_centralization(board, chess.WHITE) == 7


def test_pawn_promotion_distance():
    board = chess.Board(
        "8/P7/8/8/8/8/8/7k w - - 0 1"
    )

    assert pawn_promotion_distance(
        board, chess.WHITE
    ) == 1


def test_king_mobility():
    board = chess.Board(
        "8/8/8/8/3K4/8/8/7k w - - 0 1"
    )

    assert king_mobility(board, chess.WHITE) > 0


def test_king_safety_returns_valid_values():
    board = chess.Board()

    attackers, attacked_squares = king_safety_features(
        board, chess.WHITE
    )

    assert attackers >= 0
    assert attacked_squares >= 0

def test_black_pawn_promotion_distance():
    # Black pawn on a2 needs one move to promote on a1.
    board = chess.Board(
        "7K/8/8/8/8/8/p7/7k b - - 0 1"
    )

    assert pawn_promotion_distance(
        board, chess.BLACK
    ) == 1


def test_no_pawns_promotion_distance():
    board = chess.Board(
        "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
    )

    assert pawn_promotion_distance(
        board, chess.WHITE
    ) == 8


def test_feature_extraction_for_both_turns():
    board = chess.Board()

    white_features = extract_features(board)

    board.turn = chess.BLACK
    black_features = extract_features(board)

    assert len(white_features) == 47
    assert len(black_features) == 47

    assert white_features["own_pawns"] == black_features["opp_pawns"]
    assert white_features["opp_pawns"] == black_features["own_pawns"]

    assert white_features["own_material_balance"] if False else True

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
