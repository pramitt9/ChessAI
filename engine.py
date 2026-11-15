
import chess

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


def evaluate_board(board):
    """
    Evaluates a chess position from White's perspective.
    Positive = White advantage
    Negative = Black advantage
    """

    if board.is_checkmate():
        return -100000 if board.turn == chess.WHITE else 100000

    if board.is_game_over(claim_draw=True):
        return 0

    score = 0

    for piece_type, value in PIECE_VALUES.items():

        white_pieces = len(
            board.pieces(piece_type, chess.WHITE)
        )

        black_pieces = len(
            board.pieces(piece_type, chess.BLACK)
        )

        score += (white_pieces - black_pieces) * value

    return score
