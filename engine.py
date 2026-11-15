import chess

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


def evaluate_board(board: chess.Board) -> int:
    """Evaluate from White's perspective. Positive means White is better."""
    if board.is_checkmate():
        return -100000 if board.turn == chess.WHITE else 100000
    if board.is_game_over(claim_draw=True):
        return 0

    score = 0
    for piece_type, value in PIECE_VALUES.items():
        score += len(board.pieces(piece_type, chess.WHITE)) * value
        score -= len(board.pieces(piece_type, chess.BLACK)) * value
    return score


def move_ordering(board: chess.Board):
    """Try forcing moves first to make alpha-beta pruning more effective."""
    def score(move):
        value = 0
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            if victim is None:  # en passant
                victim_value = PIECE_VALUES[chess.PAWN]
            else:
                victim_value = PIECE_VALUES[victim.piece_type]
            attacker = board.piece_at(move.from_square)
            attacker_value = PIECE_VALUES[attacker.piece_type]
            value += 10 * victim_value - attacker_value
        if move.promotion:
            value += PIECE_VALUES[move.promotion]
        if board.gives_check(move):
            value += 50
        return value

    return sorted(list(board.legal_moves), key=score, reverse=True)


def minimax(board: chess.Board, depth: int, alpha: float, beta: float, ply: int = 0) -> int:
    """Alpha-beta minimax. Scores are always from White's perspective."""
    if board.is_checkmate():
        return -100000 + ply if board.turn == chess.WHITE else 100000 - ply
    if board.is_game_over(claim_draw=True):
        return 0
    if depth <= 0:
        return evaluate_board(board)

    maximizing = board.turn == chess.WHITE
    if maximizing:
        best = -float("inf")
        for move in move_ordering(board):
            board.push(move)
            value = minimax(board, depth - 1, alpha, beta, ply + 1)
            board.pop()
            best = max(best, value)
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return int(best)

    best = float("inf")
    for move in move_ordering(board):
        board.push(move)
        value = minimax(board, depth - 1, alpha, beta, ply + 1)
        board.pop()
        best = min(best, value)
        beta = min(beta, best)
        if beta <= alpha:
            break
    return int(best)


def get_best_move(board: chess.Board, depth: int = 3):
    """Return the best legal move for the side to move."""
    if board.is_game_over(claim_draw=True):
        return None
    if not 1 <= depth <= 5:
        raise ValueError("Depth must be between 1 and 5.")

    maximizing = board.turn == chess.WHITE
    best_move = None
    best_score = -float("inf") if maximizing else float("inf")
    alpha, beta = -float("inf"), float("inf")

    for move in move_ordering(board):
        board.push(move)
        value = minimax(board, depth - 1, alpha, beta, 1)
        board.pop()

        if (maximizing and value > best_score) or (not maximizing and value < best_score):
            best_score, best_move = value, move

        if maximizing:
            alpha = max(alpha, best_score)
        else:
            beta = min(beta, best_score)

    return best_move
