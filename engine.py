import chess

# Scores are in centipawns. Evaluation is always from White's perspective.
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 335,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}
MATE_SCORE = 100_000
INF = 1_000_000

# Piece-square tables are indexed from White's point of view (a1 = 0).
# Black's tables are mirrored vertically.
PST = {
    chess.PAWN: [
         0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
         5,  5, 10, 25, 25, 10,  5,  5,
         0,  0,  0, 20, 20,  0,  0,  0,
         5, -5,-10,  0,  0,-10, -5,  5,
         5, 10, 10,-20,-20, 10, 10,  5,
         0,  0,  0,  0,  0,  0,  0,  0,
    ],
    chess.KNIGHT: [
       -50,-40,-30,-30,-30,-30,-40,-50,
       -40,-20,  0,  5,  5,  0,-20,-40,
       -30,  5, 10, 15, 15, 10,  5,-30,
       -30,  0, 15, 20, 20, 15,  0,-30,
       -30,  5, 15, 20, 20, 15,  5,-30,
       -30,  0, 10, 15, 15, 10,  0,-30,
       -40,-20,  0,  0,  0,  0,-20,-40,
       -50,-40,-30,-30,-30,-30,-40,-50,
    ],
    chess.BISHOP: [
       -20,-10,-10,-10,-10,-10,-10,-20,
       -10,  5,  0,  0,  0,  0,  5,-10,
       -10, 10, 10, 10, 10, 10, 10,-10,
       -10,  0, 10, 10, 10, 10,  0,-10,
       -10,  5,  5, 10, 10,  5,  5,-10,
       -10,  0,  5, 10, 10,  5,  0,-10,
       -10,  0,  0,  0,  0,  0,  0,-10,
       -20,-10,-10,-10,-10,-10,-10,-20,
    ],
    chess.ROOK: [
         0,  0,  0,  5,  5,  0,  0,  0,
         5, 10, 10, 10, 10, 10, 10,  5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
         0,  0,  0,  5,  5,  0,  0,  0,
    ],
    chess.QUEEN: [
       -20,-10,-10, -5, -5,-10,-10,-20,
       -10,  0,  5,  0,  0,  0,  0,-10,
       -10,  5,  5,  5,  5,  5,  0,-10,
         0,  0,  5,  5,  5,  5,  0, -5,
        -5,  0,  5,  5,  5,  5,  0, -5,
       -10,  0,  5,  5,  5,  5,  0,-10,
       -10,  0,  0,  0,  0,  0,  0,-10,
       -20,-10,-10, -5, -5,-10,-10,-20,
    ],
    chess.KING: [
       -30,-40,-40,-50,-50,-40,-40,-30,
       -30,-40,-40,-50,-50,-40,-40,-30,
       -30,-40,-40,-50,-50,-40,-40,-30,
       -30,-40,-40,-50,-50,-40,-40,-30,
       -20,-30,-30,-40,-40,-30,-30,-20,
       -10,-20,-20,-20,-20,-20,-20,-10,
        20, 20,  0,  0,  0,  0, 20, 20,
        20, 30, 10,  0,  0, 10, 30, 20,
    ],
}


def _piece_square_value(piece_type, square, color):
    index = square if color == chess.WHITE else chess.square_mirror(square)
    return PST[piece_type][index]


def _pawn_structure(board, color):
    pawns = list(board.pieces(chess.PAWN, color))
    files = [0] * 8
    for sq in pawns:
        files[chess.square_file(sq)] += 1

    score = 0
    # Doubled pawns: extra pawns stacked on the same file are a weakness.
    for count in files:
        if count > 1:
            score -= 12 * (count - 1)

    # Isolated pawns have no friendly pawn on either adjacent file.
    for sq in pawns:
        file_ = chess.square_file(sq)
        adjacent = (file_ > 0 and files[file_ - 1] > 0) or (
            file_ < 7 and files[file_ + 1] > 0
        )
        if not adjacent:
            score -= 10

    # Passed pawns receive a bonus that grows as they approach promotion.
    enemy_pawns = board.pieces(chess.PAWN, not color)
    for sq in pawns:
        file_ = chess.square_file(sq)
        rank = chess.square_rank(sq)
        blocked_by_enemy = False
        for enemy_sq in enemy_pawns:
            enemy_file = chess.square_file(enemy_sq)
            enemy_rank = chess.square_rank(enemy_sq)
            if abs(enemy_file - file_) <= 1:
                if color == chess.WHITE and enemy_rank > rank:
                    blocked_by_enemy = True
                elif color == chess.BLACK and enemy_rank < rank:
                    blocked_by_enemy = True
        if not blocked_by_enemy:
            advance = rank if color == chess.WHITE else 7 - rank
            score += 12 + advance * 8
    return score


def _king_safety(board, color):
    king = board.king(color)
    if king is None:
        return -MATE_SCORE
    file_ = chess.square_file(king)
    rank = chess.square_rank(king)
    score = 0

    # A castled king generally benefits from a pawn shield.
    direction = 1 if color == chess.WHITE else -1
    shield_rank = rank + direction
    if 0 <= shield_rank <= 7:
        for f in range(max(0, file_ - 1), min(7, file_ + 1) + 1):
            sq = chess.square(f, shield_rank)
            if board.piece_at(sq) == chess.Piece(chess.PAWN, color):
                score += 9
    # Penalize a king that has lost castling rights in the opening/middlegame.
    if board.fullmove_number < 30 and not board.has_castling_rights(color):
        score -= 12
    return score


def evaluate_board(board):
    """Static position evaluation from White's perspective."""
    if board.is_checkmate():
        return -MATE_SCORE if board.turn == chess.WHITE else MATE_SCORE
    if board.is_game_over(claim_draw=True):
        return 0

    score = 0
    for piece_type, value in PIECE_VALUES.items():
        for sq in board.pieces(piece_type, chess.WHITE):
            score += value + _piece_square_value(piece_type, sq, chess.WHITE)
        for sq in board.pieces(piece_type, chess.BLACK):
            score -= value + _piece_square_value(piece_type, sq, chess.BLACK)

    # Bishop pair bonus.
    if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
        score += 28
    if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
        score -= 28

    score += _pawn_structure(board, chess.WHITE)
    score -= _pawn_structure(board, chess.BLACK)
    score += _king_safety(board, chess.WHITE)
    score -= _king_safety(board, chess.BLACK)

    # Small mobility term. Legal move generation is deliberately kept modest.
    turn = board.turn
    board.turn = chess.WHITE
    white_mobility = board.legal_moves.count()
    board.turn = chess.BLACK
    black_mobility = board.legal_moves.count()
    board.turn = turn
    score += 2 * (white_mobility - black_mobility)

    return int(score)


def _move_ordering(board, tt_move=None):
    def score(move):
        if tt_move is not None and move == tt_move:
            return 1_000_000
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
            value += PIECE_VALUES[move.promotion] + 800
        if board.gives_check(move):
            value += 40
        return value

    return sorted(list(board.legal_moves), key=score, reverse=True)



def _quiescence(board, alpha, beta, ply, qdepth=8, evaluator="handcrafted"):
    """Search forcing captures and check evasions at the horizon."""
    if board.is_checkmate():
        return -MATE_SCORE + ply

    if board.is_game_over(claim_draw=True):
        return 0

    in_check = board.is_check()

    # A checked side cannot use a static stand-pat evaluation.
    if not in_check:
        if evaluator == "ml":
            from ml_evaluator import evaluate_ml_white
            white_score = evaluate_ml_white(board)
        else:
            white_score = evaluate_board(board)

        stand_pat = white_score * (
            1 if board.turn == chess.WHITE else -1
        )

        if stand_pat >= beta:
            return beta

        alpha = max(alpha, stand_pat)

        # At the depth limit, a quiet position can be evaluated.
        if qdepth <= 0:
            return alpha

    # In check, search all legal evasions, even at the depth limit.
    moves = list(board.legal_moves)

    if not in_check:
        moves = [
            move for move in moves
            if board.is_capture(move) or move.promotion
        ]

    moves = sorted(
        moves,
        key=lambda move: (
            PIECE_VALUES.get(
                board.piece_type_at(move.to_square), 100
            ) if board.is_capture(move) else 0
        ) + (
            PIECE_VALUES.get(move.promotion, 0)
            if move.promotion else 0
        ),
        reverse=True,
    )

    if in_check and not moves:
        return -MATE_SCORE + ply

    for move in moves:
        board.push(move)
        score = -_quiescence(
            board, -beta, -alpha, ply + 1, qdepth - 1, evaluator
        )
        board.pop()

        if score >= beta:
            return beta

        alpha = max(alpha, score)

    return alpha



def _search(board, depth, alpha, beta, ply, table, evaluator="handcrafted"):
    if board.is_checkmate():
        return -MATE_SCORE + ply
    if board.is_game_over(claim_draw=True):
        return 0
    if depth <= 0:
        return _quiescence(board, alpha, beta, ply, evaluator=evaluator)

    key = board.fen()
    alpha_original = alpha
    tt_entry = table.get(key)
    tt_move = None
    if tt_entry:
        tt_depth, tt_score, tt_flag, tt_best = tt_entry
        tt_move = chess.Move.from_uci(tt_best) if tt_best else None
        if tt_depth >= depth:
            if tt_flag == "exact":
                return tt_score
            if tt_flag == "lower":
                alpha = max(alpha, tt_score)
            elif tt_flag == "upper":
                beta = min(beta, tt_score)
            if alpha >= beta:
                return tt_score

    best_score = -INF
    best_move = None
    for move in _move_ordering(board, tt_move):
        board.push(move)
        score = -_search(
            board, depth - 1, -beta, -alpha, ply + 1, table, evaluator
        )
        board.pop()

        if score > best_score:
            best_score = score
            best_move = move
        alpha = max(alpha, best_score)
        if alpha >= beta:
            break

    flag = "exact"
    if best_score <= alpha_original:
        flag = "upper"
    elif best_score >= beta:
        flag = "lower"
    table[key] = (depth, best_score, flag, best_move.uci() if best_move else None)
    return best_score


def get_best_move(board, depth=3, evaluator="handcrafted"):
    """Choose a legal move using negamax, alpha-beta, quiescence and a transposition table."""
    if board.is_game_over(claim_draw=True):
        return None
    if evaluator not in ("handcrafted", "ml"):
        raise ValueError("Evaluator must be 'handcrafted' or 'ml'.")
    if not 1 <= depth <= 6:
        raise ValueError("Depth must be between 1 and 6.")

    table = {}
    best_move = None
    best_score = -INF
    alpha, beta = -INF, INF

    for move in _move_ordering(board):
        board.push(move)
        score = -_search(
            board, depth - 1, -beta, -alpha, 1, table, evaluator
        )
        board.pop()
        if score > best_score:
            best_score = score
            best_move = move
        alpha = max(alpha, best_score)

    return best_move
