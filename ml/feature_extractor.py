
import csv
import chess
import chess.pgn

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "sample_100k.pgn"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "positions.csv"

POSITIONS_PER_GAME = 8

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}

PHASE_VALUES = {
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
}

FEATURE_COLUMNS = [
    "own_pawns", "own_knights", "own_bishops",
    "own_rooks", "own_queens",
    "opp_pawns", "opp_knights", "opp_bishops",
    "opp_rooks", "opp_queens",
    "material_balance",
    "own_mobility", "opp_mobility", "mobility_balance",
    "own_doubled_pawns", "opp_doubled_pawns",
    "own_isolated_pawns", "opp_isolated_pawns",
    "own_passed_pawns", "opp_passed_pawns",
    "own_bishop_pair", "opp_bishop_pair",
    "own_castling_rights", "opp_castling_rights",
    "own_king_shield", "opp_king_shield",
    "center_control_balance",
    "own_developed_minors", "opp_developed_minors",
    "game_phase",
    "in_check",
]

CSV_COLUMNS = [
    "game_id",
    "ply",
    "result",
    "label",
] + FEATURE_COLUMNS


def material_count(board, color):
    return sum(
        len(board.pieces(piece_type, color)) * value
        for piece_type, value in PIECE_VALUES.items()
    )


def mobility_count(board, color):
    copy = board.copy(stack=False)
    copy.turn = color

    # En passant is available only to the actual side to move.
    if color != board.turn:
        copy.ep_square = None

    return copy.legal_moves.count()


def pawn_structure(board, color):
    pawns = list(board.pieces(chess.PAWN, color))
    enemy_pawns = list(board.pieces(chess.PAWN, not color))

    file_counts = [0] * 8

    for square in pawns:
        file_counts[chess.square_file(square)] += 1

    doubled = sum(max(0, count - 1) for count in file_counts)

    isolated = 0
    passed = 0

    for square in pawns:
        file_ = chess.square_file(square)
        rank = chess.square_rank(square)

        has_adjacent_friendly = (
            (file_ > 0 and file_counts[file_ - 1] > 0)
            or (file_ < 7 and file_counts[file_ + 1] > 0)
        )

        if not has_adjacent_friendly:
            isolated += 1

        blocked_by_enemy_pawn = False

        for enemy_square in enemy_pawns:
            enemy_file = chess.square_file(enemy_square)
            enemy_rank = chess.square_rank(enemy_square)

            if abs(enemy_file - file_) <= 1:
                if color == chess.WHITE and enemy_rank > rank:
                    blocked_by_enemy_pawn = True
                elif color == chess.BLACK and enemy_rank < rank:
                    blocked_by_enemy_pawn = True

        if not blocked_by_enemy_pawn:
            passed += 1

    return doubled, isolated, passed


def king_shield(board, color):
    king = board.king(color)

    if king is None:
        return 0

    file_ = chess.square_file(king)
    rank = chess.square_rank(king)

    forward = 1 if color == chess.WHITE else -1
    shield_rank = rank + forward

    if not 0 <= shield_rank <= 7:
        return 0

    count = 0

    for f in range(max(0, file_ - 1), min(7, file_ + 1) + 1):
        square = chess.square(f, shield_rank)

        if board.piece_at(square) == chess.Piece(chess.PAWN, color):
            count += 1

    return count


def castling_rights(board, color):
    return (
        int(board.has_kingside_castling_rights(color))
        + int(board.has_queenside_castling_rights(color))
    )


def developed_minors(board, color):
    home_rank = 0 if color == chess.WHITE else 7

    original_squares = (
        {chess.B1, chess.G1, chess.C1, chess.F1}
        if color == chess.WHITE
        else {chess.B8, chess.G8, chess.C8, chess.F8}
    )

    developed = 0

    for piece_type in (chess.KNIGHT, chess.BISHOP):
        for square in board.pieces(piece_type, color):
            if square not in original_squares:
                developed += 1

    return developed


def center_control(board, color):
    center_squares = [
        chess.square(file_, rank)
        for file_ in range(2, 6)
        for rank in range(2, 6)
    ]

    return sum(
        board.is_attacked_by(color, square)
        for square in center_squares
    )


def game_phase(board):
    phase = 0

    for piece_type, weight in PHASE_VALUES.items():
        phase += (
            len(board.pieces(piece_type, chess.WHITE))
            + len(board.pieces(piece_type, chess.BLACK))
        ) * weight

    return min(phase / 24.0, 1.0)


def extract_features(board):
    """Extract features from the side-to-move perspective."""

    own = board.turn
    opponent = not own

    own_counts = {
        piece: len(board.pieces(piece, own))
        for piece in PIECE_VALUES
    }

    opp_counts = {
        piece: len(board.pieces(piece, opponent))
        for piece in PIECE_VALUES
    }

    own_material = material_count(board, own)
    opp_material = material_count(board, opponent)

    own_mobility = mobility_count(board, own)
    opp_mobility = mobility_count(board, opponent)

    own_pawns = pawn_structure(board, own)
    opp_pawns = pawn_structure(board, opponent)

    own_bishops = own_counts[chess.BISHOP]
    opp_bishops = opp_counts[chess.BISHOP]

    return {
        "own_pawns": own_counts[chess.PAWN],
        "own_knights": own_counts[chess.KNIGHT],
        "own_bishops": own_bishops,
        "own_rooks": own_counts[chess.ROOK],
        "own_queens": own_counts[chess.QUEEN],

        "opp_pawns": opp_counts[chess.PAWN],
        "opp_knights": opp_counts[chess.KNIGHT],
        "opp_bishops": opp_bishops,
        "opp_rooks": opp_counts[chess.ROOK],
        "opp_queens": opp_counts[chess.QUEEN],

        "material_balance": own_material - opp_material,

        "own_mobility": own_mobility,
        "opp_mobility": opp_mobility,
        "mobility_balance": own_mobility - opp_mobility,

        "own_doubled_pawns": own_pawns[0],
        "opp_doubled_pawns": opp_pawns[0],
        "own_isolated_pawns": own_pawns[1],
        "opp_isolated_pawns": opp_pawns[1],
        "own_passed_pawns": own_pawns[2],
        "opp_passed_pawns": opp_pawns[2],

        "own_bishop_pair": int(own_bishops >= 2),
        "opp_bishop_pair": int(opp_bishops >= 2),

        "own_castling_rights": castling_rights(board, own),
        "opp_castling_rights": castling_rights(board, opponent),

        "own_king_shield": king_shield(board, own),
        "opp_king_shield": king_shield(board, opponent),

        "center_control_balance": (
            center_control(board, own)
            - center_control(board, opponent)
        ),

        "own_developed_minors": developed_minors(board, own),
        "opp_developed_minors": developed_minors(board, opponent),

        "game_phase": game_phase(board),

        "in_check": int(board.is_check()),
    }


def sample_plies(total_plies):
    """Choose up to eight evenly spaced positions, excluding the final position."""

    if total_plies <= 1:
        return []

    available = total_plies - 1
    sample_count = min(POSITIONS_PER_GAME, available)

    if sample_count == 1:
        return [1]

    return sorted({
        round(i * (available - 1) / (sample_count - 1)) + 1
        for i in range(sample_count)
    })


def get_label(result, side_to_move):
    """Return outcome from the current side-to-move perspective."""

    if result == "1/2-1/2":
        return 0

    winner = chess.WHITE if result == "1-0" else chess.BLACK

    return 1 if side_to_move == winner else -1


def process_game(game, game_id, writer):
    result = game.headers.get("Result", "*")

    if result not in ("1-0", "0-1", "1/2-1/2"):
        return 0

    if game.errors:
        return 0

    if game.headers.get("Variant", "Standard") != "Standard":
        return 0

    if "FEN" in game.headers:
        return 0

    moves = list(game.mainline_moves())
    targets = set(sample_plies(len(moves)))

    if not targets:
        return 0

    board = game.board()
    saved = 0

    for ply, move in enumerate(moves, start=1):
        board.push(move)

        if ply not in targets:
            continue

        if board.is_game_over(claim_draw=False):
            continue

        row = {
            "game_id": game_id,
            "ply": ply,
            "result": result,
            "label": get_label(result, board.turn),
        }

        row.update(extract_features(board))
        writer.writerow(row)
        saved += 1

    return saved


def extract_dataset():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input PGN not found: {INPUT_FILE}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    games_seen = 0
    games_used = 0
    positions_saved = 0

    print("Starting feature extraction...")
    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")

    with open(INPUT_FILE, "r", encoding="utf-8") as pgn:
        with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as output:

            writer = csv.DictWriter(
                output,
                fieldnames=CSV_COLUMNS
            )
            writer.writeheader()

            while True:
                game = chess.pgn.read_game(pgn)

                if game is None:
                    break

                games_seen += 1

                saved = process_game(game, games_seen, writer)

                if saved:
                    games_used += 1
                    positions_saved += saved

                if games_seen % 10_000 == 0:
                    print(
                        f"Read {games_seen:,} games | "
                        f"Saved {positions_saved:,} positions"
                    )

    print("\nFEATURE EXTRACTION COMPLETE")
    print("---------------------------")
    print(f"Games read: {games_seen:,}")
    print(f"Games used: {games_used:,}")
    print(f"Positions saved: {positions_saved:,}")
    print(f"Features per position: {len(FEATURE_COLUMNS)}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    extract_dataset()
