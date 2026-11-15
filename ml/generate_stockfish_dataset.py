
import csv
import chess
import chess.pgn
import chess.engine

from pathlib import Path

from ml.feature_extractor import (
    INPUT_FILE,
    FEATURE_COLUMNS,
    extract_features,
    sample_plies,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STOCKFISH_PATH = PROJECT_ROOT / "engines" / "stockfish" / "stockfish.exe"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "stockfish_10k.csv"

MAX_POSITIONS = 10000
ANALYSIS_TIME = 0.1

CSV_COLUMNS = [
    "game_id",
    "ply",
    "fen",
    "score_cp",
] + FEATURE_COLUMNS


def evaluate_position(engine, board):
    """Return Stockfish evaluation in centipawns from side-to-move perspective."""

    info = engine.analyse(
        board,
        chess.engine.Limit(time=ANALYSIS_TIME)
    )

    score = info["score"].pov(board.turn).score()

    # Mate scores are not ordinary centipawn evaluations.
    if score is None:
        return None

    return score


def generate_dataset():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"PGN not found: {INPUT_FILE}")

    if not STOCKFISH_PATH.exists():
        raise FileNotFoundError(
            f"Stockfish executable not found: {STOCKFISH_PATH}"
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    games_seen = 0
    positions_attempted = 0
    positions_saved = 0
    mate_positions_skipped = 0

    print("Starting Stockfish dataset generation...")
    print(f"Input: {INPUT_FILE}")
    print(f"Stockfish: {STOCKFISH_PATH}")
    print(f"Target positions: {MAX_POSITIONS}")
    print(f"Analysis time per position: {ANALYSIS_TIME}s")

    engine = chess.engine.SimpleEngine.popen_uci(
        str(STOCKFISH_PATH)
    )

    try:
        engine.configure({
            "Threads": 2,
            "Hash": 256,
        })

        with open(INPUT_FILE, "r", encoding="utf-8") as pgn:
            with open(
                OUTPUT_FILE,
                "w",
                newline="",
                encoding="utf-8"
            ) as output:

                writer = csv.DictWriter(
                    output,
                    fieldnames=CSV_COLUMNS
                )
                writer.writeheader()

                while positions_saved < MAX_POSITIONS:

                    game = chess.pgn.read_game(pgn)

                    if game is None:
                        break

                    games_seen += 1

                    if game.errors:
                        continue

                    if game.headers.get("Variant", "Standard") != "Standard":
                        continue

                    if "FEN" in game.headers:
                        continue

                    result = game.headers.get("Result", "*")

                    if result not in ("1-0", "0-1", "1/2-1/2"):
                        continue

                    moves = list(game.mainline_moves())
                    targets = set(sample_plies(len(moves)))

                    if not targets:
                        continue

                    board = game.board()

                    for ply, move in enumerate(moves, start=1):

                        board.push(move)

                        if ply not in targets:
                            continue

                        if board.is_game_over(claim_draw=False):
                            continue

                        positions_attempted += 1

                        score_cp = evaluate_position(engine, board)

                        if score_cp is None:
                            mate_positions_skipped += 1
                            continue

                        row = {
                            "game_id": games_seen,
                            "ply": ply,
                            "fen": board.fen(),
                            "score_cp": score_cp,
                        }

                        row.update(extract_features(board))

                        writer.writerow(row)
                        positions_saved += 1

                        if positions_saved % 100 == 0:
                            print(
                                f"Saved {positions_saved:,}/"
                                f"{MAX_POSITIONS:,} positions | "
                                f"Games read: {games_seen:,} | "
                                f"Mate positions skipped: "
                                f"{mate_positions_skipped:,}"
                            )

                        if positions_saved >= MAX_POSITIONS:
                            break

    finally:
        engine.quit()

    print("\nSTOCKFISH DATASET GENERATION COMPLETE")
    print("-------------------------------------")
    print(f"Games read: {games_seen:,}")
    print(f"Positions evaluated: {positions_attempted:,}")
    print(f"Positions saved: {positions_saved:,}")
    print(f"Mate positions skipped: {mate_positions_skipped:,}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_dataset()
