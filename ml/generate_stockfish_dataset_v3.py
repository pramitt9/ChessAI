
import csv
from pathlib import Path

import chess
import chess.pgn
import chess.engine

from ml.feature_extractor_v2 import (
    INPUT_FILE,
    FEATURE_COLUMNS,
    extract_features,
    sample_plies,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STOCKFISH_PATH = (
    PROJECT_ROOT / "engines" / "stockfish" / "stockfish.exe"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "stockfish_10k_v3.csv"
)

MAX_POSITIONS = 10_000

ANALYSIS_TIME = 0.1
VERIFICATION_TIME = 1.0
EXTREME_SCORE_THRESHOLD = 500

CSV_COLUMNS = [
    "game_id",
    "ply",
    "fen",
    "initial_score_cp",
    "initial_mate_in",
    "initial_raw_score",
    "score_cp",
    "mate_in",
    "is_mate",
    "raw_score",
    "was_rechecked",
    "verification_time",
] + FEATURE_COLUMNS


def evaluate_position(engine, board, time_limit):
    """Evaluate from the side-to-move perspective."""

    info = engine.analyse(
        board,
        chess.engine.Limit(time=time_limit),
    )

    score = info["score"].pov(board.turn)

    return {
        "score_cp": score.score(),
        "mate_in": score.mate(),
        "raw_score": str(score),
    }


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

    cp_positions = 0
    mate_positions = 0
    rechecked_positions = 0
    changed_labels = 0

    print("Starting Stockfish V3 generation...")
    print(f"Input: {INPUT_FILE}")
    print(f"Stockfish: {STOCKFISH_PATH}")
    print(f"Target positions: {MAX_POSITIONS:,}")
    print(f"Initial analysis: {ANALYSIS_TIME}s")
    print(f"Verification analysis: {VERIFICATION_TIME}s")
    print(f"Extreme score threshold: {EXTREME_SCORE_THRESHOLD} cp")

    with open(INPUT_FILE, "r", encoding="utf-8") as pgn_file, \
         open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=CSV_COLUMNS,
        )
        writer.writeheader()

        engine = chess.engine.SimpleEngine.popen_uci(
            str(STOCKFISH_PATH)
        )

        try:
            engine.configure({
                "Threads": 2,
                "Hash": 256,
            })

            while positions_saved < MAX_POSITIONS:

                game = chess.pgn.read_game(pgn_file)

                if game is None:
                    break

                if game.errors:
                    continue

                if game.board().chess960:
                    continue

                games_seen += 1

                total_plies = game.end().ply()
                sampled_plies = set(sample_plies(total_plies))

                if not sampled_plies:
                    continue

                board = game.board()

                for ply, move in enumerate(
                    game.mainline_moves(),
                    start=1,
                ):
                    board.push(move)

                    if ply not in sampled_plies:
                        continue

                    if board.is_game_over():
                        continue

                    positions_attempted += 1

                    # First analysis: quick evaluation.
                    initial = evaluate_position(
                        engine,
                        board,
                        ANALYSIS_TIME,
                    )

                    initial_cp = initial["score_cp"]
                    initial_mate = initial["mate_in"]

                    # Recheck mates and extreme centipawn evaluations.
                    needs_recheck = (
                        initial_mate is not None
                        or (
                            initial_cp is not None
                            and abs(initial_cp) >= EXTREME_SCORE_THRESHOLD
                        )
                    )

                    final = initial
                    verification_time = 0.0

                    if needs_recheck:
                        final = evaluate_position(
                            engine,
                            board,
                            VERIFICATION_TIME,
                        )

                        verification_time = VERIFICATION_TIME
                        rechecked_positions += 1

                        if (
                            final["score_cp"] != initial_cp
                            or final["mate_in"] != initial_mate
                        ):
                            changed_labels += 1

                    is_mate = final["mate_in"] is not None

                    row = {
                        "game_id": games_seen,
                        "ply": ply,
                        "fen": board.fen(),

                        "initial_score_cp": initial_cp,
                        "initial_mate_in": initial_mate,
                        "initial_raw_score": initial["raw_score"],

                        "score_cp": final["score_cp"],
                        "mate_in": final["mate_in"],
                        "is_mate": is_mate,
                        "raw_score": final["raw_score"],

                        "was_rechecked": needs_recheck,
                        "verification_time": verification_time,
                    }

                    row.update(extract_features(board))
                    writer.writerow(row)

                    positions_saved += 1

                    if is_mate:
                        mate_positions += 1
                    else:
                        cp_positions += 1

                    if positions_saved % 250 == 0:
                        print(
                            f"Saved {positions_saved:,}/"
                            f"{MAX_POSITIONS:,} | "
                            f"Games: {games_seen:,} | "
                            f"Mate: {mate_positions:,} | "
                            f"CP: {cp_positions:,} | "
                            f"Rechecked: {rechecked_positions:,}"
                        )

                    if positions_saved >= MAX_POSITIONS:
                        break

        finally:
            engine.quit()

    print("\nV3 GENERATION COMPLETE")
    print("=" * 55)
    print(f"Games read: {games_seen:,}")
    print(f"Positions attempted: {positions_attempted:,}")
    print(f"Positions saved: {positions_saved:,}")
    print(f"Centipawn positions: {cp_positions:,}")
    print(f"Mate positions: {mate_positions:,}")
    print(f"Positions rechecked: {rechecked_positions:,}")
    print(f"Labels changed after recheck: {changed_labels:,}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_dataset()
