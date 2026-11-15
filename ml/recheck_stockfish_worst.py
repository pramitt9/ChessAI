
import csv
import chess
import chess.engine

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STOCKFISH_PATH = PROJECT_ROOT / "engines" / "stockfish" / "stockfish.exe"

INPUT_FILE = PROJECT_ROOT / "reports" / "stockfish_worst_predictions.csv"

OUTPUT_FILE = PROJECT_ROOT / "reports" / "stockfish_worst_rechecked.csv"

ANALYSIS_TIME = 1.0


def main():

    if not STOCKFISH_PATH.exists():
        raise FileNotFoundError(f"Stockfish not found: {STOCKFISH_PATH}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Worst predictions report not found: {INPUT_FILE}")

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("No positions found in the input report.")

    print(f"Positions to recheck: {len(rows)}")
    print(f"Analysis time per position: {ANALYSIS_TIME}s")

    results = []

    engine = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH_PATH))

    try:
        engine.configure({
            "Threads": 2,
            "Hash": 256,
        })

        for index, row in enumerate(rows, start=1):

            board = chess.Board(row["fen"])

            if not board.is_valid():
                print(f"Skipping invalid FEN at row {index}")
                continue

            info = engine.analyse(
                board,
                chess.engine.Limit(time=ANALYSIS_TIME),
            )

            score = info["score"].pov(board.turn)

            new_cp = score.score()
            new_mate = score.mate()

            old_cp = float(row["score_cp"])

            delta = (
                new_cp - old_cp
                if new_cp is not None
                else None
            )

            result = {
                "game_id": row["game_id"],
                "ply": row["ply"],
                "fen": row["fen"],
                "original_score_cp": old_cp,
                "rechecked_score_cp": new_cp,
                "rechecked_mate": new_mate,
                "score_change_cp": delta,
                "absolute_score_change_cp": (
                    abs(delta) if delta is not None else None
                ),
                "model_prediction_cp": float(row["predicted_cp"]),
                "model_error_original_cp": float(row["error_cp"]),
            }

            results.append(result)

            new_score_text = (
                f"{new_cp:+.0f} cp"
                if new_cp is not None
                else f"Mate in {new_mate}"
            )

            change_text = (
                f"{delta:+.0f} cp"
                if delta is not None
                else "Mate score"
            )

            print(
                f"[{index}/{len(rows)}] "
                f"Original: {old_cp:+.0f} cp | "
                f"Rechecked: {new_score_text} | "
                f"Change: {change_text}"
            )

    finally:
        engine.quit()

    if not results:
        raise RuntimeError("No positions were successfully rechecked.")

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    valid_changes = [
        r["absolute_score_change_cp"]
        for r in results
        if r["absolute_score_change_cp"] is not None
    ]

    print("\nRECHECK COMPLETE")
    print("----------------")
    print(f"Positions rechecked: {len(results)}")
    print(f"Positions with centipawn scores: {len(valid_changes)}")

    if valid_changes:
        print(
            f"Mean absolute score change: "
            f"{sum(valid_changes) / len(valid_changes):.2f} cp"
        )
        print(
            f"Maximum score change: "
            f"{max(valid_changes):.2f} cp"
        )

    print(f"Saved report: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
