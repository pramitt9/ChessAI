
from pathlib import Path
import json

import chess
import chess.engine
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

ENGINE_PATH = ROOT / "engines" / "stockfish" / "stockfish.exe"
WORST_PATH = ROOT / "reports" / "v2_worst_predictions.csv"
DATASET_PATH = ROOT / "data" / "processed" / "stockfish_10k_v2.csv"
OUTPUT_PATH = ROOT / "reports" / "v2_worst10_stockfish_recheck.csv"

ANALYSIS_TIME = 1.0
TOP_N = 10


def main():
    print("Loading worst predictions and dataset...")

    if not ENGINE_PATH.exists():
        raise FileNotFoundError(f"Stockfish not found: {ENGINE_PATH}")

    worst = pd.read_csv(WORST_PATH).head(TOP_N)
    dataset = pd.read_csv(DATASET_PATH)

    required_worst = {"game_id", "ply", "actual_score_cp", "v2_prediction_cp"}
    required_dataset = {"game_id", "ply", "fen"}

    if not required_worst.issubset(worst.columns):
        raise ValueError(
            f"Missing columns in worst predictions: "
            f"{required_worst - set(worst.columns)}"
        )

    if not required_dataset.issubset(dataset.columns):
        raise ValueError(
            f"Missing columns in dataset: "
            f"{required_dataset - set(dataset.columns)}"
        )

    if "fen" in worst.columns:
        worst = worst.drop(columns=["fen"])

    positions = worst.merge(
        dataset[["game_id", "ply", "fen"]],
        on=["game_id", "ply"],
        how="left",
        validate="one_to_one"
    )

    if positions["fen"].isna().any():
        raise ValueError("Some positions could not be matched to their FEN.")

    engine = chess.engine.SimpleEngine.popen_uci(str(ENGINE_PATH))

    results = []

    try:
        engine.configure({
            "Threads": 2,
            "Hash": 256
        })

        for index, row in positions.iterrows():
            board = chess.Board(row["fen"])

            print(
                f"\nAnalysing {len(results) + 1}/{len(positions)} "
                f"| Game {row['game_id']} | Ply {row['ply']}"
            )

            info = engine.analyse(
                board,
                chess.engine.Limit(time=ANALYSIS_TIME)
            )

            score = info["score"].pov(board.turn)
            mate_in = score.mate()

            # Convert mate scores to a large centipawn value
            rechecked_score = score.score(mate_score=10000)

            pv = info.get("pv", [])
            best_move = pv[0] if pv else None

            pv_san = []
            temp_board = board.copy()

            for move in pv[:8]:
                pv_san.append(temp_board.san(move))
                temp_board.push(move)

            original_score = float(row["actual_score_cp"])
            prediction = float(row["v2_prediction_cp"])

            results.append({
                "game_id": int(row["game_id"]),
                "ply": int(row["ply"]),
                "original_score_cp": original_score,
                "rechecked_score_cp": rechecked_score,
                "label_change_cp": rechecked_score - original_score,
                "v2_prediction_cp": prediction,
                "error_vs_original_cp": prediction - original_score,
                "error_vs_rechecked_cp": prediction - rechecked_score,
                "mate_in": mate_in,
                "best_move": (
                    board.san(best_move) if best_move else None
                ),
                "principal_variation": " ".join(pv_san),
                "depth": info.get("depth"),
                "nodes": info.get("nodes")
            })

            print(f"  Original label: {original_score:.1f} cp")
            print(f"  Rechecked:      {rechecked_score:.1f} cp")
            print(f"  V2 prediction:  {prediction:.1f} cp")
            print(f"  Best move:      {results[-1]['best_move']}")
            print(f"  Depth:          {info.get('depth')}")

    finally:
        engine.quit()

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 65)
    print("STOCKFISH RECHECK SUMMARY")
    print("=" * 65)

    for _, row in result_df.iterrows():
        print(
            f"Game {row['game_id']:>5} | Ply {row['ply']:>3} | "
            f"Original {row['original_score_cp']:>8.1f} | "
            f"Rechecked {row['rechecked_score_cp']:>8.1f} | "
            f"Label change {row['label_change_cp']:>8.1f}"
        )

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
