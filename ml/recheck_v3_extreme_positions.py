
import pandas as pd
import chess
import chess.engine
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data/processed/stockfish_10k_v3.csv"

STOCKFISH_PATH = (
    PROJECT_ROOT / "engines/stockfish/stockfish.exe"
)

GAME_IDS = [368, 820, 1202, 108, 981, 579, 1163]

ANALYSIS_TIME = 5.0

def main():
    df = pd.read_csv(DATA_PATH)

    positions = df[df["game_id"].isin(GAME_IDS)].copy()

    positions = positions[
        positions["score_cp"].abs() >= 2000
    ]

    engine = chess.engine.SimpleEngine.popen_uci(
        str(STOCKFISH_PATH)
    )

    try:
        engine.configure({
            "Threads": 2,
            "Hash": 256,
        })

        results = []

        print("\nRECHECKING EXTREME V3 POSITIONS")
        print("=" * 75)

        for _, row in positions.iterrows():
            board = chess.Board(row["fen"])

            engine.configure({"Clear Hash": None})

            info = engine.analyse(
                board,
                chess.engine.Limit(time=ANALYSIS_TIME),
            )

            score = info["score"].pov(board.turn)

            result = {
                "game_id": int(row["game_id"]),
                "original_cp": row["score_cp"],
                "rechecked_cp": score.score(),
                "rechecked_mate": score.mate(),
                "raw_score": str(score),
                "depth": info.get("depth"),
                "nodes": info.get("nodes"),
                "time": info.get("time"),
                "fen": row["fen"],
            }

            results.append(result)

            print(
                f"\nGame {result['game_id']}"
                f"\nOriginal: {result['original_cp']}"
                f"\n5-second result: {result['raw_score']}"
                f"\nDepth: {result['depth']}"
                f"\nNodes: {result['nodes']}"
            )

        output = pd.DataFrame(results)

        output.to_csv(
            PROJECT_ROOT / "reports/v3_extreme_recheck.csv",
            index=False,
        )

        print("\nSaved: reports/v3_extreme_recheck.csv")

    finally:
        engine.quit()


if __name__ == "__main__":
    main()
