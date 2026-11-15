
import random
import chess
import chess.engine
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "data" / "processed" / "stockfish_10k_v3.csv"
STOCKFISH = ROOT / "engines" / "stockfish" / "stockfish.exe"
OUTPUT = ROOT / "reports" / "v3_label_audit.csv"

AUDIT_TIME = 5.0
SEED = 42

df = pd.read_csv(DATASET)

cp = df[df["is_mate"] == False].copy()
cp["abs_score"] = cp["score_cp"].abs()

random.seed(SEED)

sample_500_1000 = cp[
    (cp["abs_score"] >= 500) &
    (cp["abs_score"] < 1000)
].sample(n=20, random_state=SEED)

sample_1000_2000 = cp[
    (cp["abs_score"] >= 1000) &
    (cp["abs_score"] < 2000)
].sample(n=20, random_state=SEED)

sample_2000_plus = cp[
    cp["abs_score"] >= 2000
]

audit = pd.concat([
    sample_500_1000,
    sample_1000_2000,
    sample_2000_plus
]).copy()

print("Positions selected:", len(audit))
print("Starting Stockfish audit...")

results = []

engine = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH))

try:
    engine.configure({
        "Threads": 2,
        "Hash": 256
    })

    for _, row in audit.iterrows():
        board = chess.Board(row["fen"])

        engine.configure({"Clear Hash": None})

        info = engine.analyse(
            board,
            chess.engine.Limit(time=AUDIT_TIME)
        )

        score = info["score"].pov(board.turn)

        result = {
            "game_id": row["game_id"],
            "fen": row["fen"],
            "original_cp": row["score_cp"],
            "original_mate": row["mate_in"],
            "was_rechecked": row["was_rechecked"],
            "audit_cp": score.score(),
            "audit_mate": score.mate(),
            "raw_score": str(score),
            "depth": info.get("depth"),
            "nodes": info.get("nodes"),
            "time": info.get("time")
        }

        results.append(result)

        print(
            f"Game {row['game_id']}: "
            f"{row['score_cp']:.0f} cp -> {score}"
        )

finally:
    engine.quit()

result_df = pd.DataFrame(results)
result_df.to_csv(OUTPUT, index=False)

print("\nAudit complete.")
print("Saved:", OUTPUT)
print("Mate discoveries:", result_df["audit_mate"].notna().sum())
print(
    "Mean CP change:",
    (
        result_df["audit_cp"] - result_df["original_cp"]
    ).abs().mean()
)
