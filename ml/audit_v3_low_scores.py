
import chess
import chess.engine
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "data/processed/stockfish_10k_v3.csv"
STOCKFISH = ROOT / "engines/stockfish/stockfish.exe"
OUTPUT = ROOT / "reports/v3_low_score_audit.csv"

SAMPLE_SIZE = 100
AUDIT_TIME = 5.0
SEED = 42

df = pd.read_csv(DATASET)

cp = df[df["is_mate"] == False].copy()

# Keep only positions below 500 cp that were never rechecked.
unverified = cp[
    (cp["score_cp"].abs() < 500) &
    (~cp["was_rechecked"].astype(str).str.lower().isin(["true", "1"]))
].copy()

sample = unverified.sample(
    n=min(SAMPLE_SIZE, len(unverified)),
    random_state=SEED
)

print("Selected positions:", len(sample))
print("Starting five-second audit...")

results = []

engine = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH))

try:
    engine.configure({
        "Threads": 2,
        "Hash": 256
    })

    for _, row in sample.iterrows():
        board = chess.Board(row["fen"])

        engine.configure({"Clear Hash": None})

        info = engine.analyse(
            board,
            chess.engine.Limit(time=AUDIT_TIME)
        )

        score = info["score"].pov(board.turn)

        audit_cp = score.score()
        audit_mate = score.mate()

        original_cp = float(row["score_cp"])

        change = (
            None if audit_cp is None
            else abs(audit_cp - original_cp)
        )

        result = {
            "game_id": row["game_id"],
            "fen": row["fen"],
            "original_cp": original_cp,
            "audit_cp": audit_cp,
            "audit_mate": audit_mate,
            "raw_score": str(score),
            "absolute_change": change,
            "depth": info.get("depth"),
            "nodes": info.get("nodes"),
            "time": info.get("time")
        }

        results.append(result)

        print(
            f"Game {row['game_id']}: "
            f"{original_cp:+.0f} cp -> {score}"
        )

finally:
    engine.quit()

result_df = pd.DataFrame(results)
result_df.to_csv(OUTPUT, index=False)

cp_results = result_df[
    result_df["audit_mate"].isna()
]["absolute_change"].dropna()

print("\nLOW-SCORE AUDIT COMPLETE")
print("=" * 50)
print("Total audited:", len(result_df))
print("Mate discoveries:", result_df["audit_mate"].notna().sum())
print("Remaining CP:", len(cp_results))

if len(cp_results):
    print("Mean absolute change:", round(cp_results.mean(), 2))
    print("Median absolute change:", round(cp_results.median(), 2))
    print("Changes above 100 cp:", int((cp_results > 100).sum()))
    print("Changes above 300 cp:", int((cp_results > 300).sum()))

print("\nSaved:", OUTPUT)
