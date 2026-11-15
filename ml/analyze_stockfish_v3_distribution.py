
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "processed" / "stockfish_10k_v3.csv"


def main():
    df = pd.read_csv(DATASET)

    cp = df[~df["is_mate"]].copy()

    print("\nV3 DATASET DISTRIBUTION")
    print("=" * 60)

    print("\nCENTIPAWN SCORE STATISTICS")
    print(cp["score_cp"].describe(
        percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    ).round(2))

    print("\nCENTIPAWN SCORE BANDS")
    bands = [
        (-float("inf"), -1000, "Below -1000"),
        (-1000, -500, "-1000 to -500"),
        (-500, -200, "-500 to -200"),
        (-200, -100, "-200 to -100"),
        (-100, 100, "-100 to +100"),
        (100, 200, "+100 to +200"),
        (200, 500, "+200 to +500"),
        (500, 1000, "+500 to +1000"),
        (1000, float("inf"), "Above +1000"),
    ]

    for low, high, label in bands:
        count = ((cp["score_cp"] > low) & (cp["score_cp"] <= high)).sum()
        percentage = count / len(cp) * 100
        print(f"{label:18} {count:5} ({percentage:5.2f}%)")

    print("\nMATE DISTANCE DISTRIBUTION")
    print(df.loc[df["is_mate"], "mate_in"].describe().round(2))

    print("\nINITIAL VS FINAL LABEL CHANGES")

    initial_mate = df["initial_mate_in"].notna()
    final_mate = df["mate_in"].notna()

    print("Initial mate, final mate:",
          (initial_mate & final_mate).sum())
    print("Initial CP, final mate:",
          (~initial_mate & final_mate).sum())
    print("Initial mate, final CP:",
          (initial_mate & ~final_mate).sum())
    print("Initial CP, final CP:",
          (~initial_mate & ~final_mate).sum())

    print("\nFINAL CP SCORE CHANGES AFTER RECHECK")

    both_cp = ~initial_mate & ~final_mate

    changes = (
        df.loc[both_cp, "score_cp"]
        - df.loc[both_cp, "initial_score_cp"]
    ).abs()

    print("Positions:", len(changes))
    print("Mean absolute change:", round(changes.mean(), 2), "cp")
    print("Median absolute change:", round(changes.median(), 2), "cp")
    print("Changes above 100 cp:", (changes > 100).sum())
    print("Changes above 300 cp:", (changes > 300).sum())

    print("\nRECHECKED POSITIONS")
    print(df["was_rechecked"].value_counts())

    print("\nGAME COVERAGE")
    print("Unique games:", df["game_id"].nunique())
    print("Average positions per game:",
          round(df.groupby("game_id").size().mean(), 2))

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
