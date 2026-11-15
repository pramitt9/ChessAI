
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "data" / "processed" / "stockfish_10k_v2.csv"


def main():
    df = pd.read_csv(DATASET_PATH)

    print("DATASET SUMMARY")
    print("=" * 55)
    print(f"Total positions: {len(df)}")
    print(f"Unique games: {df['game_id'].nunique()}")

    print("\nAVAILABLE COLUMNS")
    print(list(df.columns))

    print("\nSCORE DISTRIBUTION")
    print(df["score_cp"].describe())

    print("\nEXTREME SCORES")
    print(f"Score <= -1000: {(df['score_cp'] <= -1000).sum()}")
    print(f"Score >= +1000: {(df['score_cp'] >= 1000).sum()}")

    print("\nIMPORTANT")
    print(
        "The dataset currently stores score_cp. "
        "It does not necessarily preserve whether a score "
        "originated from a forced-mate evaluation."
    )


if __name__ == "__main__":
    main()
