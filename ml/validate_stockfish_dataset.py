
import csv
import math
import statistics
import chess

from pathlib import Path

from ml.feature_extractor import FEATURE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "stockfish_10k.csv"


def validate_dataset():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    rows = []
    invalid_fens = 0
    missing_values = 0
    invalid_features = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        required_columns = {
            "game_id", "ply", "fen", "score_cp"
        } | set(FEATURE_COLUMNS)

        actual_columns = set(reader.fieldnames or [])

        missing_columns = required_columns - actual_columns

        if missing_columns:
            raise ValueError(f"Missing columns: {missing_columns}")

        for row in reader:
            rows.append(row)

            try:
                board = chess.Board(row["fen"])
                if not board.is_valid():
                    invalid_fens += 1
            except (ValueError, TypeError):
                invalid_fens += 1

            for column in ["score_cp"] + FEATURE_COLUMNS:
                value = row[column]

                if value == "":
                    missing_values += 1
                    continue

                try:
                    number = float(value)
                    if not math.isfinite(number):
                        invalid_features += 1
                except ValueError:
                    invalid_features += 1

    if not rows:
        raise ValueError("Dataset is empty.")

    scores = [float(row["score_cp"]) for row in rows]

    scores_sorted = sorted(scores)

    def percentile(values, p):
        index = round((len(values) - 1) * p / 100)
        return values[index]

    print("\nSTOCKFISH DATASET VALIDATION")
    print("----------------------------")

    print(f"Total positions: {len(rows):,}")
    print(f"Feature columns: {len(FEATURE_COLUMNS)}")
    print(f"Total CSV columns: {len(rows[0])}")

    print("\nDATA QUALITY")
    print(f"Invalid FENs: {invalid_fens}")
    print(f"Missing values: {missing_values}")
    print(f"Invalid numerical values: {invalid_features}")

    print("\nSCORE STATISTICS (centipawns)")
    print(f"Minimum: {min(scores):.2f}")
    print(f"Maximum: {max(scores):.2f}")
    print(f"Mean: {statistics.mean(scores):.2f}")
    print(f"Median: {statistics.median(scores):.2f}")
    print(f"Standard deviation: {statistics.stdev(scores):.2f}")
    print(f"25th percentile: {percentile(scores_sorted, 25):.2f}")
    print(f"75th percentile: {percentile(scores_sorted, 75):.2f}")

    print("\nSCORE DISTRIBUTION")
    print(f"White/side-to-move advantage (> +100): {sum(s > 100 for s in scores)}")
    print(f"Near equal (-100 to +100): {sum(-100 <= s <= 100 for s in scores)}")
    print(f"Disadvantage (< -100): {sum(s < -100 for s in scores)}")

    print("\nVALIDATION RESULT")

    if invalid_fens == 0 and missing_values == 0 and invalid_features == 0:
        print("PASSED: Dataset has no detected data-quality errors.")
    else:
        print("FAILED: Investigate the errors above.")


if __name__ == "__main__":
    validate_dataset()
