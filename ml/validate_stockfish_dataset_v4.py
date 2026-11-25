
import pandas as pd
from pathlib import Path

DATASET = Path("data/processed/stockfish_10k_v4.csv")

df = pd.read_csv(DATASET)

print("=" * 65)
print("V4 DATASET VALIDATION")
print("=" * 65)

# 1. Basic structure
print("\n1. Dataset structure")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Unique FENs:", df["fen"].nunique())

# 2. Feature validation
feature_columns = [
    c for c in df.columns
    if c not in {
        "game_id", "ply", "fen",
        "initial_score_cp", "initial_mate_in",
        "initial_raw_score", "score_cp", "mate_in",
        "is_mate", "raw_score", "was_rechecked",
        "verification_time",
        "v3_original_score_cp", "v3_original_mate_in",
        "v3_original_is_mate",
        "v4_score_cp", "v4_mate_in", "v4_is_mate",
        "v4_verified", "v4_label_source",
        "v4_verification_status",
        "v4_verification_depth", "v4_verification_nodes",
        "v4_verification_time", "v4_audit_record_count",
        "v4_label_valid", "v4_absolute_change_cp",
        "v4_mate_discovered",
    }
]

print("\n2. Feature columns:", len(feature_columns))

# 3. Missing features
missing_features = df[feature_columns].isna().sum()

print("\n3. Missing feature values:")
print(missing_features[missing_features > 0])

# 4. Label distribution
print("\n4. V4 mate distribution")
print(df["v4_is_mate"].value_counts(dropna=False))

print("\n5. Verification distribution")
print(df["v4_verification_status"].value_counts())

# 5. Label consistency
invalid_mate = (
    df["v4_is_mate"]
    & df["v4_score_cp"].notna()
)

invalid_cp = (
    ~df["v4_is_mate"]
    & df["v4_mate_in"].notna()
)

print("\n6. Label consistency")
print("Mate rows with CP labels:", invalid_mate.sum())
print("CP rows with mate distances:", invalid_cp.sum())

# 6. Verification statistics
print("\n7. Verified versus unverified")
print(df["v4_verified"].value_counts())

print("\nVerified unique FENs:",
      df.loc[df["v4_verified"], "fen"].nunique())

print("Unverified unique FENs:",
      df.loc[~df["v4_verified"], "fen"].nunique())

# 7. Duplicate FEN analysis
print("\n8. Duplicate FENs")

duplicate_rows = df[
    df.duplicated("fen", keep=False)
]

print("Rows belonging to repeated FENs:",
      len(duplicate_rows))

print("Unique repeated FENs:",
      duplicate_rows["fen"].nunique())

# 8. Final checks
print("\n9. Assertions")

assert len(df) == 10000
assert df["fen"].nunique() == 8666
assert df["v4_verified"].notna().all()
assert not invalid_mate.any()
assert not invalid_cp.any()
assert df["v4_label_valid"].all()

print("All validation assertions passed.")

print("\nV4 validation completed.")
