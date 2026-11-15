
import pandas as pd
import numpy as np

pred_path = "reports/v3_regression_test_predictions.csv"
data_path = "data/processed/stockfish_10k_v3.csv"

pred = pd.read_csv(pred_path)
data = pd.read_csv(data_path)

best_column = "pred_hist_gradient_boosting_clipped"

pred["error"] = pred[best_column] - pred["actual_score_cp"]
pred["absolute_error"] = pred["error"].abs()

# Restore original labels and verification information.
columns = [
    "fen",
    "is_mate",
    "mate_in",
    "score_cp",
    "initial_score_cp",
    "initial_mate_in",
    "was_rechecked",
    "verification_time",
    "game_phase",
]

df = pred.merge(
    data[columns].drop_duplicates("fen"),
    on="fen",
    how="left",
    validate="one_to_one",
)

print("\nEXTREME LABEL INVESTIGATION")
print("=" * 75)

print("\nWorst 25 predictions with original label information:")

worst = df.nlargest(25, "absolute_error")

display_columns = [
    "game_id",
    "actual_score_cp",
    best_column,
    "error",
    "is_mate",
    "mate_in",
    "initial_score_cp",
    "initial_mate_in",
    "was_rechecked",
    "game_phase",
]

print(worst[display_columns].to_string(index=False))

# Investigate very large CP labels.
extreme = df[df["actual_score_cp"].abs() >= 2000]

print("\n\nPOSITIONS WITH ABSOLUTE SCORE >= 2000 CP")
print("=" * 75)
print("Count:", len(extreme))
print("Mate-labelled:", int(extreme["is_mate"].sum()))
print("Initially mate:", int(extreme["initial_mate_in"].notna().sum()))
print("Rechecked:", int(extreme["was_rechecked"].sum()))

print("\nExtreme label distribution:")
print(
    extreme[
        ["actual_score_cp", "initial_score_cp",
         "mate_in", "initial_mate_in", "was_rechecked"]
    ].describe(include="all").to_string()
)

print("\nInvestigation complete.")
