
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PREDICTIONS_PATH = Path("reports/v3_regression_test_predictions.csv")
DATASET_PATH = Path("data/processed/stockfish_10k_v3.csv")
OUTPUT_DIR = Path("reports")

PREDICTION_COLUMN = "pred_hist_gradient_boosting_clipped"


def calculate_metrics(group):
    actual = group["actual_score_cp"]
    predicted = group["prediction"]
    errors = predicted - actual
    absolute_errors = errors.abs()

    return pd.Series({
        "positions": len(group),
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "mean_error": errors.mean(),
        "median_absolute_error": absolute_errors.median(),
        "within_50cp_pct": (absolute_errors <= 50).mean() * 100,
        "within_100cp_pct": (absolute_errors <= 100).mean() * 100,
        "within_200cp_pct": (absolute_errors <= 200).mean() * 100,
        "sign_accuracy_pct": (
            np.sign(actual) == np.sign(predicted)
        ).mean() * 100,
    })


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(PREDICTIONS_PATH)
    dataset = pd.read_csv(DATASET_PATH)

    if PREDICTION_COLUMN not in predictions.columns:
        raise ValueError(
            f"Missing prediction column: {PREDICTION_COLUMN}"
        )

    # Recover features associated with the held-out test positions.
    features = dataset[
        ["fen", "game_phase", "own_attacked_material",
         "opp_attacked_material", "own_hanging_pieces",
         "opp_hanging_pieces"]
    ].drop_duplicates("fen")

    df = predictions.merge(
        features,
        on="fen",
        how="left",
        validate="one_to_one",
    )

    if df["game_phase"].isna().any():
        raise ValueError("Some test positions could not be matched.")

    df = df.rename(columns={
        PREDICTION_COLUMN: "prediction"
    })

    df["error"] = df["prediction"] - df["actual_score_cp"]
    df["absolute_error"] = df["error"].abs()

    actual = df["actual_score_cp"]
    predicted = df["prediction"]

    print("\n" + "=" * 55)
    print("V3 REGRESSION ERROR ANALYSIS")
    print("=" * 55)

    print(f"Test positions: {len(df)}")
    print(f"MAE: {mean_absolute_error(actual, predicted):.2f} cp")
    print(
        f"RMSE: {np.sqrt(mean_squared_error(actual, predicted)):.2f} cp"
    )
    print(f"R2: {r2_score(actual, predicted):.4f}")
    print(f"Mean signed error: {df['error'].mean():.2f} cp")
    print(
        f"Median absolute error: "
        f"{df['absolute_error'].median():.2f} cp"
    )

    # --------------------------------------------------
    # 1. Accuracy by actual evaluation range
    # --------------------------------------------------

    bins = [-np.inf, -1000, -500, -200, -100, 100,
            200, 500, 1000, np.inf]

    labels = [
        "Below -1000",
        "-1000 to -500",
        "-500 to -200",
        "-200 to -100",
        "-100 to +100",
        "+100 to +200",
        "+200 to +500",
        "+500 to +1000",
        "Above +1000",
    ]

    df["score_range"] = pd.cut(
        df["actual_score_cp"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    range_results = (
        df.groupby("score_range", observed=False)
        .apply(calculate_metrics, include_groups=False)
        .reset_index()
    )

    print("\nERROR BY ACTUAL SCORE RANGE")
    print(range_results.round(2).to_string(index=False))

    range_results.to_csv(
        OUTPUT_DIR / "v3_error_by_score_range.csv",
        index=False,
    )

    # --------------------------------------------------
    # 2. Accuracy by game phase
    # --------------------------------------------------

    phase_results = (
        df.groupby("game_phase", dropna=False)
        .apply(calculate_metrics, include_groups=False)
        .reset_index()
    )

    print("\nERROR BY GAME PHASE")
    print(phase_results.round(2).to_string(index=False))

    phase_results.to_csv(
        OUTPUT_DIR / "v3_error_by_game_phase.csv",
        index=False,
    )

    # --------------------------------------------------
    # 3. Accuracy by tactical activity
    # --------------------------------------------------

    df["tactical_activity"] = (
        df["own_attacked_material"].fillna(0)
        + df["opp_attacked_material"].fillna(0)
        + df["own_hanging_pieces"].fillna(0)
        + df["opp_hanging_pieces"].fillna(0)
    )

    df["tactical_category"] = pd.cut(
        df["tactical_activity"],
        bins=[-0.001, 0, 2, 5, np.inf],
        labels=["None", "Low", "Medium", "High"],
    )

    tactical_results = (
        df.groupby("tactical_category", observed=False)
        .apply(calculate_metrics, include_groups=False)
        .reset_index()
    )

    print("\nERROR BY TACTICAL ACTIVITY")
    print(tactical_results.round(2).to_string(index=False))

    tactical_results.to_csv(
        OUTPUT_DIR / "v3_error_by_tactical_activity.csv",
        index=False,
    )

    # --------------------------------------------------
    # 4. Worst predictions
    # --------------------------------------------------

    worst = df.sort_values(
        "absolute_error",
        ascending=False,
    ).head(25)

    worst_columns = [
        "game_id",
        "fen",
        "actual_score_cp",
        "prediction",
        "error",
        "absolute_error",
        "game_phase",
        "tactical_activity",
    ]

    worst[worst_columns].to_csv(
        OUTPUT_DIR / "v3_worst_predictions.csv",
        index=False,
    )

    print("\n25 WORST PREDICTIONS")
    print(
        worst[[
            "game_id",
            "actual_score_cp",
            "prediction",
            "error",
            "absolute_error",
            "game_phase",
        ]].round(2).to_string(index=False)
    )

    # --------------------------------------------------
    # 5. Prediction direction
    # --------------------------------------------------

    sign_accuracy = (
        np.sign(actual) == np.sign(predicted)
    ).mean() * 100

    print("\nPREDICTION DIRECTION")
    print(f"Correct score sign: {sign_accuracy:.2f}%")
    print(f"Actual positive positions: {(actual > 0).sum()}")
    print(f"Actual negative positions: {(actual < 0).sum()}")
    print(f"Actual balanced positions: {(actual == 0).sum()}")

    print("\nAnalysis reports saved in reports/")


if __name__ == "__main__":
    main()
