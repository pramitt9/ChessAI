import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_FILE = PROJECT_ROOT / "reports" / "stockfish_v1_v2_paired_predictions.csv"
V2_DATA_FILE = PROJECT_ROOT / "data" / "processed" / "stockfish_10k_v2.csv"
REPORT_DIR = PROJECT_ROOT / "reports"

KEY_COLUMNS = ["game_id", "ply"]
ERROR_COLUMN = "v2_absolute_error_cp"


def assign_score_band(score):
    if score < -500:
        return "Below -500"
    if score < -200:
        return "-500 to -200"
    if score < -100:
        return "-200 to -100"
    if score <= 100:
        return "-100 to +100"
    if score <= 200:
        return "+100 to +200"
    if score <= 500:
        return "+200 to +500"
    return "Above +500"


def assign_phase(phase):
    if phase >= 0.67:
        return "Opening"
    if phase >= 0.34:
        return "Middlegame"
    return "Endgame"


def assign_tactical_level(value):
    if value == 0:
        return "No detected tactical pressure"
    if value <= 2:
        return "Low (1-2)"
    return "High (3+)"


def grouped_error_report(df, group_column):
    rows = []

    for group, part in df.groupby(group_column, observed=False):
        errors = part["v2_prediction_cp"] - part["actual_score_cp"]
        absolute_errors = errors.abs()

        rows.append({
            group_column: str(group),
            "positions": int(len(part)),
            "mae_cp": float(absolute_errors.mean()),
            "median_absolute_error_cp": float(absolute_errors.median()),
            "rmse_cp": float(np.sqrt(np.mean(errors ** 2))),
            "mean_signed_error_cp": float(errors.mean()),
            "p90_absolute_error_cp": float(absolute_errors.quantile(0.90)),
            "p95_absolute_error_cp": float(absolute_errors.quantile(0.95)),
        })

    return pd.DataFrame(rows)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Paired predictions not found: {PREDICTIONS_FILE}. "
            "Run ml.compare_stockfish_v1_v2 first."
        )
    if not V2_DATA_FILE.exists():
        raise FileNotFoundError(f"V2 dataset not found: {V2_DATA_FILE}")

    print("Loading paired predictions and V2 features...")
    predictions = pd.read_csv(PREDICTIONS_FILE)
    v2 = pd.read_csv(V2_DATA_FILE)

    required_predictions = KEY_COLUMNS + [
        "fen",
        "actual_score_cp",
        "v1_prediction_cp",
        "v2_prediction_cp",
    ]
    required_features = KEY_COLUMNS + [
        "game_phase",
        "own_attacked_pieces",
        "opp_attacked_pieces",
        "own_hanging_pieces",
        "opp_hanging_pieces",
        "own_attacked_material",
        "opp_attacked_material",
    ]

    missing_predictions = set(required_predictions) - set(predictions.columns)
    missing_features = set(required_features) - set(v2.columns)

    if missing_predictions:
        raise ValueError(
            f"Prediction report missing columns: {missing_predictions}"
        )
    if missing_features:
        raise ValueError(f"V2 dataset missing columns: {missing_features}")

    if predictions.duplicated(KEY_COLUMNS).any():
        raise ValueError("Paired predictions contain duplicate game_id/ply keys.")
    if v2.duplicated(KEY_COLUMNS).any():
        raise ValueError("V2 dataset contains duplicate game_id/ply keys.")

    df = predictions.merge(
        v2[required_features],
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )

    if df["game_phase"].isna().any():
        raise ValueError("Some paired predictions could not be matched to V2 features.")

    df["v2_error_cp"] = df["v2_prediction_cp"] - df["actual_score_cp"]
    df[ERROR_COLUMN] = df["v2_error_cp"].abs()
    df["score_band"] = df["actual_score_cp"].apply(assign_score_band)
    df["phase_name"] = df["game_phase"].apply(assign_phase)

    # A simple feature-based proxy for tactical activity, not a chess-engine
    # tactical classifier or a legal exchange evaluation.
    df["tactical_activity"] = (
        df["own_attacked_pieces"]
        + df["opp_attacked_pieces"]
        + df["own_hanging_pieces"]
        + df["opp_hanging_pieces"]
    )
    df["tactical_level"] = df["tactical_activity"].apply(assign_tactical_level)

    overall = {
        "positions": int(len(df)),
        "v2_mae_cp": float(df[ERROR_COLUMN].mean()),
        "v2_median_absolute_error_cp": float(df[ERROR_COLUMN].median()),
        "v2_rmse_cp": float(np.sqrt(np.mean(df["v2_error_cp"] ** 2))),
        "v2_mean_signed_error_cp": float(df["v2_error_cp"].mean()),
        "v2_p90_absolute_error_cp": float(df[ERROR_COLUMN].quantile(0.90)),
        "v2_p95_absolute_error_cp": float(df[ERROR_COLUMN].quantile(0.95)),
        "v1_mae_on_same_positions_cp": float(
            np.abs(
                df["v1_prediction_cp"] - df["actual_score_cp"]
            ).mean()
        ),
    }

    score_report = grouped_error_report(df, "score_band")
    phase_report = grouped_error_report(df, "phase_name")
    tactical_report = grouped_error_report(df, "tactical_level")

    # Sort score bands in chess-evaluation order rather than alphabetically.
    score_order = [
        "Below -500",
        "-500 to -200",
        "-200 to -100",
        "-100 to +100",
        "+100 to +200",
        "+200 to +500",
        "Above +500",
    ]
    score_report["score_band"] = pd.Categorical(
        score_report["score_band"], categories=score_order, ordered=True
    )
    score_report = score_report.sort_values("score_band")

    # Export the worst positions for later inspection.
    worst = df.sort_values(ERROR_COLUMN, ascending=False).head(50).copy()
    worst_columns = KEY_COLUMNS + [
        "fen",
        "actual_score_cp",
        "v2_prediction_cp",
        "v2_error_cp",
        ERROR_COLUMN,
        "game_phase",
        "phase_name",
        "tactical_activity",
        "tactical_level",
        "own_hanging_pieces",
        "opp_hanging_pieces",
        "own_attacked_material",
        "opp_attacked_material",
    ]
    worst = worst[worst_columns]

    score_path = REPORT_DIR / "v2_error_by_score_band.csv"
    phase_path = REPORT_DIR / "v2_error_by_phase.csv"
    tactical_path = REPORT_DIR / "v2_error_by_tactical_activity.csv"
    worst_path = REPORT_DIR / "v2_worst_predictions.csv"
    summary_path = REPORT_DIR / "v2_error_analysis_summary.json"

    score_report.to_csv(score_path, index=False)
    phase_report.to_csv(phase_path, index=False)
    tactical_report.to_csv(tactical_path, index=False)
    worst.to_csv(worst_path, index=False)

    summary = {
        "method": (
            "Analysis of V2 predictions from the matched V1/V2 test positions; "
            "V1 Stockfish labels are the common target"
        ),
        "overall": overall,
        "by_score_band": score_report.astype({"score_band": str}).to_dict(
            orient="records"
        ),
        "by_phase": phase_report.to_dict(orient="records"),
        "by_tactical_activity": tactical_report.to_dict(orient="records"),
        "reports": {
            "score_band": str(score_path),
            "phase": str(phase_path),
            "tactical_activity": str(tactical_path),
            "worst_predictions": str(worst_path),
        },
    }

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\nOVERALL V2 TEST ERROR")
    print("---------------------")
    print(f"Positions: {overall['positions']:,}")
    print(f"MAE: {overall['v2_mae_cp']:.2f} cp")
    print(f"Median absolute error: {overall['v2_median_absolute_error_cp']:.2f} cp")
    print(f"RMSE: {overall['v2_rmse_cp']:.2f} cp")
    print(f"Mean signed error: {overall['v2_mean_signed_error_cp']:.2f} cp")
    print(f"90th percentile absolute error: {overall['v2_p90_absolute_error_cp']:.2f} cp")
    print(f"95th percentile absolute error: {overall['v2_p95_absolute_error_cp']:.2f} cp")
    print(f"V1 MAE on same positions: {overall['v1_mae_on_same_positions_cp']:.2f} cp")

    print("\nERROR BY ACTUAL STOCKFISH SCORE")
    print("-------------------------------")
    print(score_report.to_string(index=False))

    print("\nERROR BY GAME PHASE")
    print("-------------------")
    print(phase_report.to_string(index=False))

    print("\nERROR BY TACTICAL ACTIVITY")
    print("--------------------------")
    print(tactical_report.to_string(index=False))

    print("\nWORST 10 PREDICTIONS")
    print("--------------------")
    print(
        worst.head(10)[
            [
                "game_id",
                "ply",
                "actual_score_cp",
                "v2_prediction_cp",
                "v2_error_cp",
                "phase_name",
                "tactical_level",
            ]
        ].to_string(index=False)
    )

    print("\nSAVED REPORTS")
    print("-------------")
    print(score_path)
    print(phase_path)
    print(tactical_path)
    print(worst_path)
    print(summary_path)


if __name__ == "__main__":
    main()
