
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error

from ml.feature_extractor import FEATURE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "stockfish_10k.csv"
MODEL_FILE = PROJECT_ROOT / "models" / "stockfish_regressor.joblib"
REPORT_DIR = PROJECT_ROOT / "reports"

RANDOM_STATE = 42
TARGET = "score_cp"


def recreate_test_split(df):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET]
    groups = df["game_id"]

    first = GroupShuffleSplit(
        n_splits=1,
        test_size=0.15,
        random_state=RANDOM_STATE,
    )

    train_val_idx, test_idx = next(
        first.split(X, y, groups)
    )

    train_val_groups = groups.iloc[train_val_idx]

    second = GroupShuffleSplit(
        n_splits=1,
        test_size=0.17647,
        random_state=RANDOM_STATE,
    )

    train_idx, val_idx = next(
        second.split(
            X.iloc[train_val_idx],
            y.iloc[train_val_idx],
            train_val_groups,
        )
    )

    train_global = train_val_idx[train_idx]
    val_global = train_val_idx[val_idx]

    return train_global, val_global, test_idx


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset and trained model...")

    df = pd.read_csv(DATA_FILE)
    model = joblib.load(MODEL_FILE)

    train_idx, val_idx, test_idx = recreate_test_split(df)

    train = df.iloc[train_idx].copy()
    val = df.iloc[val_idx].copy()
    test = df.iloc[test_idx].copy()

    print("\nSCORE DISTRIBUTION BY SPLIT")
    print("---------------------------")

    for name, part in [
        ("Train", train),
        ("Validation", val),
        ("Test", test),
    ]:
        print(
            f"{name}: "
            f"mean={part[TARGET].mean():.2f} cp | "
            f"median={part[TARGET].median():.2f} cp | "
            f"std={part[TARGET].std():.2f} cp"
        )

    X_test = test[FEATURE_COLUMNS]
    actual = test[TARGET].to_numpy()
    predicted = model.predict(X_test)

    test["predicted_cp"] = predicted
    test["error_cp"] = predicted - actual
    test["absolute_error_cp"] = np.abs(test["error_cp"])

    print("\nTEST ERROR SUMMARY")
    print("------------------")
    print(f"MAE: {mean_absolute_error(actual, predicted):.2f} cp")
    print(
        f"Median absolute error: "
        f"{test['absolute_error_cp'].median():.2f} cp"
    )
    print(
        f"90th percentile error: "
        f"{test['absolute_error_cp'].quantile(0.90):.2f} cp"
    )
    print(
        f"95th percentile error: "
        f"{test['absolute_error_cp'].quantile(0.95):.2f} cp"
    )

    print("\nERROR BY POSITION EVALUATION")
    print("----------------------------")

    bins = [-np.inf, -500, -200, -100, 100, 200, 500, np.inf]

    labels = [
        "Below -500",
        "-500 to -200",
        "-200 to -100",
        "-100 to +100",
        "+100 to +200",
        "+200 to +500",
        "Above +500",
    ]

    test["score_group"] = pd.cut(
        test[TARGET],
        bins=bins,
        labels=labels,
    )

    score_analysis = test.groupby(
        "score_group",
        observed=False,
    ).agg(
        positions=(TARGET, "size"),
        mean_absolute_error=("absolute_error_cp", "mean"),
        mean_actual=(TARGET, "mean"),
        mean_predicted=("predicted_cp", "mean"),
    )

    print(score_analysis.to_string())

    print("\nERROR BY GAME PHASE")
    print("-------------------")

    test["phase_group"] = pd.cut(
        test["game_phase"],
        bins=[-0.01, 0.25, 0.60, 1.01],
        labels=["Endgame", "Middlegame", "Opening"],
    )

    phase_analysis = test.groupby(
        "phase_group",
        observed=False,
    ).agg(
        positions=(TARGET, "size"),
        mean_absolute_error=("absolute_error_cp", "mean"),
    )

    print(phase_analysis.to_string())

    print("\n20 WORST PREDICTIONS")
    print("--------------------")

    worst = test.nlargest(
        20,
        "absolute_error_cp"
    )[
        [
            "game_id",
            "ply",
            "fen",
            TARGET,
            "predicted_cp",
            "error_cp",
            "absolute_error_cp",
            "game_phase",
        ]
    ]

    print(worst.to_string(index=False))

    worst.to_csv(
        REPORT_DIR / "stockfish_worst_predictions.csv",
        index=False,
    )

    score_analysis.to_csv(
        REPORT_DIR / "stockfish_error_by_score.csv"
    )

    phase_analysis.to_csv(
        REPORT_DIR / "stockfish_error_by_phase.csv"
    )

    test.to_csv(
        REPORT_DIR / "stockfish_test_predictions.csv",
        index=False,
    )

    print("\nANALYSIS COMPLETE")
    print("-----------------")
    print(f"Reports saved in: {REPORT_DIR}")


if __name__ == "__main__":
    main()
