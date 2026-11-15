import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

from ml.feature_extractor import FEATURE_COLUMNS as V1_FEATURES
from ml.feature_extractor_v2 import FEATURE_COLUMNS as V2_FEATURES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V1_DATA = PROJECT_ROOT / "data" / "processed" / "stockfish_10k.csv"
V2_DATA = PROJECT_ROOT / "data" / "processed" / "stockfish_10k_v2.csv"
REPORT_DIR = PROJECT_ROOT / "reports"

RANDOM_STATE = 42
TARGET = "score_cp"
KEY_COLUMNS = ["game_id", "ply"]


def load_and_match_datasets():
    print("Loading V1 and V2 datasets...")

    if not V1_DATA.exists():
        raise FileNotFoundError(f"V1 dataset not found: {V1_DATA}")
    if not V2_DATA.exists():
        raise FileNotFoundError(f"V2 dataset not found: {V2_DATA}")

    v1 = pd.read_csv(V1_DATA)
    v2 = pd.read_csv(V2_DATA)

    required_v1 = KEY_COLUMNS + ["fen", TARGET] + V1_FEATURES
    required_v2 = KEY_COLUMNS + ["fen", TARGET] + V2_FEATURES

    missing_v1 = set(required_v1) - set(v1.columns)
    missing_v2 = set(required_v2) - set(v2.columns)

    if missing_v1:
        raise ValueError(f"V1 dataset is missing columns: {missing_v1}")
    if missing_v2:
        raise ValueError(f"V2 dataset is missing columns: {missing_v2}")

    if v1[required_v1].isnull().any().any():
        raise ValueError("V1 dataset contains missing required values.")
    if v2[required_v2].isnull().any().any():
        raise ValueError("V2 dataset contains missing required values.")

    for name, df in [("V1", v1), ("V2", v2)]:
        duplicates = df.duplicated(KEY_COLUMNS).sum()
        if duplicates:
            raise ValueError(
                f"{name} has {duplicates} duplicate game_id/ply keys. "
                "Cannot safely pair positions."
            )

    v1_columns = KEY_COLUMNS + ["fen", TARGET] + V1_FEATURES
    v2_columns = KEY_COLUMNS + ["fen", TARGET] + V2_FEATURES

    paired = v1[v1_columns].merge(
        v2[v2_columns],
        on=KEY_COLUMNS,
        how="inner",
        suffixes=("_v1", "_v2"),
        validate="one_to_one",
    )

    if paired.empty:
        raise ValueError("No matching positions found between V1 and V2.")

    fen_mismatch = paired["fen_v1"] != paired["fen_v2"]
    if fen_mismatch.any():
        examples = paired.loc[
            fen_mismatch, KEY_COLUMNS + ["fen_v1", "fen_v2"]
        ].head(5)
        raise ValueError(
            "Some matched game_id/ply keys have different FENs. "
            f"Examples:\n{examples.to_string(index=False)}"
        )

    label_mismatch = paired[f"{TARGET}_v1"] != paired[f"{TARGET}_v2"]

    print("\nDATASET ALIGNMENT")
    print("-----------------")
    print(f"V1 positions: {len(v1):,}")
    print(f"V2 positions: {len(v2):,}")
    print(f"Matched positions: {len(paired):,}")
    print(f"Unmatched V1 positions: {len(v1) - len(paired):,}")
    print(f"Unmatched V2 positions: {len(v2) - len(paired):,}")
    print(f"Matched games: {paired['game_id'].nunique():,}")
    print(f"FEN mismatches: {int(fen_mismatch.sum())}")
    print(f"Stockfish label mismatches: {int(label_mismatch.sum())}")

    if label_mismatch.any():
        differences = (
            paired.loc[label_mismatch, f"{TARGET}_v1"]
            - paired.loc[label_mismatch, f"{TARGET}_v2"]
        ).abs()
        print(
            "Absolute label difference for mismatches: "
            f"mean={differences.mean():.2f} cp, "
            f"median={differences.median():.2f} cp, "
            f"max={differences.max():.2f} cp"
        )
        print(
            "Using V1's Stockfish labels as the common target for BOTH "
            "feature versions."
        )

    # Construct feature matrices explicitly. Shared feature names were
    # suffixed by pandas during merge; V2-only feature names were not.
    v1_features = paired[[f"{feature}_v1" for feature in V1_FEATURES]].copy()
    v1_features.columns = V1_FEATURES

    v2_feature_data = {}
    v1_feature_set = set(V1_FEATURES)

    for feature in V2_FEATURES:
        if feature in v1_feature_set:
            source_column = f"{feature}_v2"
        else:
            source_column = feature
        v2_feature_data[feature] = paired[source_column].to_numpy()

    v2_features = pd.DataFrame(v2_feature_data, index=paired.index)

    targets = paired[f"{TARGET}_v1"].astype(float)

    for name, feature_frame in [
        ("V1", v1_features),
        ("V2", v2_features),
    ]:
        values = feature_frame.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{name} features contain non-finite values.")

    if not np.isfinite(targets.to_numpy(dtype=float)).all():
        raise ValueError("Common target contains non-finite values.")

    # Keep a clean, aligned frame with explicit V1 and V2 feature blocks.
    aligned = pd.DataFrame({
        "game_id": paired["game_id"].to_numpy(),
        "ply": paired["ply"].to_numpy(),
        "fen": paired["fen_v1"].to_numpy(),
        TARGET: targets.to_numpy(),
    })

    for feature in V1_FEATURES:
        aligned[f"v1__{feature}"] = v1_features[feature].to_numpy()

    for feature in V2_FEATURES:
        aligned[f"v2__{feature}"] = v2_features[feature].to_numpy()

    return aligned


def split_common_positions(df):
    groups = df["game_id"]

    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.15,
        random_state=RANDOM_STATE,
    )
    train_val_idx, test_idx = next(
        first_split.split(df, df[TARGET], groups)
    )

    train_val = df.iloc[train_val_idx]
    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.17647,
        random_state=RANDOM_STATE,
    )
    train_idx, val_idx = next(
        second_split.split(
            train_val,
            train_val[TARGET],
            train_val["game_id"],
        )
    )

    train_global_idx = train_val_idx[train_idx]
    val_global_idx = train_val_idx[val_idx]

    train_df = df.iloc[train_global_idx].copy()
    val_df = df.iloc[val_global_idx].copy()
    test_df = df.iloc[test_idx].copy()

    train_games = set(train_df["game_id"])
    val_games = set(val_df["game_id"])
    test_games = set(test_df["game_id"])

    assert train_games.isdisjoint(val_games)
    assert train_games.isdisjoint(test_games)
    assert val_games.isdisjoint(test_games)

    print("\nSHARED GAME-BASED SPLIT")
    print("-----------------------")
    for name, part in [
        ("Train", train_df),
        ("Validation", val_df),
        ("Test", test_df),
    ]:
        print(
            f"{name}: {len(part):,} positions | "
            f"{part['game_id'].nunique():,} games"
        )
    print("Game overlap check: PASSED")

    return train_df, val_df, test_df


def make_models():
    return {
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=3,
            max_features=0.9,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            early_stopping=True,
            random_state=RANDOM_STATE,
        ),
    }


def calculate_metrics(y_true, predictions):
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "r2": float(r2_score(y_true, predictions)),
        "correlation": float(np.corrcoef(y_true, predictions)[0, 1]),
    }


def train_and_evaluate_version(
    version, feature_names, train_df, val_df, test_df
):
    prefix = f"{version.lower()}__"
    columns = [f"{prefix}{feature}" for feature in feature_names]

    X_train = train_df[columns]
    y_train = train_df[TARGET]
    X_val = val_df[columns]
    y_val = val_df[TARGET]
    X_test = test_df[columns]
    y_test = test_df[TARGET]

    models = make_models()
    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining {version} {name}...")
        model.fit(X_train, y_train)

        val_predictions = model.predict(X_val)
        val_metrics = calculate_metrics(y_val, val_predictions)

        results[name] = {"validation": val_metrics}
        trained_models[name] = model

        print(
            f"{version} {name}: "
            f"Validation MAE={val_metrics['mae']:.2f} cp | "
            f"RMSE={val_metrics['rmse']:.2f} cp | "
            f"R2={val_metrics['r2']:.4f} | "
            f"Correlation={val_metrics['correlation']:.4f}"
        )

    selected_name = min(
        results,
        key=lambda name: results[name]["validation"]["mae"],
    )
    selected_model = trained_models[selected_name]

    test_predictions = selected_model.predict(X_test)
    test_metrics = calculate_metrics(y_test, test_predictions)
    results[selected_name]["test"] = test_metrics

    print(f"\n{version} selected model: {selected_name}")
    print(
        f"{version} TEST: "
        f"MAE={test_metrics['mae']:.2f} cp | "
        f"RMSE={test_metrics['rmse']:.2f} cp | "
        f"R2={test_metrics['r2']:.4f} | "
        f"Correlation={test_metrics['correlation']:.4f}"
    )

    return selected_name, results, test_predictions


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    paired = load_and_match_datasets()
    train_df, val_df, test_df = split_common_positions(paired)

    print("\nTRAINING V1")
    print("===========")
    v1_selected, v1_results, v1_predictions = train_and_evaluate_version(
        "V1", V1_FEATURES, train_df, val_df, test_df
    )

    print("\nTRAINING V2")
    print("===========")
    v2_selected, v2_results, v2_predictions = train_and_evaluate_version(
        "V2", V2_FEATURES, train_df, val_df, test_df
    )

    y_test = test_df[TARGET].to_numpy()
    comparison = pd.DataFrame({
        "game_id": test_df["game_id"].to_numpy(),
        "ply": test_df["ply"].to_numpy(),
        "fen": test_df["fen"].to_numpy(),
        "actual_score_cp": y_test,
        "v1_prediction_cp": v1_predictions,
        "v2_prediction_cp": v2_predictions,
        "v1_absolute_error_cp": np.abs(y_test - v1_predictions),
        "v2_absolute_error_cp": np.abs(y_test - v2_predictions),
    })

    prediction_path = REPORT_DIR / "stockfish_v1_v2_paired_predictions.csv"
    comparison.to_csv(prediction_path, index=False)

    summary = {
        "method": (
            "Matched game_id/ply positions; V1 score_cp labels used as "
            "common target; identical game-based train/validation/test split"
        ),
        "matched_positions": int(len(paired)),
        "matched_games": int(paired["game_id"].nunique()),
        "train_positions": int(len(train_df)),
        "validation_positions": int(len(val_df)),
        "test_positions": int(len(test_df)),
        "v1_selected_model": v1_selected,
        "v2_selected_model": v2_selected,
        "v1_results": v1_results,
        "v2_results": v2_results,
        "paired_test_mae_difference_v2_minus_v1": float(
            np.mean(comparison["v2_absolute_error_cp"])
            - np.mean(comparison["v1_absolute_error_cp"])
        ),
        "paired_test_v2_lower_error_positions": int(
            (
                comparison["v2_absolute_error_cp"]
                < comparison["v1_absolute_error_cp"]
            ).sum()
        ),
        "paired_test_v1_lower_error_positions": int(
            (
                comparison["v1_absolute_error_cp"]
                < comparison["v2_absolute_error_cp"]
            ).sum()
        ),
        "paired_test_tied_error_positions": int(
            (
                comparison["v1_absolute_error_cp"]
                == comparison["v2_absolute_error_cp"]
            ).sum()
        ),
        "prediction_file": str(prediction_path),
    }

    summary_path = REPORT_DIR / "stockfish_v1_v2_comparison.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\nCONTROLLED V1 vs V2 TEST COMPARISON")
    print("-----------------------------------")
    print(f"V1 selected model: {v1_selected}")
    print(f"V2 selected model: {v2_selected}")
    print(f"V1 test MAE: {v1_results[v1_selected]['test']['mae']:.2f} cp")
    print(f"V2 test MAE: {v2_results[v2_selected]['test']['mae']:.2f} cp")
    print(
        "V2 minus V1 test MAE: "
        f"{summary['paired_test_mae_difference_v2_minus_v1']:.2f} cp"
    )
    print(
        "Positions where V2 has lower absolute error: "
        f"{summary['paired_test_v2_lower_error_positions']:,}"
    )
    print(
        "Positions where V1 has lower absolute error: "
        f"{summary['paired_test_v1_lower_error_positions']:,}"
    )
    print(
        "Tied positions: "
        f"{summary['paired_test_tied_error_positions']:,}"
    )
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved paired predictions: {prediction_path}")


if __name__ == "__main__":
    main()
