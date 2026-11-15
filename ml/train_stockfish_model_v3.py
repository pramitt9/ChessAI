
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_sample_weight

from feature_extractor_v2 import FEATURE_COLUMNS


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_PATH = Path("data/processed/stockfish_10k_v3.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

RANDOM_STATE = 42
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.1764706  # 15% of the original full dataset

CLIP_LIMIT = 2000
N_JOBS = 2


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def parse_bool_column(series):
    """Safely convert CSV boolean values into actual booleans."""
    if series.dtype == bool:
        return series

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }

    converted = series.astype(str).str.strip().str.lower().map(mapping)

    if converted.isna().any():
        raise ValueError("Could not parse every is_mate value.")

    return converted.astype(bool)


def regression_metrics(y_true, predictions):
    y_true = np.asarray(y_true)
    predictions = np.asarray(predictions)

    correlation = np.corrcoef(y_true, predictions)[0, 1]

    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "r2": float(r2_score(y_true, predictions)),
        "correlation": float(correlation),
    }


def classifier_metrics(y_true, probabilities, threshold):
    predictions = probabilities >= threshold

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(
            y_true, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y_true, predictions, zero_division=0
        )),
        "f1": float(f1_score(
            y_true, predictions, zero_division=0
        )),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
        "confusion_matrix": confusion_matrix(
            y_true, predictions
        ).tolist(),
    }


def best_f1_threshold(y_true, probabilities):
    """Choose a classification threshold using validation data only."""
    thresholds = np.arange(0.05, 0.96, 0.01)

    scores = [
        f1_score(
            y_true,
            probabilities >= threshold,
            zero_division=0,
        )
        for threshold in thresholds
    ]

    return float(thresholds[int(np.argmax(scores))])


# --------------------------------------------------
# Load and deduplicate data
# --------------------------------------------------

def load_dataset():
    print("\nLoading V3 dataset...")

    raw = pd.read_csv(DATA_PATH)

    required_columns = (
        ["game_id", "ply", "fen", "score_cp", "is_mate"]
        + list(FEATURE_COLUMNS)
    )

    missing = [c for c in required_columns if c not in raw.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    raw["is_mate"] = parse_bool_column(raw["is_mate"])
    raw["score_cp"] = pd.to_numeric(
        raw["score_cp"], errors="coerce"
    )

    if raw["fen"].isna().any():
        raise ValueError("Dataset contains missing FEN values.")

    # Identical FENs must have consistent mate classifications.
    mate_conflicts = raw.groupby("fen")["is_mate"].nunique()

    if (mate_conflicts > 1).any():
        raise ValueError(
            "Some identical FENs have conflicting mate classifications."
        )

    # Keep one representative row per FEN.
    # All board features are determined by the FEN.
    unique_positions = raw.drop_duplicates(
        subset="fen", keep="first"
    ).copy()

    # Use median CP evaluation for repeated non-mate positions.
    # Mate positions have NaN score_cp and remain classified as mates.
    label_summary = raw.groupby("fen", sort=False).agg(
        score_cp=("score_cp", "median"),
        is_mate=("is_mate", "first"),
    )

    df = unique_positions.drop(
        columns=["score_cp", "is_mate"]
    ).merge(
        label_summary,
        left_on="fen",
        right_index=True,
        validate="one_to_one",
    )

    df = df.reset_index(drop=True)

    # Validate target consistency.
    if df.loc[~df["is_mate"], "score_cp"].isna().any():
        raise ValueError("Non-mate positions contain missing CP labels.")

    if df.loc[df["is_mate"], "score_cp"].notna().any():
        raise ValueError("Mate positions unexpectedly contain CP labels.")

    X = df[FEATURE_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )

    if X.isna().any().any():
        raise ValueError("Feature data contains missing or invalid values.")

    df.loc[:, FEATURE_COLUMNS] = X

    print(f"Original rows: {len(raw):,}")
    print(f"Unique FEN positions: {len(df):,}")
    print(f"Removed duplicate rows: {len(raw) - len(df):,}")
    print(f"Mate positions: {int(df['is_mate'].sum()):,}")
    print(f"CP positions: {int((~df['is_mate']).sum()):,}")
    print(f"Unique representative games: {df['game_id'].nunique():,}")

    return df


# --------------------------------------------------
# Game-based train / validation / test split
# --------------------------------------------------

def create_splits(df):
    print("\nCreating game-based splits...")

    indices = np.arange(len(df))
    groups = df["game_id"].to_numpy()

    outer_split = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_val_idx, test_idx = next(
        outer_split.split(indices, groups=groups)
    )

    inner_split = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
    )

    train_rel, val_rel = next(
        inner_split.split(
            train_val_idx,
            groups=groups[train_val_idx],
        )
    )

    train_idx = train_val_idx[train_rel]
    val_idx = train_val_idx[val_rel]

    train = df.iloc[train_idx].copy()
    val = df.iloc[val_idx].copy()
    test = df.iloc[test_idx].copy()

    # Verify that no representative game crosses splits.
    train_games = set(train["game_id"])
    val_games = set(val["game_id"])
    test_games = set(test["game_id"])

    assert train_games.isdisjoint(val_games)
    assert train_games.isdisjoint(test_games)
    assert val_games.isdisjoint(test_games)

    # FEN deduplication ensures no identical position crosses splits.
    assert set(train["fen"]).isdisjoint(val["fen"])
    assert set(train["fen"]).isdisjoint(test["fen"])
    assert set(val["fen"]).isdisjoint(test["fen"])

    for name, part in [
        ("Train", train),
        ("Validation", val),
        ("Test", test),
    ]:
        mates = int(part["is_mate"].sum())
        print(
            f"{name}: {len(part):,} positions | "
            f"{part['game_id'].nunique():,} games | "
            f"{mates:,} mates"
        )

    return train, val, test


# --------------------------------------------------
# Regression
# --------------------------------------------------

def train_regressors(train, val, test):
    print("\n" + "=" * 55)
    print("V3 CENTIPAWN REGRESSION")
    print("=" * 55)

    train = train[~train["is_mate"]].copy()
    val = val[~val["is_mate"]].copy()
    test = test[~test["is_mate"]].copy()

    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("A regression split contains no CP positions.")

    X_train = train[FEATURE_COLUMNS]
    X_val = val[FEATURE_COLUMNS]
    X_test = test[FEATURE_COLUMNS]

    y_train = train["score_cp"].to_numpy()
    y_val = val["score_cp"].to_numpy()
    y_test = test["score_cp"].to_numpy()

    candidates = {}

    for target_mode in ["raw", "clipped"]:
        if target_mode == "raw":
            training_target = y_train
        else:
            training_target = np.clip(
                y_train, -CLIP_LIMIT, CLIP_LIMIT
            )

        models = {
            "random_forest": RandomForestRegressor(
                n_estimators=300,
                max_depth=20,
                min_samples_leaf=3,
                max_features=0.9,
                n_jobs=N_JOBS,
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

        for model_name, model in models.items():
            candidate_name = f"{model_name}_{target_mode}"

            print(f"\nTraining {candidate_name}...")

            model.fit(X_train, training_target)

            val_predictions = model.predict(X_val)
            metrics = regression_metrics(y_val, val_predictions)

            candidates[candidate_name] = {
                "model": model,
                "validation_metrics": metrics,
            }

            print(
                f"Validation MAE: {metrics['mae']:.2f} cp | "
                f"RMSE: {metrics['rmse']:.2f} cp | "
                f"R2: {metrics['r2']:.4f}"
            )

    # Select using validation MAE on the original, unclipped labels.
    best_name = min(
        candidates,
        key=lambda name: candidates[name]["validation_metrics"]["mae"],
    )

    best_model = candidates[best_name]["model"]
    test_predictions = best_model.predict(X_test)

    test_metrics = regression_metrics(y_test, test_predictions)

    print("\nSelected regressor:", best_name)
    print("Final test results:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        best_model,
        MODEL_DIR / "stockfish_regressor_v3.joblib",
    )

    # Save predictions from every candidate on the untouched test set.
    prediction_data = pd.DataFrame({
        "fen": test["fen"].to_numpy(),
        "game_id": test["game_id"].to_numpy(),
        "actual_score_cp": y_test,
    })

    for name, result in candidates.items():
        prediction_data[f"pred_{name}"] = (
            result["model"].predict(X_test)
        )

    prediction_data.to_csv(
        REPORT_DIR / "v3_regression_test_predictions.csv",
        index=False,
    )

    comparison = []

    for name, result in candidates.items():
        row = {"model": name}
        row.update(result["validation_metrics"])
        comparison.append(row)

    pd.DataFrame(comparison).to_csv(
        REPORT_DIR / "v3_regression_comparison.csv",
        index=False,
    )

    return {
        "selected_model": best_name,
        "validation_metrics": candidates[best_name]["validation_metrics"],
        "test_metrics": test_metrics,
    }


# --------------------------------------------------
# Mate classification
# --------------------------------------------------

def train_mate_classifiers(train, val, test):
    print("\n" + "=" * 55)
    print("V3 MATE CLASSIFICATION")
    print("=" * 55)

    X_train = train[FEATURE_COLUMNS]
    X_val = val[FEATURE_COLUMNS]
    X_test = test[FEATURE_COLUMNS]

    y_train = train["is_mate"].astype(int).to_numpy()
    y_val = val["is_mate"].astype(int).to_numpy()
    y_test = test["is_mate"].astype(int).to_numpy()

    if len(np.unique(y_train)) < 2:
        raise ValueError("Training split needs both mate classes.")

    if len(np.unique(y_val)) < 2 or len(np.unique(y_test)) < 2:
        raise ValueError(
            "Validation and test splits must each contain both classes. "
            "Inspect split distributions before proceeding."
        )

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=3,
            max_features=0.9,
            class_weight="balanced",
            n_jobs=N_JOBS,
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            early_stopping=True,
            random_state=RANDOM_STATE,
        ),
    }

    candidates = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")

        if name == "hist_gradient_boosting":
            sample_weights = compute_sample_weight(
                class_weight="balanced",
                y=y_train,
            )
            model.fit(
                X_train,
                y_train,
                sample_weight=sample_weights,
            )
        else:
            model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_val)[:, 1]

        # Average precision is useful for an imbalanced positive class.
        val_ap = average_precision_score(y_val, probabilities)
        threshold = best_f1_threshold(y_val, probabilities)

        val_metrics = classifier_metrics(
            y_val, probabilities, threshold
        )

        candidates[name] = {
            "model": model,
            "validation_ap": float(val_ap),
            "threshold": threshold,
            "validation_metrics": val_metrics,
        }

        print(f"Validation average precision: {val_ap:.4f}")
        print(f"Validation F1: {val_metrics['f1']:.4f}")
        print(f"Validation threshold: {threshold:.2f}")

    # Select using validation average precision.
    best_name = max(
        candidates,
        key=lambda name: candidates[name]["validation_ap"],
    )

    best = candidates[best_name]
    best_model = best["model"]
    threshold = best["threshold"]

    test_probabilities = best_model.predict_proba(X_test)[:, 1]

    test_metrics = classifier_metrics(
        y_test,
        test_probabilities,
        threshold,
    )

    print("\nSelected classifier:", best_name)
    print("Test results:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value}")

    print("\nTest confusion matrix:")
    print(np.array(test_metrics["confusion_matrix"]))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        best_model,
        MODEL_DIR / "stockfish_mate_classifier_v3.joblib",
    )

    pd.DataFrame({
        "fen": test["fen"].to_numpy(),
        "game_id": test["game_id"].to_numpy(),
        "actual_is_mate": y_test,
        "mate_probability": test_probabilities,
        "predicted_is_mate": (
            test_probabilities >= threshold
        ).astype(int),
    }).to_csv(
        REPORT_DIR / "v3_mate_test_predictions.csv",
        index=False,
    )

    comparison = []

    for name, result in candidates.items():
        comparison.append({
            "model": name,
            "validation_average_precision": result["validation_ap"],
            "validation_f1": result["validation_metrics"]["f1"],
            "validation_threshold": result["threshold"],
        })

    pd.DataFrame(comparison).to_csv(
        REPORT_DIR / "v3_mate_classifier_comparison.csv",
        index=False,
    )

    return {
        "selected_model": best_name,
        "validation_average_precision": best["validation_ap"],
        "validation_threshold": threshold,
        "test_metrics": test_metrics,
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()

    train, val, test = create_splits(df)

    regression_results = train_regressors(
        train, val, test
    )

    classification_results = train_mate_classifiers(
        train, val, test
    )

    metadata = {
        "dataset": str(DATA_PATH),
        "feature_count": len(FEATURE_COLUMNS),
        "features": list(FEATURE_COLUMNS),
        "original_rows": 10000,
        "unique_fen_positions": len(df),
        "random_state": RANDOM_STATE,
        "split_method": "GroupShuffleSplit by representative game_id after FEN deduplication",
        "regression": regression_results,
        "classification": classification_results,
    }

    with open(
        REPORT_DIR / "stockfish_v3_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=4)

    print("\n" + "=" * 55)
    print("V3 TRAINING COMPLETE")
    print("=" * 55)
    print("Regressor: models/stockfish_regressor_v3.joblib")
    print("Mate classifier: models/stockfish_mate_classifier_v3.joblib")
    print("Reports: reports/")


if __name__ == "__main__":
    main()
