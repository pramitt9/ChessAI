import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor, HistGradientBoostingRegressor,
    RandomForestClassifier, HistGradientBoostingClassifier,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)
from sklearn.utils.class_weight import compute_sample_weight

from feature_extractor_v2 import FEATURE_COLUMNS


V3_DATA_PATH = Path("data/processed/stockfish_10k_v3.csv")
V4_DATA_PATH = Path("data/processed/stockfish_10k_v4.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

RANDOM_STATE = 42
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.1764706
CLIP_LIMIT = 2000
N_JOBS = 2


def parse_bool_column(series, column_name):
    if series.dtype == bool:
        return series
    mapping = {"true": True, "false": False, "1": True, "0": False}
    converted = series.astype(str).str.strip().str.lower().map(mapping)
    if converted.isna().any():
        raise ValueError(f"Could not parse every value in {column_name}.")
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
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
    }


def best_f1_threshold(y_true, probabilities):
    thresholds = np.arange(0.05, 0.96, 0.01)
    scores = [
        f1_score(y_true, probabilities >= threshold, zero_division=0)
        for threshold in thresholds
    ]
    return float(thresholds[int(np.argmax(scores))])


def make_regressors():
    return {
        "random_forest": RandomForestRegressor(
            n_estimators=300, max_depth=20, min_samples_leaf=3,
            max_features=0.9, n_jobs=N_JOBS, random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=True,
            random_state=RANDOM_STATE,
        ),
    }


def make_classifiers():
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=20, min_samples_leaf=3,
            max_features=0.9, class_weight="balanced", n_jobs=N_JOBS,
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=True,
            random_state=RANDOM_STATE,
        ),
    }


def load_aligned_dataset():
    print("\\nLoading V3 and V4 datasets...")
    v3_raw = pd.read_csv(V3_DATA_PATH)
    v4_raw = pd.read_csv(V4_DATA_PATH)

    required_v3 = ["game_id", "ply", "fen", "score_cp", "is_mate"] + list(FEATURE_COLUMNS)
    required_v4 = ["fen", "v4_score_cp", "v4_is_mate", "v4_verified"] + list(FEATURE_COLUMNS)
    missing_v3 = [c for c in required_v3 if c not in v3_raw.columns]
    missing_v4 = [c for c in required_v4 if c not in v4_raw.columns]
    if missing_v3:
        raise ValueError(f"V3 dataset is missing columns: {missing_v3}")
    if missing_v4:
        raise ValueError(f"V4 dataset is missing columns: {missing_v4}")

    v3_raw["is_mate"] = parse_bool_column(v3_raw["is_mate"], "is_mate")
    v4_raw["v4_is_mate"] = parse_bool_column(v4_raw["v4_is_mate"], "v4_is_mate")
    v4_raw["v4_verified"] = parse_bool_column(v4_raw["v4_verified"], "v4_verified")
    v3_raw["score_cp"] = pd.to_numeric(v3_raw["score_cp"], errors="coerce")
    v4_raw["v4_score_cp"] = pd.to_numeric(v4_raw["v4_score_cp"], errors="coerce")

    if v3_raw["fen"].isna().any() or v4_raw["fen"].isna().any():
        raise ValueError("A dataset contains missing FEN values.")

    if (v3_raw.groupby("fen")["is_mate"].nunique() > 1).any():
        raise ValueError("V3 contains conflicting mate labels for a FEN.")
    if (v4_raw.groupby("fen")["v4_is_mate"].nunique() > 1).any():
        raise ValueError("V4 contains conflicting mate labels for a FEN.")

    # Preserve V3's original representative game_id and row order.
    v3_unique = v3_raw.drop_duplicates("fen", keep="first").copy()
    v3_labels = v3_raw.groupby("fen", sort=False).agg(
        v3_score_cp=("score_cp", "median"),
        v3_is_mate=("is_mate", "first"),
    )
    v3_unique = v3_unique.drop(columns=["score_cp", "is_mate"]).merge(
        v3_labels, left_on="fen", right_index=True, validate="one_to_one"
    )

    v4_labels = v4_raw.groupby("fen", sort=False).agg(
        v4_score_cp=("v4_score_cp", "median"),
        v4_is_mate=("v4_is_mate", "first"),
        v4_verified=("v4_verified", "max"),
    ).reset_index()

    df = v3_unique.merge(v4_labels, on="fen", how="inner", validate="one_to_one")
    if len(df) != len(v3_unique):
        raise ValueError("V3 and V4 do not contain the same unique FEN set.")

    for prefix in ("v3", "v4"):
        mate_col, score_col = f"{prefix}_is_mate", f"{prefix}_score_cp"
        if df.loc[~df[mate_col], score_col].isna().any():
            raise ValueError(f"{prefix.upper()} CP positions have missing labels.")
        if df.loc[df[mate_col], score_col].notna().any():
            raise ValueError(f"{prefix.upper()} mate positions have CP labels.")

    X = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        raise ValueError(f"Invalid feature values: {X.columns[X.isna().any()].tolist()}")
    df.loc[:, FEATURE_COLUMNS] = X

    print(f"V3 rows: {len(v3_raw):,}")
    print(f"V4 rows: {len(v4_raw):,}")
    print(f"Aligned unique FENs: {len(df):,}")
    print(f"V3 mates: {int(df['v3_is_mate'].sum()):,}")
    print(f"V4 mates: {int(df['v4_is_mate'].sum()):,}")
    print(f"Audited unique FENs: {int(df['v4_verified'].sum()):,}")
    return df


def create_splits(df):
    print("\\nCreating shared game-based splits...")
    indices = np.arange(len(df))
    groups = df["game_id"].to_numpy()

    outer = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_val_idx, test_idx = next(outer.split(indices, groups=groups))

    inner = GroupShuffleSplit(
        n_splits=1, test_size=VALIDATION_SIZE, random_state=RANDOM_STATE
    )
    train_rel, val_rel = next(
        inner.split(train_val_idx, groups=groups[train_val_idx])
    )

    splits = {
        "train": df.iloc[train_val_idx[train_rel]].copy(),
        "validation": df.iloc[train_val_idx[val_rel]].copy(),
        "test": df.iloc[test_idx].copy(),
    }

    game_sets = {k: set(v["game_id"]) for k, v in splits.items()}
    fen_sets = {k: set(v["fen"]) for k, v in splits.items()}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert game_sets[a].isdisjoint(game_sets[b])
        assert fen_sets[a].isdisjoint(fen_sets[b])

    for name, part in splits.items():
        print(
            f"{name.title()}: {len(part):,} positions | "
            f"{part['game_id'].nunique():,} games | "
            f"V3 mates {int(part['v3_is_mate'].sum()):,} | "
            f"V4 mates {int(part['v4_is_mate'].sum()):,} | "
            f"audited {int(part['v4_verified'].sum()):,}"
        )
    return splits


def train_regression_comparison(splits):
    print("\\n" + "=" * 58 + "\\nV3 VS V4 CENTIPAWN REGRESSION\\n" + "=" * 58)
    train, val, test = splits["train"], splits["validation"], splits["test"]

    # Compare on positions labelled CP by both versions, against V4 targets.
    common_val = val[(~val["v3_is_mate"]) & (~val["v4_is_mate"])]
    common_test = test[(~test["v3_is_mate"]) & (~test["v4_is_mate"])]
    if common_val.empty or common_test.empty:
        raise ValueError("No common V3/V4 CP validation or test positions.")

    X_val, X_test = common_val[FEATURE_COLUMNS], common_test[FEATURE_COLUMNS]
    y_val = common_val["v4_score_cp"].to_numpy()
    y_test = common_test["v4_score_cp"].to_numpy()

    results, test_predictions = {}, {}
    for version in ("v3", "v4"):
        print(f"\\nTraining {version.upper()} regressors...")
        train_cp = train[~train[f"{version}_is_mate"]]
        if train_cp.empty:
            raise ValueError(f"{version.upper()} has no CP training positions.")
        X_train = train_cp[FEATURE_COLUMNS]
        y_train = train_cp[f"{version}_score_cp"].to_numpy()
        candidates = {}

        for target_mode in ("raw", "clipped"):
            target = y_train if target_mode == "raw" else np.clip(
                y_train, -CLIP_LIMIT, CLIP_LIMIT
            )
            for model_name, model in make_regressors().items():
                name = f"{model_name}_{target_mode}"
                print(f"  Fitting {name}...")
                model.fit(X_train, target)
                metrics = regression_metrics(y_val, model.predict(X_val))
                candidates[name] = {"model": model, "validation_metrics": metrics}
                print(f"    V4 validation MAE: {metrics['mae']:.2f} cp")

        best_name = min(
            candidates, key=lambda n: candidates[n]["validation_metrics"]["mae"]
        )
        best_model = candidates[best_name]["model"]
        predictions = best_model.predict(X_test)
        metrics = regression_metrics(y_test, predictions)
        results[version] = {
            "selected_model": best_name,
            "validation_metrics": candidates[best_name]["validation_metrics"],
            "test_metrics_on_common_v4_targets": metrics,
            "train_cp_positions": int(len(train_cp)),
        }
        test_predictions[version] = predictions
        print(
            f"{version.upper()} selected {best_name}: "
            f"MAE {metrics['mae']:.2f} | RMSE {metrics['rmse']:.2f} | "
            f"R2 {metrics['r2']:.4f}"
        )
        if version == "v4":
            joblib.dump(best_model, MODEL_DIR / "stockfish_regressor_v4.joblib")

    pred_df = pd.DataFrame({
        "fen": common_test["fen"].to_numpy(),
        "game_id": common_test["game_id"].to_numpy(),
        "v4_verified": common_test["v4_verified"].to_numpy(),
        "actual_v4_score_cp": y_test,
        "v3_model_prediction": test_predictions["v3"],
        "v4_model_prediction": test_predictions["v4"],
    })
    pred_df["v3_absolute_error"] = np.abs(
        pred_df["actual_v4_score_cp"] - pred_df["v3_model_prediction"]
    )
    pred_df["v4_absolute_error"] = np.abs(
        pred_df["actual_v4_score_cp"] - pred_df["v4_model_prediction"]
    )
    pred_df.to_csv(REPORT_DIR / "v3_v4_regression_test_comparison.csv", index=False)

    subgroup = {}
    for verified, part in pred_df.groupby("v4_verified"):
        subgroup["audited" if verified else "unaudited"] = {
            "positions": int(len(part)),
            "v3_mae": float(part["v3_absolute_error"].mean()),
            "v4_mae": float(part["v4_absolute_error"].mean()),
        }
    results["common_test_positions"] = int(len(common_test))
    results["test_performance_by_audit_status"] = subgroup
    return results


def train_classifier_comparison(splits):
    print("\\n" + "=" * 58 + "\\nV3 VS V4 MATE CLASSIFICATION\\n" + "=" * 58)
    train, val, test = splits["train"], splits["validation"], splits["test"]
    X_val, X_test = val[FEATURE_COLUMNS], test[FEATURE_COLUMNS]
    y_val = val["v4_is_mate"].astype(int).to_numpy()
    y_test = test["v4_is_mate"].astype(int).to_numpy()

    if len(np.unique(y_val)) < 2 or len(np.unique(y_test)) < 2:
        raise ValueError("V4 validation/test splits must contain both mate classes.")

    results, predictions_by_version = {}, {}
    for version in ("v3", "v4"):
        print(f"\\nTraining {version.upper()} classifiers...")
        y_train = train[f"{version}_is_mate"].astype(int).to_numpy()
        X_train = train[FEATURE_COLUMNS]
        if len(np.unique(y_train)) < 2:
            raise ValueError(f"{version.upper()} training split needs both classes.")

        candidates = {}
        for name, model in make_classifiers().items():
            print(f"  Fitting {name}...")
            if name == "hist_gradient_boosting":
                weights = compute_sample_weight("balanced", y_train)
                model.fit(X_train, y_train, sample_weight=weights)
            else:
                model.fit(X_train, y_train)

            probabilities = model.predict_proba(X_val)[:, 1]
            threshold = best_f1_threshold(y_val, probabilities)
            metrics = classifier_metrics(y_val, probabilities, threshold)
            candidates[name] = {
                "model": model,
                "threshold": threshold,
                "validation_metrics": metrics,
                "validation_ap": metrics["average_precision"],
            }
            print(
                f"    V4 validation AP: {metrics['average_precision']:.4f} | "
                f"F1: {metrics['f1']:.4f}"
            )

        best_name = max(candidates, key=lambda n: candidates[n]["validation_ap"])
        best = candidates[best_name]
        probabilities = best["model"].predict_proba(X_test)[:, 1]
        metrics = classifier_metrics(y_test, probabilities, best["threshold"])
        results[version] = {
            "selected_model": best_name,
            "validation_metrics": best["validation_metrics"],
            "test_metrics_on_v4_labels": metrics,
        }
        predictions_by_version[version] = {
            "probabilities": probabilities,
            "threshold": best["threshold"],
        }
        print(
            f"{version.upper()} selected {best_name}: "
            f"AP {metrics['average_precision']:.4f} | "
            f"F1 {metrics['f1']:.4f} | Recall {metrics['recall']:.4f}"
        )
        if version == "v4":
            joblib.dump(
                best["model"], MODEL_DIR / "stockfish_mate_classifier_v4.joblib"
            )

    pred_df = pd.DataFrame({
        "fen": test["fen"].to_numpy(),
        "game_id": test["game_id"].to_numpy(),
        "v4_verified": test["v4_verified"].to_numpy(),
        "actual_v4_is_mate": y_test,
        "v3_mate_probability": predictions_by_version["v3"]["probabilities"],
        "v4_mate_probability": predictions_by_version["v4"]["probabilities"],
    })
    for version in ("v3", "v4"):
        pred_df[f"{version}_predicted_mate"] = (
            pred_df[f"{version}_mate_probability"]
            >= predictions_by_version[version]["threshold"]
        ).astype(int)
    pred_df.to_csv(REPORT_DIR / "v3_v4_mate_test_comparison.csv", index=False)

    subgroup = {}
    for verified, part in pred_df.groupby("v4_verified"):
        subgroup["audited" if verified else "unaudited"] = {
            "positions": int(len(part)),
            "v4_mate_positions": int(part["actual_v4_is_mate"].sum()),
        }
    results["test_counts_by_audit_status"] = subgroup
    return results


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_aligned_dataset()
    splits = create_splits(df)
    regression = train_regression_comparison(splits)
    classification = train_classifier_comparison(splits)

    metadata = {
        "v3_dataset": str(V3_DATA_PATH),
        "v4_dataset": str(V4_DATA_PATH),
        "feature_count": len(FEATURE_COLUMNS),
        "features": list(FEATURE_COLUMNS),
        "unique_fen_positions": int(len(df)),
        "random_state": RANDOM_STATE,
        "split_method": (
            "Shared GroupShuffleSplit by representative V3 game_id after "
            "FEN deduplication"
        ),
        "evaluation": "Both versions evaluated against V4 labels on shared test positions",
        "v4_verified_used_as_feature": False,
        "regression": regression,
        "classification": classification,
        "caveat": (
            "V4 is partially audited. Unverified positions retain V3 labels. "
            "Stockfish search labels are not mathematical proof of forced mate."
        ),
    }
    with open(REPORT_DIR / "stockfish_v4_training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("\\n" + "=" * 58)
    print("V4 TRAINING AND COMPARISON COMPLETE")
    print("=" * 58)
    print("V4 regressor: models/stockfish_regressor_v4.joblib")
    print("V4 mate classifier: models/stockfish_mate_classifier_v4.joblib")
    print("Metadata: reports/stockfish_v4_training_metadata.json")
    print("Regression comparison: reports/v3_v4_regression_test_comparison.csv")
    print("Mate comparison: reports/v3_v4_mate_test_comparison.csv")


if __name__ == "__main__":
    main()
