import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml.feature_extractor_v2 import FEATURE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "stockfish_10k_v2.csv"
MODEL_DIR = PROJECT_ROOT / "models"
RANDOM_STATE = 42
TARGET = "score_cp"


def evaluate_model(model, X, y, name):
    predictions = model.predict(X)
    return {
        "name": name,
        "mae": float(mean_absolute_error(y, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y, predictions))),
        "r2": float(r2_score(y, predictions)),
        "correlation": float(np.corrcoef(y, predictions)[0, 1]),
    }


def split_dataset(df):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET]
    groups = df["game_id"]

    first_split = GroupShuffleSplit(
        n_splits=1, test_size=0.15, random_state=RANDOM_STATE
    )
    train_val_idx, test_idx = next(first_split.split(X, y, groups))

    X_train_val = X.iloc[train_val_idx]
    y_train_val = y.iloc[train_val_idx]
    groups_train_val = groups.iloc[train_val_idx]

    second_split = GroupShuffleSplit(
        n_splits=1, test_size=0.17647, random_state=RANDOM_STATE
    )
    train_idx, val_idx = next(
        second_split.split(X_train_val, y_train_val, groups_train_val)
    )

    train_global_idx = train_val_idx[train_idx]
    val_global_idx = train_val_idx[val_idx]

    X_train, y_train = X.iloc[train_global_idx], y.iloc[train_global_idx]
    X_val, y_val = X.iloc[val_global_idx], y.iloc[val_global_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    train_games = set(groups.iloc[train_global_idx])
    val_games = set(groups.iloc[val_global_idx])
    test_games = set(groups.iloc[test_idx])

    assert train_games.isdisjoint(val_games)
    assert train_games.isdisjoint(test_games)
    assert val_games.isdisjoint(test_games)

    print("\nDATASET SPLIT")
    print("-------------")
    for name, X_part, y_part, games in [
        ("Train", X_train, y_train, train_games),
        ("Validation", X_val, y_val, val_games),
        ("Test", X_test, y_test, test_games),
    ]:
        print(
            f"{name}: {len(X_part):,} positions | "
            f"{len(games):,} games | mean score: {y_part.mean():.2f} cp"
        )
    print("Game overlap check: PASSED")

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_models(X_train, y_train):
    models = {
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

    trained = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        trained[name] = model
        print(f"{name} training complete.")
    return trained


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading Stockfish V2 dataset...")
    df = pd.read_csv(DATA_FILE)

    required_columns = ["game_id", "fen", TARGET] + FEATURE_COLUMNS
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df[required_columns].isnull().any().any():
        raise ValueError("Dataset contains missing values.")

    numeric_data = df[FEATURE_COLUMNS + [TARGET]].to_numpy(dtype=float)
    if not np.isfinite(numeric_data).all():
        raise ValueError("Dataset contains infinite or non-finite numerical values.")

    print(f"Loaded {len(df):,} positions.")
    print(f"Using {len(FEATURE_COLUMNS)} V2 features.")

    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(df)
    models = train_models(X_train, y_train)

    print("\nVALIDATION RESULTS")
    print("------------------")
    validation_results = {}
    for name, model in models.items():
        result = evaluate_model(model, X_val, y_val, name)
        validation_results[name] = result
        print(
            f"{name}: MAE={result['mae']:.2f} cp | "
            f"RMSE={result['rmse']:.2f} cp | "
            f"R2={result['r2']:.4f} | "
            f"Correlation={result['correlation']:.4f}"
        )

    best_name = min(
        validation_results, key=lambda name: validation_results[name]["mae"]
    )
    best_model = models[best_name]
    print(f"\nSelected model: {best_name}")

    # The test set is held out until model selection is complete.
    test_result = evaluate_model(best_model, X_test, y_test, best_name)

    print("\nFINAL TEST RESULTS")
    print("------------------")
    print(f"MAE: {test_result['mae']:.2f} cp")
    print(f"RMSE: {test_result['rmse']:.2f} cp")
    print(f"R2: {test_result['r2']:.4f}")
    print(f"Correlation: {test_result['correlation']:.4f}")

    # V2-specific filenames preserve all V1 model artifacts.
    model_filenames = {
        "random_forest": "random_forest_stockfish_v2.joblib",
        "hist_gradient_boosting": "hist_gradient_boosting_stockfish_v2.joblib",
    }
    for name, model in models.items():
        model_path = MODEL_DIR / model_filenames[name]
        joblib.dump(model, model_path)
        print(f"Saved {name}: {model_path}")

    best_model_path = MODEL_DIR / "stockfish_regressor_v2.joblib"
    joblib.dump(best_model, best_model_path)

    metadata = {
        "dataset": str(DATA_FILE),
        "feature_version": "v2",
        "target": TARGET,
        "features": FEATURE_COLUMNS,
        "random_state": RANDOM_STATE,
        "selected_model": best_name,
        "validation_results": validation_results,
        "test_results": test_result,
        "train_positions": len(X_train),
        "validation_positions": len(X_val),
        "test_positions": len(X_test),
        "score_units": "centipawns from side-to-move perspective",
        "model_files": {
            **model_filenames,
            "selected_model": best_model_path.name,
        },
    }
    metadata_path = MODEL_DIR / "stockfish_training_metadata_v2.json"
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("\nV2 TRAINING COMPLETE")
    print("--------------------")
    print(f"Selected model: {best_name}")
    print(f"Saved model: {best_model_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
