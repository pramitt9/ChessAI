
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    classification_report,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier


# ==================================================
# CONFIGURATION
# ==================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "processed" / "positions.csv"
MODEL_DIR = ROOT_DIR / "models"

RANDOM_STATE = 42
CLASSES = [-1, 0, 1]

FEATURE_COLUMNS = [
    "own_pawns",
    "own_knights",
    "own_bishops",
    "own_rooks",
    "own_queens",
    "opp_pawns",
    "opp_knights",
    "opp_bishops",
    "opp_rooks",
    "opp_queens",
    "material_balance",
    "own_mobility",
    "opp_mobility",
    "mobility_balance",
    "own_doubled_pawns",
    "opp_doubled_pawns",
    "own_isolated_pawns",
    "opp_isolated_pawns",
    "own_passed_pawns",
    "opp_passed_pawns",
    "own_bishop_pair",
    "opp_bishop_pair",
    "own_castling_rights",
    "opp_castling_rights",
    "own_king_shield",
    "opp_king_shield",
    "center_control_balance",
    "own_developed_minors",
    "opp_developed_minors",
    "game_phase",
    "in_check",
]


# ==================================================
# DATA LOADING
# ==================================================

def load_dataset():
    print("\nLoading dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = FEATURE_COLUMNS + [
        "game_id",
        "label",
    ]

    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if df[FEATURE_COLUMNS].isnull().any().any():
        raise ValueError(
            "Feature data contains missing values."
        )

    if not np.isfinite(
        df[FEATURE_COLUMNS].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Feature data contains infinite values."
        )

    if not set(df["label"].unique()).issubset(set(CLASSES)):
        raise ValueError(
            "Unexpected label values found."
        )

    X = df[FEATURE_COLUMNS]
    y = df["label"].astype(int)
    groups = df["game_id"]

    print(f"Positions: {len(df):,}")
    print(f"Games: {groups.nunique():,}")
    print(f"Features: {len(FEATURE_COLUMNS)}")

    print("\nOverall label distribution:")
    print(
        (y.value_counts(normalize=True)
         .reindex(CLASSES, fill_value=0) * 100)
        .round(2)
        .to_string()
    )

    return X, y, groups


# ==================================================
# GAME-BASED DATA SPLITTING
# ==================================================

def split_dataset(X, y, groups):

    # First split: 70% training, 30% temporary.
    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=RANDOM_STATE,
    )

    train_idx, temp_idx = next(
        first_split.split(X, y, groups)
    )

    # Second split: 15% validation, 15% testing.
    temp_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=RANDOM_STATE,
    )

    val_relative, test_relative = next(
        temp_split.split(
            X.iloc[temp_idx],
            y.iloc[temp_idx],
            groups.iloc[temp_idx],
        )
    )

    val_idx = temp_idx[val_relative]
    test_idx = temp_idx[test_relative]

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    X_val = X.iloc[val_idx]
    y_val = y.iloc[val_idx]

    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    # Verify that no game appears in multiple splits.
    train_games = set(groups.iloc[train_idx])
    val_games = set(groups.iloc[val_idx])
    test_games = set(groups.iloc[test_idx])

    assert train_games.isdisjoint(val_games)
    assert train_games.isdisjoint(test_games)
    assert val_games.isdisjoint(test_games)

    print("\nDATASET SPLIT")

    splits = {
        "Training": (X_train, y_train, groups.iloc[train_idx]),
        "Validation": (X_val, y_val, groups.iloc[val_idx]),
        "Testing": (X_test, y_test, groups.iloc[test_idx]),
    }

    for name, (split_X, split_y, split_groups) in splits.items():

        print(f"\n{name}:")
        print(f"  Positions: {len(split_X):,}")
        print(f"  Games: {split_groups.nunique():,}")

        print("  Label distribution (%):")

        print(
            (split_y.value_counts(normalize=True)
             .reindex(CLASSES, fill_value=0) * 100)
            .round(2)
            .to_string()
        )

        if set(split_y.unique()) != set(CLASSES):
            raise ValueError(
                f"{name} split does not contain all three classes."
            )

    print("\nGame overlap check: PASSED")

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )


# ==================================================
# MODEL DEFINITIONS
# ==================================================

def create_models():

    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=20,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),

        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            early_stopping=True,
            random_state=RANDOM_STATE,
        ),

        "XGBoost": XGBClassifier(
            n_estimators=250,
            max_depth=8,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.9,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


# ==================================================
# PROBABILITY HANDLING
# ==================================================

def get_probabilities_and_predictions(
    name,
    model,
    X,
):

    raw_probabilities = model.predict_proba(X)

    if name == "XGBoost":

        probabilities = raw_probabilities[:, [0, 1, 2]]

        predictions = np.array(CLASSES)[
            np.argmax(probabilities, axis=1)
        ]

    else:

        model_classes = list(model.classes_)

        probabilities = raw_probabilities[
            :, [
                model_classes.index(c)
                for c in CLASSES
            ]
        ]

        predictions = model.predict(X)

    return probabilities, predictions


# ==================================================
# TRAINING AND VALIDATION
# ==================================================

def evaluate_model(
    name,
    model,
    X_train,
    y_train,
    X_val,
    y_val,
):

    print("\n" + "=" * 55)
    print(f"TRAINING: {name}")
    print("=" * 55)

    start = time.perf_counter()

    # Apply balanced sample weights consistently.
    weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    )

    if name == "XGBoost":

        # XGBoost uses encoded classes: 0, 1, 2.
        encoded_y = y_train.map({
            -1: 0,
            0: 1,
            1: 2,
        })

        model.fit(
            X_train,
            encoded_y,
            sample_weight=weights,
            verbose=False,
        )

    else:

        model.fit(
            X_train,
            y_train,
            sample_weight=weights,
        )

    training_time = time.perf_counter() - start

    probabilities, predictions = (
        get_probabilities_and_predictions(
            name,
            model,
            X_val,
        )
    )

    accuracy = accuracy_score(
        y_val,
        predictions,
    )

    macro_f1 = f1_score(
        y_val,
        predictions,
        labels=CLASSES,
        average="macro",
    )

    loss = log_loss(
        y_val,
        probabilities,
        labels=CLASSES,
    )

    print(f"\nTraining time: {training_time:.2f} seconds")
    print(f"Accuracy:      {accuracy:.4f}")
    print(f"Macro F1:      {macro_f1:.4f}")
    print(f"Log loss:      {loss:.4f}")

    print("\nClassification report:")

    print(
        classification_report(
            y_val,
            predictions,
            labels=CLASSES,
            zero_division=0,
        )
    )

    print("Confusion matrix:")

    print(
        confusion_matrix(
            y_val,
            predictions,
            labels=CLASSES,
        )
    )

    return {
        "model": model,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "log_loss": loss,
        "training_time": training_time,
    }


# ==================================================
# FINAL TEST EVALUATION
# ==================================================

def evaluate_test_set(
    name,
    model,
    X_test,
    y_test,
):

    print("\n" + "=" * 55)
    print("FINAL TEST EVALUATION")
    print(f"Selected model: {name}")
    print("=" * 55)

    probabilities, predictions = (
        get_probabilities_and_predictions(
            name,
            model,
            X_test,
        )
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        labels=CLASSES,
        average="macro",
    )

    loss = log_loss(
        y_test,
        probabilities,
        labels=CLASSES,
    )

    print(f"\nTest accuracy: {accuracy:.4f}")
    print(f"Test macro F1: {macro_f1:.4f}")
    print(f"Test log loss: {loss:.4f}")

    print("\nTest classification report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=CLASSES,
            zero_division=0,
        )
    )

    print("Test confusion matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=CLASSES,
        )
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "log_loss": loss,
    }


# ==================================================
# MAIN PIPELINE
# ==================================================

def main():

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Load data.
    X, y, groups = load_dataset()

    # Split by complete games.
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = split_dataset(X, y, groups)

    # Create models.
    models = create_models()

    results = {}

    # Train and evaluate each model.
    for name, model in models.items():

        results[name] = evaluate_model(
            name,
            model,
            X_train,
            y_train,
            X_val,
            y_val,
        )

    # Select the best model using validation macro F1.
    best_name = max(
        results,
        key=lambda name: results[name]["macro_f1"],
    )

    best_model = results[best_name]["model"]

    print("\n" + "=" * 55)
    print(f"BEST VALIDATION MODEL: {best_name}")
    print("=" * 55)

    # Evaluate only the selected model on the untouched test set.
    test_results = evaluate_test_set(
        best_name,
        best_model,
        X_test,
        y_test,
    )

    # Save all trained models.
    print("\nSaving trained models...")

    for name, result in results.items():

        filename = (
            name.lower().replace(" ", "_") + ".joblib"
        )

        joblib.dump(
            result["model"],
            MODEL_DIR / filename,
        )

        print(f"Saved: {filename}")

    # Save training metadata.
    metadata = {
        "best_model": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "classes": CLASSES,

        "validation_results": {
            name: {
                key: result[key]
                for key in (
                    "accuracy",
                    "macro_f1",
                    "log_loss",
                    "training_time",
                )
            }
            for name, result in results.items()
        },

        "test_results": test_results,
    }

    with open(
        MODEL_DIR / "training_metadata.json",
        "w",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    print("\n" + "=" * 55)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 55)

    print(f"Best model: {best_name}")
    print(f"Models saved in: {MODEL_DIR}")


if __name__ == "__main__":
    main()
