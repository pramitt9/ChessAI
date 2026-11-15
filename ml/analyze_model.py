
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupShuffleSplit


# ==================================================
# CONFIGURATION
# ==================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT_DIR / "data" / "processed" / "positions.csv"
MODEL_PATH = ROOT_DIR / "models" / "random_forest.joblib"
OUTPUT_DIR = ROOT_DIR / "reports"

RANDOM_STATE = 42

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
# LOAD MODEL AND DATA
# ==================================================

def load_data_and_model():

    print("Loading dataset and trained model...")

    df = pd.read_csv(DATA_PATH)

    model = joblib.load(MODEL_PATH)

    X = df[FEATURE_COLUMNS]
    y = df["label"].astype(int)
    groups = df["game_id"]

    return X, y, groups, model


# ==================================================
# RECREATE VALIDATION SPLIT
# ==================================================

def get_validation_data(X, y, groups):

    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=RANDOM_STATE,
    )

    train_idx, temp_idx = next(
        first_split.split(X, y, groups)
    )

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

    X_val = X.iloc[val_idx]
    y_val = y.iloc[val_idx]

    print(f"Validation positions: {len(X_val):,}")
    print(f"Validation games: {groups.iloc[val_idx].nunique():,}")

    return X_val, y_val


# ==================================================
# RANDOM FOREST BUILT-IN IMPORTANCE
# ==================================================

def calculate_builtin_importance(model):

    print("\nCalculating built-in feature importance...")

    importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": model.feature_importances_,
    })

    importance = importance.sort_values(
        "importance",
        ascending=False,
    )

    return importance


# ==================================================
# PERMUTATION IMPORTANCE
# ==================================================

def calculate_permutation_importance(model, X_val, y_val):

    print("\nCalculating permutation importance...")

    print("This may take a few minutes.")

    result = permutation_importance(
        model,
        X_val,
        y_val,
        scoring="f1_macro",
        n_repeats=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    })

    importance = importance.sort_values(
        "importance_mean",
        ascending=False,
    )

    return importance


# ==================================================
# SAVE RESULTS
# ==================================================

def save_results(builtin, permutation):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    builtin.to_csv(
        OUTPUT_DIR / "builtin_feature_importance.csv",
        index=False,
    )

    permutation.to_csv(
        OUTPUT_DIR / "permutation_importance.csv",
        index=False,
    )

    print("\nSaved importance reports.")


# ==================================================
# VISUALIZATION
# ==================================================

def plot_importance(builtin, permutation):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    top = permutation.head(15).sort_values(
        "importance_mean",
        ascending=True,
    )

    plt.figure(figsize=(11, 7))

    plt.barh(
        top["feature"],
        top["importance_mean"],
        xerr=top["importance_std"],
    )

    plt.xlabel("Decrease in Macro F1 after permutation")
    plt.ylabel("Chess feature")
    plt.title("Random Forest: Permutation Feature Importance")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "permutation_importance.png",
        dpi=300,
    )

    plt.close()

    print("Saved feature importance plot.")


# ==================================================
# MAIN
# ==================================================

def main():

    X, y, groups, model = load_data_and_model()

    X_val, y_val = get_validation_data(
        X,
        y,
        groups,
    )

    builtin = calculate_builtin_importance(model)

    permutation = calculate_permutation_importance(
        model,
        X_val,
        y_val,
    )

    print("\nTOP 15 BUILT-IN FEATURES")
    print(builtin.head(15).to_string(index=False))

    print("\nTOP 15 PERMUTATION FEATURES")
    print(permutation.head(15).to_string(index=False))

    save_results(
        builtin,
        permutation,
    )

    plot_importance(
        builtin,
        permutation,
    )

    print("\nFEATURE ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
