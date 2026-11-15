import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
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

# Keep this deliberately modest for the user's 16 GB laptop.
MAX_VALIDATION_SAMPLES = 1_000
PERMUTATION_REPEATS = 3


def load_aligned_data():
    v1 = pd.read_csv(V1_DATA)
    v2 = pd.read_csv(V2_DATA)

    v1_cols = KEY_COLUMNS + ["fen", TARGET] + V1_FEATURES
    v2_cols = KEY_COLUMNS + ["fen", TARGET] + V2_FEATURES

    for name, df, required in [
        ("V1", v1, v1_cols),
        ("V2", v2, v2_cols),
    ]:
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"{name} dataset missing columns: {missing}")
        if df[required].isnull().any().any():
            raise ValueError(f"{name} dataset has missing required values.")
        if df.duplicated(KEY_COLUMNS).any():
            raise ValueError(f"{name} has duplicate game_id/ply keys.")

    paired = v1[v1_cols].merge(
        v2[v2_cols],
        on=KEY_COLUMNS,
        how="inner",
        suffixes=("_v1", "_v2"),
        validate="one_to_one",
    )

    if paired.empty:
        raise ValueError("No matching V1/V2 positions found.")

    if (paired["fen_v1"] != paired["fen_v2"]).any():
        raise ValueError("FEN mismatch found for matched game_id/ply positions.")

    v1_features = pd.DataFrame(
        {
            feature: paired[f"{feature}_v1"].to_numpy()
            for feature in V1_FEATURES
        },
        index=paired.index,
    )

    v2_features = pd.DataFrame(
        {
            feature: paired[
                f"{feature}_v2" if feature in V1_FEATURES else feature
            ].to_numpy()
            for feature in V2_FEATURES
        },
        index=paired.index,
    )

    # Same common labels and same aligned positions as the controlled comparison.
    y = paired[f"{TARGET}_v1"].astype(float)

    aligned = pd.DataFrame({
        "game_id": paired["game_id"].to_numpy(),
        "ply": paired["ply"].to_numpy(),
        TARGET: y.to_numpy(),
    })

    for feature in V2_FEATURES:
        aligned[feature] = v2_features[feature].to_numpy()

    if not np.isfinite(aligned[V2_FEATURES + [TARGET]].to_numpy(dtype=float)).all():
        raise ValueError("Aligned V2 data contains non-finite numerical values.")

    print(f"Matched positions: {len(aligned):,}")
    print(f"Matched games: {aligned['game_id'].nunique():,}")
    return aligned


def create_split(df):
    groups = df["game_id"]

    first = GroupShuffleSplit(
        n_splits=1,
        test_size=0.15,
        random_state=RANDOM_STATE,
    )
    train_val_idx, test_idx = next(
        first.split(df, df[TARGET], groups)
    )

    train_val = df.iloc[train_val_idx]
    second = GroupShuffleSplit(
        n_splits=1,
        test_size=0.17647,
        random_state=RANDOM_STATE,
    )
    train_idx, val_idx = next(
        second.split(
            train_val,
            train_val[TARGET],
            train_val["game_id"],
        )
    )

    train_global_idx = train_val_idx[train_idx]
    val_global_idx = train_val_idx[val_idx]

    train_df = df.iloc[train_global_idx].copy()
    val_df = df.iloc[val_global_idx].copy()

    train_games = set(train_df["game_id"])
    val_games = set(val_df["game_id"])
    test_games = set(df.iloc[test_idx]["game_id"])

    assert train_games.isdisjoint(val_games)
    assert train_games.isdisjoint(test_games)
    assert val_games.isdisjoint(test_games)

    print(
        f"Train: {len(train_df):,} positions / "
        f"{len(train_games):,} games"
    )
    print(
        f"Validation: {len(val_df):,} positions / "
        f"{len(val_games):,} games"
    )
    print("Game separation: PASSED")

    return train_df, val_df


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading matched V1/V2 positions...")
    df = load_aligned_data()

    print("\nCreating the same game-based split...")
    train_df, val_df = create_split(df)

    print("\nTraining V2 HistGradientBoosting model...")
    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        early_stopping=True,
        random_state=RANDOM_STATE,
    )
    model.fit(train_df[V2_FEATURES], train_df[TARGET])
    print("Training complete.")

    sample_size = min(MAX_VALIDATION_SAMPLES, len(val_df))
    validation_sample = val_df.sample(
        n=sample_size,
        random_state=RANDOM_STATE,
    )

    X_sample = validation_sample[V2_FEATURES]
    y_sample = validation_sample[TARGET]

    print("\nPERMUTATION IMPORTANCE")
    print("----------------------")
    print(f"Validation sample: {sample_size:,} positions")
    print(f"Repeats per feature: {PERMUTATION_REPEATS}")
    print("Parallel jobs: 1")
    print("Scoring: increase in MAE after shuffling a feature")

    result = permutation_importance(
        model,
        X_sample,
        y_sample,
        scoring="neg_mean_absolute_error",
        n_repeats=PERMUTATION_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    importance = pd.DataFrame({
        "feature": V2_FEATURES,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)

    importance["feature_group"] = importance["feature"].apply(
        lambda name: "New V2 feature" if name not in V1_FEATURES else "Original V1 feature"
    )

    output_csv = REPORT_DIR / "v2_permutation_importance.csv"
    importance.to_csv(output_csv, index=False)

    baseline_mae = float(
        np.mean(np.abs(y_sample.to_numpy() - model.predict(X_sample)))
    )

    new_features = importance[
        importance["feature_group"] == "New V2 feature"
    ]
    original_features = importance[
        importance["feature_group"] == "Original V1 feature"
    ]

    summary = {
        "method": "Permutation importance on a 1,000-position validation sample",
        "scoring": "increase in MAE after feature permutation",
        "validation_sample_size": int(sample_size),
        "repeats": PERMUTATION_REPEATS,
        "baseline_sample_mae_cp": baseline_mae,
        "top_10_features": importance.head(10).to_dict(orient="records"),
        "new_v2_features_positive_importance": int(
            (new_features["importance_mean"] > 0).sum()
        ),
        "new_v2_features_total": int(len(new_features)),
        "new_v2_features_mean_importance": float(
            new_features["importance_mean"].mean()
        ),
        "original_features_mean_importance": float(
            original_features["importance_mean"].mean()
        ),
        "importance_csv": str(output_csv),
    }

    output_json = REPORT_DIR / "v2_feature_importance_summary.json"
    with open(output_json, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    top = importance.head(20).sort_values("importance_mean")
    plt.figure(figsize=(10, 8))
    plt.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
    plt.axvline(0, linewidth=0.8)
    plt.xlabel("Increase in MAE when feature is shuffled (centipawns)")
    plt.ylabel("Feature")
    plt.title("V2 Feature Importance (Permutation, Validation Sample)")
    plt.tight_layout()

    output_plot = REPORT_DIR / "v2_permutation_importance.png"
    plt.savefig(output_plot, dpi=160)
    plt.close()

    print("\nTOP 20 FEATURES")
    print("---------------")
    print(
        importance.head(20).to_string(
            index=False,
            formatters={
                "importance_mean": "{:.3f}".format,
                "importance_std": "{:.3f}".format,
            },
        )
    )

    print("\nNEW V2 FEATURE SUMMARY")
    print("----------------------")
    print(
        f"New features with positive importance: "
        f"{summary['new_v2_features_positive_importance']}/"
        f"{summary['new_v2_features_total']}"
    )
    print(
        "Mean importance of new V2 features: "
        f"{summary['new_v2_features_mean_importance']:.3f} cp"
    )
    print(
        "Mean importance of original features: "
        f"{summary['original_features_mean_importance']:.3f} cp"
    )

    print("\nSAVED REPORTS")
    print("-------------")
    print(output_csv)
    print(output_json)
    print(output_plot)


if __name__ == "__main__":
    main()
