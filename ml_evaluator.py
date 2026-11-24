"""Inference wrapper for the trained V2 Stockfish evaluation model.

The model predicts centipawns from the side-to-move perspective.
The search engine uses White's perspective for static evaluations, so
this module converts the prediction before returning it.
"""
from functools import lru_cache
from pathlib import Path
import pandas as pd
import joblib


from ml.feature_extractor_v2 import FEATURE_COLUMNS, extract_features

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "stockfish_regressor_v2.joblib"


@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"V2 ML model not found: {MODEL_PATH}. "
            "Place stockfish_regressor_v2.joblib in the models directory."
        )

    model = joblib.load(MODEL_PATH)

    expected = list(getattr(model, "feature_names_in_", []))
    if expected and expected != FEATURE_COLUMNS:
        raise ValueError(
            "Model feature order does not match feature_extractor_v2.py. "
            "Do not run inference until the training and inference schemas match."
        )

    feature_count = getattr(model, "n_features_in_", None)
    if feature_count is not None and feature_count != len(FEATURE_COLUMNS):
        raise ValueError(
            f"Model expects {feature_count} features, "
            f"but extractor provides {len(FEATURE_COLUMNS)}."
        )

    return model



def evaluate_ml_white(board):
    """Return ML evaluation in centipawns from White's perspective."""
    if board.is_checkmate():
        return -100_000 if board.turn else 100_000

    if board.is_game_over(claim_draw=True):
        return 0

    features = extract_features(board)
    row = pd.DataFrame(
        [[features[name] for name in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
    )

    score_stm = float(_load_model().predict(row)[0])
    score_white = score_stm if board.turn else -score_stm

    return int(round(score_white))

