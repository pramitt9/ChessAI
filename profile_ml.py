
import time
import chess
import pandas as pd

from ml_evaluator import _load_model
from ml.feature_extractor_v2 import FEATURE_COLUMNS, extract_features

board = chess.Board()
model = _load_model()

N = 100

# Warm up
features = extract_features(board)
row = pd.DataFrame(
    [[features[name] for name in FEATURE_COLUMNS]],
    columns=FEATURE_COLUMNS,
)
model.predict(row)

# Feature extraction
start = time.perf_counter()
for _ in range(N):
    features = extract_features(board)
feature_time = (time.perf_counter() - start) / N

# Model inference
start = time.perf_counter()
for _ in range(N):
    model.predict(row)
predict_time = (time.perf_counter() - start) / N

print(f"Feature extraction: {feature_time * 1000:.3f} ms")
print(f"Model prediction:   {predict_time * 1000:.3f} ms")
print(f"Total:              {(feature_time + predict_time) * 1000:.3f} ms")
