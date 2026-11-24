
import time
import chess
from statistics import mean

from engine import get_best_move
from ml_evaluator import evaluate_ml_white


def benchmark(evaluator, depth=2, runs=3):
    board = chess.Board()

    if evaluator == "ml":
        evaluate_ml_white(board)  # Warm-up

    times = []

    print(f"\nEvaluator: {evaluator}")

    for i in range(runs):
        start = time.perf_counter()
        move = get_best_move(board, depth=depth, evaluator=evaluator)
        elapsed = time.perf_counter() - start

        times.append(elapsed)

        print(
            f"Run {i + 1}: Move={move}, "
            f"Time={elapsed:.3f}s, "
            f"Legal={move in board.legal_moves}"
        )

    print(f"Average: {mean(times):.3f}s")


if __name__ == "__main__":
    benchmark("handcrafted", depth=2)
    benchmark("ml", depth=2)
