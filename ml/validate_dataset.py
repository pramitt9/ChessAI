
import chess.pgn
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PGN_FILE = PROJECT_ROOT / "data" / "raw" / "sample_100k.pgn"

def validate_dataset():
    counts = Counter()
    total_plies = 0
    errors = 0

    with open(PGN_FILE, "r", encoding="utf-8") as pgn:
        while True:
            game = chess.pgn.read_game(pgn)

            if game is None:
                break

            result = game.headers.get("Result", "*")
            counts[result] += 1

            total_plies += game.end().ply()
            errors += len(game.errors)

    total_games = sum(counts.values())

    print("\nDATASET VALIDATION")
    print("------------------")
    print(f"Total games: {total_games:,}")
    print(f"White wins: {counts['1-0']:,}")
    print(f"Black wins: {counts['0-1']:,}")
    print(f"Draws: {counts['1/2-1/2']:,}")
    print(f"Unfinished: {counts['*']:,}")
    print(f"Total plies: {total_plies:,}")
    print(f"Parsing errors: {errors:,}")

    if total_games != 100_000:
        print("WARNING: Expected exactly 100,000 games.")

    if errors:
        print("WARNING: Some games contain parsing errors.")

if __name__ == "__main__":
    validate_dataset()
