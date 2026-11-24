
import chess
import chess.pgn
import time

from engine import get_best_move

DEPTH = 1
MAX_PLIES = 120

MATCHES = [
    {"white": "ml", "black": "handcrafted"},
    {"white": "handcrafted", "black": "ml"},
]


def play_game(game_number, white_eval, black_eval):
    board = chess.Board()
    game = chess.pgn.Game()

    game.headers["Event"] = "ChessAI Evaluator Tournament"
    game.headers["Round"] = str(game_number)
    game.headers["White"] = white_eval
    game.headers["Black"] = black_eval

    node = game
    start_time = time.perf_counter()
    move_times = {"ml": [], "handcrafted": []}

    print(f"\nGame {game_number}:")
    print(f"White: {white_eval} | Black: {black_eval}")

    while not board.is_game_over(claim_draw=True):
        if board.ply() >= MAX_PLIES:
            break

        evaluator = white_eval if board.turn == chess.WHITE else black_eval

        move_start = time.perf_counter()
        move = get_best_move(board, depth=DEPTH, evaluator=evaluator)
        elapsed = time.perf_counter() - move_start

        if move is None or move not in board.legal_moves:
            raise RuntimeError(
                f"{evaluator} returned an invalid move: {move}"
            )

        move_times[evaluator].append(elapsed)

        move_number = board.fullmove_number
        prefix = "." if board.turn == chess.WHITE else "..."

        san = board.san(move)
        board.push(move)
        node = node.add_variation(move)

        print(
                f"{move_number}{prefix} {san} "
                f"[{evaluator}, {elapsed:.3f}s]"
            )

    outcome = board.outcome(claim_draw=True)
    result = outcome.result() if outcome else "1/2-1/2"

    game.headers["Result"] = result
    game.headers["Termination"] = (
        outcome.termination.name if outcome else "Move limit"
    )

    total_time = time.perf_counter() - start_time

    print(f"\nResult: {result}")
    print(f"Termination: {game.headers['Termination']}")
    print(f"Total time: {total_time:.2f}s")

    for evaluator in ("ml", "handcrafted"):
        times = move_times[evaluator]
        if times:
            print(
                f"{evaluator}: {len(times)} moves, "
                f"average {sum(times) / len(times):.3f}s/move"
            )

    return game, result


def main():
    scores = {"ml": 0, "handcrafted": 0, "draw": 0}

    for i, match in enumerate(MATCHES, start=1):
        game, result = play_game(
            i, match["white"], match["black"]
        )

        if result == "1-0":
            scores[match["white"]] += 1
        elif result == "0-1":
            scores[match["black"]] += 1
        else:
            scores["draw"] += 1

        filename = f"tournament_game_{i}.pgn"
        with open(filename, "w", encoding="utf-8") as file:
            print(game, file=file)

        print(f"Saved: {filename}")

    print("\n========== TOURNAMENT SUMMARY ==========")
    print(f"ML points: {scores['ml']}")
    print(f"Handcrafted points: {scores['handcrafted']}")
    print(f"Draws: {scores['draw']}")


if __name__ == "__main__":
    main()
