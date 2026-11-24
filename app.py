
from flask import Flask, jsonify, render_template, request
import chess

from engine import get_best_move, evaluate_board
from ml_evaluator import evaluate_ml_white

app = Flask(__name__)


def read_board(fen, history=None):
    try:
        # Without history, retain FEN-only support for endpoints
        # such as legal move generation.
        if history is None:
            board = chess.Board(fen) if fen else chess.Board()
        else:
            if not isinstance(history, list):
                return None

            board = chess.Board()

            # Replay every move to preserve repetition history.
            for uci in history:
                if not isinstance(uci, str):
                    return None

                move = chess.Move.from_uci(uci)

                if move not in board.legal_moves:
                    return None

                board.push(move)

            # Ensure the submitted history matches the submitted position.
            if fen and board.fen() != fen:
                return None

    except (ValueError, TypeError):
        return None

    return board if board.is_valid() else None


def game_status(board):
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        return {"over": True, "message": f"Checkmate. {winner} wins."}

    if board.is_stalemate():
        return {"over": True, "message": "Draw by stalemate."}

    if board.is_insufficient_material():
        return {"over": True, "message": "Draw by insufficient material."}

    if board.is_game_over(claim_draw=True):
        return {"over": True, "message": "Game drawn."}

    turn = "White" if board.turn == chess.WHITE else "Black"
    message = f"{turn} to move"

    if board.is_check():
        message += " · Check"

    return {"over": False, "message": message}


def evaluate_position(board, evaluator):
    """Return the position evaluation in centipawns from White's perspective."""
    if evaluator == "ml":
        return evaluate_ml_white(board)

    return evaluate_board(board)


def payload(board, moves=None, bot_move=None, evaluator="handcrafted"):
    return {
        "fen": board.fen(),
        "turn": "white" if board.turn == chess.WHITE else "black",
        "status": game_status(board),
        "moves": moves or [],
        "bot_move": bot_move,
        "legal_moves": [m.uci() for m in board.legal_moves],
        "evaluation": evaluate_position(board, evaluator),
        "evaluator": evaluator,
    }


def read_settings(data):
    player_color = data.get("player_color", "white")
    depth = int(data.get("depth", 3))
    evaluator = data.get("evaluator", "handcrafted")

    if player_color not in ("white", "black"):
        raise ValueError("Invalid player colour.")

    if not 1 <= depth <= 5:
        raise ValueError("Depth must be between 1 and 5.")

    if evaluator not in ("handcrafted", "ml"):
        raise ValueError("Evaluator must be handcrafted or ml.")

    return player_color, depth, evaluator


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({
        "ok": True,
        "engine": "ChessAI Minimax + Alpha-Beta",
        "default_depth": 3,
        "evaluators": ["handcrafted", "ml"],
    })


@app.post("/api/new-game")
def new_game():
    data = request.get_json(silent=True) or {}

    try:
        player_color, depth, evaluator = read_settings(data)
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid game settings."}), 400

    board = chess.Board()
    moves = []
    bot_move = None

    if player_color == "black":
        move = get_best_move(board, depth, evaluator)

        if move:
            bot_san = board.san(move)
            board.push(move)
            bot_move = move.uci()
            moves.append({"uci": bot_move, "san": bot_san})

    result = payload(board, moves, bot_move, evaluator)
    result["player_color"] = player_color
    result["depth"] = depth

    return jsonify(result)


@app.post("/api/legal-moves")
def legal_moves():
    data = request.get_json(silent=True) or {}
    board = read_board(data.get("fen"))

    if board is None:
        return jsonify({"error": "Invalid chess position."}), 400

    return jsonify({
        "legal_moves": [m.uci() for m in board.legal_moves]
    })


@app.post("/api/move")
def make_move():
    data = request.get_json(silent=True) or {}
    board = read_board(data.get("fen"), data.get("history"))

    try:
        player_color, depth, evaluator = read_settings(data)
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid game settings."}), 400

    uci = data.get("move", "")

    if board is None:
        return jsonify({"error": "Invalid chess position."}), 400

    if board.is_game_over(claim_draw=True):
        return jsonify({
            "error": "Game is already over.",
            "status": game_status(board)
        }), 400

    human_side = chess.WHITE if player_color == "white" else chess.BLACK

    if board.turn != human_side:
        return jsonify({"error": "It is not your turn."}), 400

    try:
        move = chess.Move.from_uci(uci)
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid move notation."}), 400

    if move not in board.legal_moves:
        return jsonify({"error": "That move is not legal."}), 400

    human_san = board.san(move)
    board.push(move)

    played = [{"uci": move.uci(), "san": human_san}]
    bot_move = None

    if not board.is_game_over(claim_draw=True):
        reply = get_best_move(board, depth, evaluator)

        if reply:
            bot_san = board.san(reply)
            board.push(reply)
            bot_move = reply.uci()
            played.append({"uci": bot_move, "san": bot_san})

    result = payload(board, played, bot_move, evaluator)
    result["player_color"] = player_color
    result["depth"] = depth

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
