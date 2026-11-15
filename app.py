from flask import Flask, jsonify, render_template, request
import chess

from engine import get_best_move

app = Flask(__name__)


def read_board(fen):
    try:
        board = chess.Board(fen) if fen else chess.Board()
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


def payload(board, moves=None, bot_move=None):
    return {
        "fen": board.fen(),
        "turn": "white" if board.turn == chess.WHITE else "black",
        "status": game_status(board),
        "moves": moves or [],
        "bot_move": bot_move,
        "legal_moves": [m.uci() for m in board.legal_moves],
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "engine": "ChessAI Minimax + Alpha-Beta", "default_depth": 3})


@app.post("/api/new-game")
def new_game():
    data = request.get_json(silent=True) or {}
    player_color = data.get("player_color", "white")
    depth = int(data.get("depth", 3))
    if player_color not in ("white", "black") or not 1 <= depth <= 5:
        return jsonify({"error": "Invalid colour or search depth."}), 400

    board = chess.Board()
    moves = []
    bot_move = None
    if player_color == "black":
        move = get_best_move(board, depth)
        if move:
            bot_san = board.san(move)
            board.push(move)
            bot_move = move.uci()
            moves.append({"uci": bot_move, "san": bot_san})
    result = payload(board, moves, bot_move)
    result["player_color"] = player_color
    result["depth"] = depth
    return jsonify(result)


@app.post("/api/legal-moves")
def legal_moves():
    data = request.get_json(silent=True) or {}
    board = read_board(data.get("fen"))
    if board is None:
        return jsonify({"error": "Invalid chess position."}), 400
    return jsonify({"legal_moves": [m.uci() for m in board.legal_moves]})


@app.post("/api/move")
def make_move():
    data = request.get_json(silent=True) or {}
    board = read_board(data.get("fen"))
    player_color = data.get("player_color", "white")
    depth = int(data.get("depth", 3))
    uci = data.get("move", "")

    if board is None or player_color not in ("white", "black") or not 1 <= depth <= 5:
        return jsonify({"error": "Invalid game state or settings."}), 400

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
        reply = get_best_move(board, depth)
        if reply:
            bot_san = board.san(reply)
            board.push(reply)
            bot_move = reply.uci()
            played.append({"uci": bot_move, "san": bot_san})

    result = payload(board, played, bot_move)
    result["player_color"] = player_color
    result["depth"] = depth
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
