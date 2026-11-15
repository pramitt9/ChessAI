# ChessAI

A playable chess application with a Python chess engine using Minimax and Alpha-Beta pruning.

## Features

- Play as White or Black against the engine.
- Select search depth (1–5); default is 3.
- Legal move validation, castling, en passant, promotion and game-ending detection through `python-chess`.
- Browser-based interactive board built with HTML, CSS and vanilla JavaScript.
- Engine evaluation from White's perspective using material values.

## Setup (Windows PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000 in your browser.

If you already have a virtual environment, activate it and run `pip install -r requirements.txt`.

## Tests

```powershell
python -m pytest tests
```

Or run the basic test directly:

```powershell
python tests/test_engine.py
```

## Current engine limitations

This is the first playable baseline. Evaluation currently counts material only. Search depth is capped at 5 to avoid long blocking requests. Positional evaluation, iterative deepening, transposition tables, clocks and stronger move ordering are future improvements.
