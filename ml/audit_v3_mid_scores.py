
"""
Audit V3 Stockfish labels in the 1,000–2,000 centipawn range.

Purpose:
    Sample 100 positions whose final V3 centipawn evaluation has an
    absolute value between 1,000 and 2,000 cp.

    Reanalyse each position with Stockfish for 5 seconds and compare
    the new evaluation against the existing V3 label.

Important:
    We deliberately include positions where was_rechecked=True.
    V3 already rechecked many of these positions for 1 second.
    This audit tests whether those labels remain stable with 5 seconds
    of additional analysis.

Input:
    data/processed/stockfish_10k_v3.csv

Output:
    reports/v3_mid_score_audit.csv
"""

import random
from pathlib import Path

import chess
import chess.engine
import pandas as pd


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = ROOT_DIR / "data" / "processed" / "stockfish_10k_v3.csv"

STOCKFISH_PATH = (
    ROOT_DIR / "engines" / "stockfish" / "stockfish.exe"
)

OUTPUT_PATH = ROOT_DIR / "reports" / "v3_mid_score_audit.csv"

SAMPLE_SIZE = 100
AUDIT_TIME = 5.0

MIN_SCORE = 1000
MAX_SCORE = 2000

RANDOM_SEED = 42

THREADS = 2
HASH_MB = 256


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def get_stockfish_evaluation(engine, fen):
    """
    Analyse a FEN position and return Stockfish's evaluation
    from the perspective of the side to move.

    Returns:
        score_cp: Centipawn score, or None for mate.
        mate_in: Mate distance, or None for centipawn score.
        depth: Search depth.
        nodes: Nodes searched.
    """

    board = chess.Board(fen)

    info = engine.analyse(
        board,
        chess.engine.Limit(time=AUDIT_TIME)
    )

    score = info["score"].pov(board.turn)

    score_cp = score.score()
    mate_in = score.mate()

    return {
        "audit_score_cp": score_cp,
        "audit_mate_in": mate_in,
        "audit_depth": info.get("depth"),
        "audit_nodes": info.get("nodes"),
    }


# ---------------------------------------------------------
# Main audit
# ---------------------------------------------------------

def main():

    print("=" * 65)
    print("V3 MID-SCORE LABEL AUDIT")
    print("=" * 65)

    # Validate paths
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    if not STOCKFISH_PATH.exists():
        raise FileNotFoundError(
            f"Stockfish executable not found: {STOCKFISH_PATH}"
        )

    # Load dataset
    print(f"\nLoading dataset: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    required_columns = [
        "fen",
        "score_cp",
        "is_mate",
        "was_rechecked",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    print(f"Total dataset rows: {len(df)}")

    # Select centipawn positions in the target range.
    # Do NOT filter out was_rechecked=True.
    cp_positions = df[
        (df["is_mate"] == False)
        & (df["score_cp"].notna())
    ].copy()

    cp_positions["abs_score_cp"] = (
        cp_positions["score_cp"].abs()
    )

    candidates = cp_positions[
        (cp_positions["abs_score_cp"] >= MIN_SCORE)
        & (cp_positions["abs_score_cp"] < MAX_SCORE)
    ].copy()

    print(
        f"\nPositions with absolute score "
        f"between {MIN_SCORE} and {MAX_SCORE} cp: "
        f"{len(candidates)}"
    )

    if len(candidates) == 0:
        print("\nNo positions found in this score range.")
        return

    # Show how many positions had already been rechecked in V3.
    print("\nExisting V3 verification status:")

    print(
        candidates["was_rechecked"]
        .value_counts(dropna=False)
        .to_string()
    )

    # Sample positions reproducibly.
    sample_count = min(SAMPLE_SIZE, len(candidates))

    sample = candidates.sample(
        n=sample_count,
        random_state=RANDOM_SEED
    ).copy()

    print(f"\nSelected {sample_count} positions for audit.")
    print(f"Stockfish analysis time: {AUDIT_TIME} seconds each")

    # Prepare output directory.
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results = []

    # Start Stockfish.
    print("\nStarting Stockfish...")

    engine = chess.engine.SimpleEngine.popen_uci(
        str(STOCKFISH_PATH)
    )

    engine.configure({
        "Threads": THREADS,
        "Hash": HASH_MB,
    })

    try:

        for index, (_, row) in enumerate(
            sample.iterrows(),
            start=1
        ):

            fen = row["fen"]
            original_score = float(row["score_cp"])

            print(
                f"\n[{index}/{sample_count}] "
                f"Original: {original_score:+.0f} cp"
            )

            try:

                audit = get_stockfish_evaluation(
                    engine,
                    fen
                )

                audit_cp = audit["audit_score_cp"]
                audit_mate = audit["audit_mate_in"]

                # Calculate evaluation change.
                if audit_cp is not None:
                    absolute_change = abs(
                        audit_cp - original_score
                    )

                    signed_change = (
                        audit_cp - original_score
                    )

                    mate_discovered = False

                else:
                    absolute_change = None
                    signed_change = None
                    mate_discovered = True

                result = {
                    "fen": fen,
                    "game_id": row.get("game_id"),
                    "original_score_cp": original_score,
                    "was_rechecked": row["was_rechecked"],
                    "audit_score_cp": audit_cp,
                    "audit_mate_in": audit_mate,
                    "absolute_change_cp": absolute_change,
                    "signed_change_cp": signed_change,
                    "mate_discovered": mate_discovered,
                    "audit_depth": audit["audit_depth"],
                    "audit_nodes": audit["audit_nodes"],
                }

                results.append(result)

                if mate_discovered:
                    print(
                        f"  MATE DISCOVERED: #{audit_mate}"
                    )
                else:
                    print(
                        f"  New evaluation: {audit_cp:+.0f} cp"
                    )
                    print(
                        f"  Absolute change: {absolute_change:.0f} cp"
                    )

            except Exception as exc:

                print(f"  ERROR: {exc}")

                results.append({
                    "fen": fen,
                    "game_id": row.get("game_id"),
                    "original_score_cp": original_score,
                    "was_rechecked": row["was_rechecked"],
                    "audit_score_cp": None,
                    "audit_mate_in": None,
                    "absolute_change_cp": None,
                    "signed_change_cp": None,
                    "mate_discovered": None,
                    "audit_depth": None,
                    "audit_nodes": None,
                    "error": str(exc),
                })

    finally:

        engine.quit()

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    successful = results_df[
        results_df["audit_score_cp"].notna()
        | results_df["audit_mate_in"].notna()
    ]

    mate_rows = successful[
        successful["mate_discovered"] == True
    ]

    cp_rows = successful[
        successful["audit_score_cp"].notna()
    ]

    print("\n" + "=" * 65)
    print("AUDIT SUMMARY")
    print("=" * 65)

    print(f"\nPositions sampled: {sample_count}")
    print(f"Successfully analysed: {len(successful)}")

    print(
        f"Mate discoveries: {len(mate_rows)} "
        f"({len(mate_rows) / len(successful) * 100:.2f}%)"
        if len(successful) else "Mate discoveries: 0"
    )

    print(f"Positions remaining centipawn: {len(cp_rows)}")

    if len(cp_rows):

        changes = cp_rows["absolute_change_cp"].dropna()

        print("\nCentipawn evaluation changes:")

        print(
            f"Mean absolute change: "
            f"{changes.mean():.2f} cp"
        )

        print(
            f"Median absolute change: "
            f"{changes.median():.2f} cp"
        )

        print(
            f"Maximum absolute change: "
            f"{changes.max():.2f} cp"
        )

        print(
            f"Positions changing by more than 100 cp: "
            f"{(changes > 100).sum()}"
        )

        print(
            f"Positions changing by more than 300 cp: "
            f"{(changes > 300).sum()}"
        )

    print("\nMate discoveries:")

    if len(mate_rows):

        for _, row in mate_rows.iterrows():

            print(
                f"  Game {row['game_id']}: "
                f"{row['original_score_cp']:+.0f} cp "
                f"-> #{int(row['audit_mate_in'])}"
            )

    else:
        print("  None")

    print(f"\nFull results saved to:\n{OUTPUT_PATH}")

    print("\nAudit completed.")
    print("=" * 65)


if __name__ == "__main__":
    main()
