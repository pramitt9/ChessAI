
"""
V4 Stockfish Dataset Generator
==============================

Builds a better-verified evaluation dataset from the existing V3 dataset.

Features:
    - Reuses all 47 V3 features.
    - Preserves all original V3 labels.
    - Deduplicates positions by FEN before Stockfish analysis.
    - Rechecks centipawn positions with abs(score) >= 1000.
    - Rechecks existing V3 mate positions.
    - Uses a second, longer analysis for suspicious results.
    - Saves resumable checkpoints.
    - Keeps mate labels separate from centipawn regression labels.
    - Never modifies V1, V2 or V3 files.

Evaluation convention:
    Scores are always from the perspective of the side to move,
    matching the V3 dataset generation protocol.

Run from project root:
    python ml/generate_stockfish_dataset_v4.py
"""

import json
import time
from pathlib import Path

import chess
import chess.engine
import pandas as pd


# =========================================================
# CONFIGURATION
# =========================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "stockfish_10k_v3.csv"
)

OUTPUT_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "stockfish_10k_v4.csv"
)

CHECKPOINT_PATH = (
    ROOT_DIR
    / "reports"
    / "v4_verification_checkpoint.csv"
)

METADATA_PATH = (
    ROOT_DIR
    / "reports"
    / "stockfish_v4_generation_metadata.json"
)

STOCKFISH_PATH = (
    ROOT_DIR
    / "engines"
    / "stockfish"
    / "stockfish.exe"
)

# Stockfish resource limits
THREADS = 2
HASH_MB = 256

# First verification
PRIMARY_ANALYSIS_TIME = 5.0

# Additional verification for suspicious results
SECONDARY_ANALYSIS_TIME = 10.0

# Positions at or above this original absolute score
# receive the primary verification.
RECHECK_THRESHOLD_CP = 1000

# A change larger than this triggers secondary verification.
LARGE_CHANGE_THRESHOLD_CP = 300

# Positions at or above this original score always receive
# secondary verification after the primary analysis.
EXTREME_SCORE_THRESHOLD_CP = 2000

# Recheck existing V3 mate labels.
RECHECK_EXISTING_MATES = True

# Save checkpoint after this many newly analysed positions.
CHECKPOINT_INTERVAL = 10

# Reproducibility / metadata
GENERATOR_VERSION = "4.0"
EVALUATION_PERSPECTIVE = "side_to_move"


# =========================================================
# VALIDATION
# =========================================================

def validate_paths():
    """Check that all required input files exist."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"V3 dataset not found: {INPUT_PATH}"
        )

    if not STOCKFISH_PATH.exists():
        raise FileNotFoundError(
            f"Stockfish executable not found: {STOCKFISH_PATH}"
        )


def load_v3_dataset():
    """Load and validate the V3 dataset."""

    print("\nLoading V3 dataset...")

    df = pd.read_csv(INPUT_PATH)

    required_columns = {
        "fen",
        "game_id",
        "score_cp",
        "mate_in",
        "is_mate",
        "was_rechecked",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"V3 dataset missing columns: {sorted(missing)}"
        )

    if df["fen"].isna().any():
        raise ValueError("Dataset contains missing FEN values.")

    print(f"Loaded rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Unique FENs: {df['fen'].nunique():,}")

    return df


# =========================================================
# STOCKFISH ANALYSIS
# =========================================================

def analyse_position(engine, fen, analysis_time):
    """
    Analyse one position using Stockfish.

    Returns evaluation from the side-to-move perspective.

    For centipawn evaluations:
        score_cp = integer
        mate_in = None
        is_mate = False

    For mate evaluations:
        score_cp = None
        mate_in = integer
        is_mate = True
    """

    board = chess.Board(fen)

    if not board.is_valid():
        raise ValueError(f"Invalid chess position: {fen}")

    info = engine.analyse(
        board,
        chess.engine.Limit(time=analysis_time),
    )

    pov_score = info["score"].pov(board.turn)

    score_cp = pov_score.score()
    mate_in = pov_score.mate()

    is_mate = mate_in is not None

    return {
        "score_cp": score_cp,
        "mate_in": mate_in,
        "is_mate": is_mate,
        "raw_score": str(pov_score),
        "depth": info.get("depth"),
        "nodes": info.get("nodes"),
        "nps": info.get("nps"),
        "actual_time": info.get("time"),
    }


# =========================================================
# POSITION VERIFICATION
# =========================================================

def get_position_result(engine, row):
    """
    Verify one unique FEN.

    The original V3 label determines whether the position
    needs rechecking.

    Primary verification:
        - Existing V3 mate positions, if enabled.
        - Centipawn positions with abs(score) >= 1000.

    Secondary verification:
        - Mate discovered in primary verification.
        - Original absolute score >= 2000 cp.
        - Primary evaluation differs by more than 300 cp.

    The secondary result becomes the final V4 label.
    """

    fen = row["fen"]

    original_score = row["score_cp"]
    original_mate = row["mate_in"]
    original_is_mate = bool(row["is_mate"])

    result = {
        "fen": fen,

        "v3_score_cp": original_score,
        "v3_mate_in": original_mate,
        "v3_is_mate": original_is_mate,

        "v4_score_cp": original_score,
        "v4_mate_in": original_mate,
        "v4_is_mate": original_is_mate,

        "verification_status": "retained_v3",
        "verification_reason": "below_threshold",

        "primary_score_cp": None,
        "primary_mate_in": None,
        "primary_is_mate": None,
        "primary_depth": None,
        "primary_nodes": None,
        "primary_time": None,

        "secondary_score_cp": None,
        "secondary_mate_in": None,
        "secondary_is_mate": None,
        "secondary_depth": None,
        "secondary_nodes": None,
        "secondary_time": None,

        "total_verification_time": 0.0,
        "absolute_change_cp": None,
        "error": None,
    }

    # -----------------------------------------------------
    # Determine whether this position needs verification.
    # -----------------------------------------------------

    if original_is_mate:

        if not RECHECK_EXISTING_MATES:
            result["verification_reason"] = (
                "existing_mate_retained"
            )
            return result

        result["verification_reason"] = (
            "existing_v3_mate"
        )

    elif pd.notna(original_score):

        original_score = float(original_score)

        if abs(original_score) < RECHECK_THRESHOLD_CP:
            return result

        result["verification_reason"] = (
            "high_centipawn_score"
        )

    else:
        result["verification_status"] = "invalid_original_label"
        result["verification_reason"] = "missing_evaluation"
        return result

    # -----------------------------------------------------
    # Primary verification
    # -----------------------------------------------------

    primary = analyse_position(
        engine,
        fen,
        PRIMARY_ANALYSIS_TIME,
    )

    result["primary_score_cp"] = primary["score_cp"]
    result["primary_mate_in"] = primary["mate_in"]
    result["primary_is_mate"] = primary["is_mate"]
    result["primary_depth"] = primary["depth"]
    result["primary_nodes"] = primary["nodes"]
    result["primary_time"] = primary["actual_time"]

    primary_cp = primary["score_cp"]
    primary_mate = primary["mate_in"]
    primary_is_mate = primary["is_mate"]

    result["v4_score_cp"] = primary_cp
    result["v4_mate_in"] = primary_mate
    result["v4_is_mate"] = primary_is_mate

    primary_change = None

    if (
        not original_is_mate
        and primary_cp is not None
    ):
        primary_change = abs(
            primary_cp - original_score
        )

        result["absolute_change_cp"] = primary_change

    # -----------------------------------------------------
    # Determine whether secondary verification is needed.
    # -----------------------------------------------------

    needs_secondary = False

    if primary_is_mate:
        needs_secondary = True

    if (
        not original_is_mate
        and abs(original_score) >= EXTREME_SCORE_THRESHOLD_CP
    ):
        needs_secondary = True

    if (
        primary_change is not None
        and primary_change > LARGE_CHANGE_THRESHOLD_CP
    ):
        needs_secondary = True

    # -----------------------------------------------------
    # Secondary verification
    # -----------------------------------------------------

    if needs_secondary:

        secondary = analyse_position(
            engine,
            fen,
            SECONDARY_ANALYSIS_TIME,
        )

        result["secondary_score_cp"] = secondary["score_cp"]
        result["secondary_mate_in"] = secondary["mate_in"]
        result["secondary_is_mate"] = secondary["is_mate"]
        result["secondary_depth"] = secondary["depth"]
        result["secondary_nodes"] = secondary["nodes"]
        result["secondary_time"] = secondary["actual_time"]

        # The longer analysis becomes the final label.
        result["v4_score_cp"] = secondary["score_cp"]
        result["v4_mate_in"] = secondary["mate_in"]
        result["v4_is_mate"] = secondary["is_mate"]

        result["verification_status"] = (
            "secondary_verified"
        )

    else:

        result["verification_status"] = (
            "primary_verified"
        )

    # -----------------------------------------------------
    # Calculate final label change.
    # -----------------------------------------------------

    if not original_is_mate and result["v4_score_cp"] is not None:

        result["absolute_change_cp"] = abs(
            result["v4_score_cp"] - original_score
        )

    # Verification time is the requested analysis budget.
    result["total_verification_time"] = (
        PRIMARY_ANALYSIS_TIME
        + (
            SECONDARY_ANALYSIS_TIME
            if needs_secondary
            else 0.0
        )
    )

    return result


# =========================================================
# CHECKPOINT MANAGEMENT
# =========================================================

def load_checkpoint():
    """Load previous verification results if available."""

    if not CHECKPOINT_PATH.exists():
        print("\nNo previous checkpoint found.")
        return {}

    print("\nLoading existing checkpoint...")

    checkpoint = pd.read_csv(CHECKPOINT_PATH)

    if "fen" not in checkpoint.columns:
        raise ValueError(
            "Checkpoint does not contain a FEN column."
        )

    cache = {}

    for _, row in checkpoint.iterrows():

        fen = row["fen"]

        cache[fen] = row.to_dict()

    print(
        f"Recovered {len(cache):,} previously processed positions."
    )

    return cache


def save_checkpoint(cache):
    """Save all completed verification results."""

    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_df = pd.DataFrame(
        list(cache.values())
    )

    temp_path = CHECKPOINT_PATH.with_suffix(".tmp.csv")

    checkpoint_df.to_csv(
        temp_path,
        index=False,
    )

    temp_path.replace(CHECKPOINT_PATH)


# =========================================================
# DATASET GENERATION
# =========================================================

def generate_dataset(df):
    """Verify unique positions and construct the V4 dataset."""

    unique_positions = (
        df.drop_duplicates(subset=["fen"])
        .copy()
    )

    print("\n" + "=" * 65)
    print("V4 DATASET GENERATION")
    print("=" * 65)

    print(f"Original rows: {len(df):,}")
    print(f"Unique positions: {len(unique_positions):,}")
    print(
        f"Duplicate rows: "
        f"{len(df) - len(unique_positions):,}"
    )

    # Load resumable results.
    cache = load_checkpoint()

    pending = unique_positions[
        ~unique_positions["fen"].isin(cache.keys())
    ]

    print(f"\nAlready processed: {len(cache):,}")
    print(f"Remaining positions: {len(pending):,}")

    # Estimate analysis budget.
    needs_verification = pending[
        (
            pending["is_mate"]
            & RECHECK_EXISTING_MATES
        )
        | (
            ~pending["is_mate"]
            & pending["score_cp"].notna()
            & (
                pending["score_cp"].abs()
                >= RECHECK_THRESHOLD_CP
            )
        )
    ]

    print(
        f"Positions requiring primary verification: "
        f"{len(needs_verification):,}"
    )

    print(
        "\nNOTE: This is a rough estimate. "
        "Secondary verification adds extra analysis time."
    )

    # Start Stockfish.
    print("\nStarting Stockfish...")

    engine = chess.engine.SimpleEngine.popen_uci(
        str(STOCKFISH_PATH)
    )

    engine.configure({
        "Threads": THREADS,
        "Hash": HASH_MB,
    })

    start_time = time.time()

    newly_processed = 0
    failed_positions = 0

    try:

        for index, (_, row) in enumerate(
            pending.iterrows(),
            start=1,
        ):

            fen = row["fen"]

            print(
                f"\n[{index}/{len(pending)}] "
                f"Game {row['game_id']} | Ply {row['ply']}"
            )

            try:

                position_result = get_position_result(
                    engine,
                    row,
                )

                cache[fen] = position_result

                if position_result["v4_is_mate"]:
                    print(
                        "  V4 evaluation: MATE "
                        f"#{position_result['v4_mate_in']}"
                    )

                else:
                    print(
                        "  V4 evaluation: "
                        f"{position_result['v4_score_cp']:+.0f} cp"
                    )

                print(
                    "  Status: "
                    f"{position_result['verification_status']}"
                )

                if position_result["absolute_change_cp"] is not None:
                    print(
                        "  Absolute change: "
                        f"{position_result['absolute_change_cp']:.0f} cp"
                    )

            except Exception as exc:

                failed_positions += 1

                print(f"  ERROR: {exc}")

                # Do not silently retain a potentially invalid label.
                # Failed positions will be excluded from V4 training.
                cache[fen] = {
                    "fen": fen,
                    "v3_score_cp": row["score_cp"],
                    "v3_mate_in": row["mate_in"],
                    "v3_is_mate": bool(row["is_mate"]),
                    "v4_score_cp": None,
                    "v4_mate_in": None,
                    "v4_is_mate": None,
                    "verification_status": "verification_failed",
                    "verification_reason": "stockfish_error",
                    "primary_score_cp": None,
                    "primary_mate_in": None,
                    "primary_is_mate": None,
                    "primary_depth": None,
                    "primary_nodes": None,
                    "primary_time": None,
                    "secondary_score_cp": None,
                    "secondary_mate_in": None,
                    "secondary_is_mate": None,
                    "secondary_depth": None,
                    "secondary_nodes": None,
                    "secondary_time": None,
                    "total_verification_time": 0.0,
                    "absolute_change_cp": None,
                    "error": str(exc),
                }

            newly_processed += 1

            if newly_processed % CHECKPOINT_INTERVAL == 0:

                save_checkpoint(cache)

                elapsed = time.time() - start_time

                print(
                    f"\nCheckpoint saved. "
                    f"Elapsed: {elapsed / 60:.1f} minutes"
                )

    finally:

        # Always save completed work before closing Stockfish.
        save_checkpoint(cache)

        engine.quit()

    elapsed = time.time() - start_time

    print("\nVerification stage completed.")
    print(f"Newly processed: {newly_processed:,}")
    print(f"Failed positions: {failed_positions:,}")
    print(f"Elapsed time: {elapsed / 60:.1f} minutes")

    # -----------------------------------------------------
    # Map unique-position results back to all original rows.
    # -----------------------------------------------------

    result_df = df.copy()

    cache_df = pd.DataFrame(
        list(cache.values())
    )

    result_df = result_df.merge(
        cache_df,
        on="fen",
        how="left",
        suffixes=("", "_verified"),
        validate="many_to_one",
    )
    result_df["v3_score_cp"] = result_df["score_cp"]
    result_df["v3_mate_in"] = result_df["mate_in"]
    result_df["v3_is_mate"] = result_df["is_mate"]

    # -----------------------------------------------------
    # Final validation fields
    # -----------------------------------------------------

    result_df["v4_is_mate"] = (
        result_df["v4_is_mate"].astype("boolean")
    )

    result_df["v4_score_cp"] = pd.to_numeric(
        result_df["v4_score_cp"],
        errors="coerce",
    )

    result_df["v4_mate_in"] = pd.to_numeric(
        result_df["v4_mate_in"],
        errors="coerce",
    )

    # A row is usable if it has a valid final CP or mate label.
    result_df["v4_label_valid"] = (
        result_df["verification_status"].ne(
            "verification_failed"
        )
        & (
            result_df["v4_score_cp"].notna()
            | (
                result_df["v4_is_mate"].eq(True).fillna(False)
                & result_df["v4_mate_in"].notna()
            )
        )
    )

    # -----------------------------------------------------
    # Save final dataset
    # -----------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    return result_df, elapsed, failed_positions


# =========================================================
# REPORTING
# =========================================================

def generate_metadata(result_df, elapsed, failed_positions):
    """Create a summary of V4 generation."""

    valid = result_df[
        result_df["v4_label_valid"]
    ]

    cp_rows = valid[
        valid["v4_is_mate"] == False
    ]

    mate_rows = valid[
        valid["v4_is_mate"] == True
    ]

    status_counts = (
        result_df["verification_status"]
        .value_counts(dropna=False)
        .to_dict()
    )

    metadata = {
        "generator_version": GENERATOR_VERSION,
        "source_dataset": str(INPUT_PATH),
        "output_dataset": str(OUTPUT_PATH),

        "original_rows": int(len(result_df)),
        "unique_fen_positions": int(
            result_df["fen"].nunique()
        ),

        "feature_count": 47,

        "evaluation_perspective": EVALUATION_PERSPECTIVE,

        "stockfish": {
            "path": str(STOCKFISH_PATH),
            "threads": THREADS,
            "hash_mb": HASH_MB,
            "primary_analysis_time": PRIMARY_ANALYSIS_TIME,
            "secondary_analysis_time": SECONDARY_ANALYSIS_TIME,
        },

        "verification_policy": {
            "recheck_threshold_cp": RECHECK_THRESHOLD_CP,
            "large_change_threshold_cp": LARGE_CHANGE_THRESHOLD_CP,
            "extreme_score_threshold_cp": EXTREME_SCORE_THRESHOLD_CP,
            "recheck_existing_mates": RECHECK_EXISTING_MATES,
        },

        "valid_rows": int(len(valid)),
        "invalid_rows": int(len(result_df) - len(valid)),

        "centipawn_rows": int(len(cp_rows)),
        "mate_rows": int(len(mate_rows)),

        "failed_positions": int(failed_positions),

        "verification_status_counts": {
            str(key): int(value)
            for key, value in status_counts.items()
        },

        "mean_absolute_label_change_cp": (
            float(
                result_df["absolute_change_cp"]
                .dropna()
                .mean()
            )
            if result_df["absolute_change_cp"].notna().any()
            else None
        ),

        "median_absolute_label_change_cp": (
            float(
                result_df["absolute_change_cp"]
                .dropna()
                .median()
            )
            if result_df["absolute_change_cp"].notna().any()
            else None
        ),

        "elapsed_seconds": float(elapsed),
    }

    METADATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            indent=4,
        )

    return metadata


def print_summary(result_df, metadata):
    """Print final generation statistics."""

    print("\n" + "=" * 65)
    print("V4 GENERATION SUMMARY")
    print("=" * 65)

    print(f"\nOriginal rows: {len(result_df):,}")

    print(
        f"Unique FEN positions: "
        f"{result_df['fen'].nunique():,}"
    )

    print(
        f"Valid rows: "
        f"{metadata['valid_rows']:,}"
    )

    print(
        f"Invalid rows: "
        f"{metadata['invalid_rows']:,}"
    )

    print(
        f"Centipawn rows: "
        f"{metadata['centipawn_rows']:,}"
    )

    print(
        f"Mate rows: "
        f"{metadata['mate_rows']:,}"
    )

    print("\nVerification statuses:")

    for status, count in metadata[
        "verification_status_counts"
    ].items():

        print(f"  {status}: {count}")

    print("\nLabel changes:")

    print(
        "  Mean absolute change: "
        f"{metadata['mean_absolute_label_change_cp']}"
    )

    print(
        "  Median absolute change: "
        f"{metadata['median_absolute_label_change_cp']}"
    )

    print("\nOutput files:")

    print(f"  Dataset: {OUTPUT_PATH}")
    print(f"  Checkpoint: {CHECKPOINT_PATH}")
    print(f"  Metadata: {METADATA_PATH}")

    print("\nV4 dataset generation completed.")


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 65)
    print("STOCKFISH V4 DATASET GENERATOR")
    print("=" * 65)

    validate_paths()

    df = load_v3_dataset()

    result_df, elapsed, failed_positions = generate_dataset(df)

    metadata = generate_metadata(
        result_df,
        elapsed,
        failed_positions,
    )

    print_summary(
        result_df,
        metadata,
    )


if __name__ == "__main__":
    main()
