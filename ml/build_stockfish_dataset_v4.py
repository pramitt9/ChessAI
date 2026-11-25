
"""
Build Stockfish V4 Dataset
===========================

Combines:
    1. stockfish_10k_v3.csv
    2. v3_label_audit.csv
    3. v3_low_score_audit.csv
    4. v3_mid_score_audit.csv

No Stockfish analysis is performed.

Features:
    - Reuses all V3 features.
    - Reuses existing five-second audit results.
    - Matches audit results by FEN.
    - Resolves duplicate audit records using search depth and nodes.
    - Preserves all original V3 rows and labels.
    - Separates centipawn and mate labels.
    - Records label provenance and verification information.
    - Produces a metadata report.

Run from project root:
    python ml/build_stockfish_dataset_v4.py
"""

import json
from pathlib import Path

import pandas as pd


# =========================================================
# PATHS
# =========================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

V3_DATASET = (
    ROOT_DIR
    / "data"
    / "processed"
    / "stockfish_10k_v3.csv"
)

AUDIT_FILES = {
    "v3_label_audit": (
        ROOT_DIR / "reports" / "v3_label_audit.csv"
    ),
    "v3_low_score_audit": (
        ROOT_DIR / "reports" / "v3_low_score_audit.csv"
    ),
    "v3_mid_score_audit": (
        ROOT_DIR / "reports" / "v3_mid_score_audit.csv"
    ),
}

OUTPUT_DATASET = (
    ROOT_DIR
    / "data"
    / "processed"
    / "stockfish_10k_v4.csv"
)

OUTPUT_METADATA = (
    ROOT_DIR
    / "reports"
    / "stockfish_v4_generation_metadata.json"
)


# =========================================================
# HELPERS
# =========================================================

def to_number(value):
    """Convert a value to a number, returning None if invalid."""

    if pd.isna(value):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_boolean(value):
    """Safely interpret CSV boolean values."""

    if pd.isna(value):
        return False

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    return str(value).strip().lower() in {
        "true", "1", "yes", "y"
    }


def first_available(row, columns):
    """Return the first non-null value among candidate columns."""

    for column in columns:
        if column in row.index and pd.notna(row[column]):
            return row[column]

    return None


# =========================================================
# LOAD V3 DATASET
# =========================================================

def load_v3_dataset():

    print("\nLoading V3 dataset...")

    if not V3_DATASET.exists():
        raise FileNotFoundError(
            f"V3 dataset not found: {V3_DATASET}"
        )

    df = pd.read_csv(V3_DATASET)

    required = {
        "fen",
        "game_id",
        "score_cp",
        "mate_in",
        "is_mate",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"V3 dataset is missing columns: {sorted(missing)}"
        )

    if df["fen"].isna().any():
        raise ValueError("V3 dataset contains missing FEN values.")

    df["is_mate"] = df["is_mate"].map(to_boolean)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Unique FENs: {df['fen'].nunique():,}")

    return df


# =========================================================
# NORMALISE AUDIT FILES
# =========================================================

def normalise_audit_file(path, source_name):
    """
    Convert different audit CSV schemas into one common format.

    Supported schemas:
        audit_cp / audit_mate
        audit_score_cp / audit_mate_in
        mate_discovered
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Audit file not found: {path}"
        )

    df = pd.read_csv(path)

    if "fen" not in df.columns:
        raise ValueError(
            f"{path.name} does not contain a FEN column."
        )

    records = []

    for _, row in df.iterrows():

        fen = row["fen"]

        if pd.isna(fen):
            continue

        audit_cp = to_number(
            first_available(
                row,
                ["audit_score_cp", "audit_cp"]
            )
        )

        audit_mate = to_number(
            first_available(
                row,
                ["audit_mate_in", "audit_mate"]
            )
        )

        mate_discovered = to_boolean(
            row.get("mate_discovered", False)
        )

        # A non-null mate distance means Stockfish returned mate.
        is_mate = (
            audit_mate is not None
            or mate_discovered
        )

        # If a mate was discovered but the distance is missing,
        # the position cannot be used as a valid mate training label.
        if is_mate and audit_mate is None:
            print(
                f"WARNING: Mate distance missing in {source_name}: "
                f"{str(fen)[:45]}"
            )

        # Do not accept a row without a usable evaluation.
        if not is_mate and audit_cp is None:
            continue

        records.append({
            "fen": str(fen),

            "audit_score_cp": (
                None if is_mate else audit_cp
            ),

            "audit_mate_in": audit_mate,

            "audit_is_mate": is_mate,

            "audit_depth": to_number(
                first_available(
                    row,
                    ["audit_depth", "depth"]
                )
            ),

            "audit_nodes": to_number(
                first_available(
                    row,
                    ["audit_nodes", "nodes"]
                )
            ),

            "audit_time": to_number(
                first_available(
                    row,
                    ["time", "audit_time"]
                )
            ),

            "audit_raw_score": first_available(
                row,
                ["raw_score"]
            ),

            "audit_source": source_name,
        })

    result = pd.DataFrame(records)

    print(
        f"{source_name}: "
        f"{len(df)} source rows, "
        f"{len(result)} usable audit rows, "
        f"{result['fen'].nunique() if len(result) else 0} unique FENs"
    )

    return result


def load_all_audits():

    print("\nLoading existing audit reports...")

    audit_frames = []

    for source_name, path in AUDIT_FILES.items():

        audit_df = normalise_audit_file(
            path,
            source_name
        )

        audit_frames.append(audit_df)

    combined = pd.concat(
        audit_frames,
        ignore_index=True
    )

    print("\nCombined audit statistics:")
    print(f"Total audit records: {len(combined)}")
    print(f"Unique audited FENs: {combined['fen'].nunique()}")

    return combined


# =========================================================
# RESOLVE DUPLICATE AUDIT RESULTS
# =========================================================

def resolve_duplicate_audits(audits):
    """
    Select one audit result per FEN.

    When the same FEN appears more than once, prefer:
        1. Greater search depth.
        2. More nodes searched.
        3. Longer recorded analysis time.

    This provides a deterministic selection rule.
    """

    if audits.empty:
        return audits.copy(), {}

    audits = audits.copy()

    audits["audit_depth"] = pd.to_numeric(
        audits["audit_depth"],
        errors="coerce"
    )

    audits["audit_nodes"] = pd.to_numeric(
        audits["audit_nodes"],
        errors="coerce"
    )

    audits["audit_time"] = pd.to_numeric(
        audits["audit_time"],
        errors="coerce"
    )

    audits = audits.sort_values(
        by=[
            "audit_depth",
            "audit_nodes",
            "audit_time",
        ],
        ascending=[False, False, False],
        na_position="last",
        kind="stable",
    )

    audit_counts = (
        audits.groupby("fen")
        .size()
        .to_dict()
    )

    selected = (
        audits.drop_duplicates(
            subset=["fen"],
            keep="first"
        )
        .copy()
    )

    print("\nDuplicate audit resolution:")
    print(f"Audit records: {len(audits)}")
    print(f"Selected unique results: {len(selected)}")
    print(f"Duplicate records removed: {len(audits) - len(selected)}")

    return selected, audit_counts


# =========================================================
# BUILD V4 DATASET
# =========================================================

def build_v4_dataset(v3, selected_audits, audit_counts):

    print("\nBuilding V4 dataset...")

    result = v3.copy()

    # Preserve the original V3 labels exactly as stored.
    result["v3_original_score_cp"] = result["score_cp"]
    result["v3_original_mate_in"] = result["mate_in"]
    result["v3_original_is_mate"] = result["is_mate"]

    # Default V4 labels are the original V3 labels.
    result["v4_score_cp"] = result["score_cp"]
    result["v4_mate_in"] = result["mate_in"]
    result["v4_is_mate"] = result["is_mate"]

    # Default status: retained from V3, not independently verified.
    result["v4_verified"] = False
    result["v4_label_source"] = "v3_unverified"
    result["v4_verification_status"] = "retained_v3_unverified"

    result["v4_verification_depth"] = None
    result["v4_verification_nodes"] = None
    result["v4_verification_time"] = None
    result["v4_audit_record_count"] = 0

    # -----------------------------------------------------
    # Build lookup by FEN.
    # -----------------------------------------------------

    audit_lookup = {}

    for _, audit in selected_audits.iterrows():

        audit_lookup[audit["fen"]] = audit.to_dict()

    # -----------------------------------------------------
    # Apply verified labels to every matching V3 row.
    # -----------------------------------------------------

    for index, row in result.iterrows():

        fen = row["fen"]

        audit = audit_lookup.get(fen)

        if audit is None:
            continue

        is_mate = bool(audit["audit_is_mate"])

        if is_mate:

            result.at[index, "v4_score_cp"] = None
            result.at[index, "v4_mate_in"] = audit["audit_mate_in"]
            result.at[index, "v4_is_mate"] = True

        else:

            result.at[index, "v4_score_cp"] = audit["audit_score_cp"]
            result.at[index, "v4_mate_in"] = None
            result.at[index, "v4_is_mate"] = False

        result.at[index, "v4_verified"] = True
        result.at[index, "v4_label_source"] = audit["audit_source"]
        result.at[index, "v4_verification_status"] = "audited_5s"

        result.at[index, "v4_verification_depth"] = audit["audit_depth"]
        result.at[index, "v4_verification_nodes"] = audit["audit_nodes"]
        result.at[index, "v4_verification_time"] = audit["audit_time"]

        result.at[index, "v4_audit_record_count"] = (
            audit_counts.get(fen, 1)
        )

    # -----------------------------------------------------
    # Normalise data types.
    # -----------------------------------------------------

    result["v4_score_cp"] = pd.to_numeric(
        result["v4_score_cp"],
        errors="coerce"
    )

    result["v4_mate_in"] = pd.to_numeric(
        result["v4_mate_in"],
        errors="coerce"
    )

    result["v4_is_mate"] = result["v4_is_mate"].map(
        to_boolean
    )

    result["v4_verified"] = result["v4_verified"].map(
        to_boolean
    )

    result["v4_audit_record_count"] = pd.to_numeric(
        result["v4_audit_record_count"],
        errors="coerce"
    ).fillna(0).astype(int)

    # -----------------------------------------------------
    # Label validity.
    # -----------------------------------------------------

    valid_cp = (
        ~result["v4_is_mate"]
        & result["v4_score_cp"].notna()
    )

    valid_mate = (
        result["v4_is_mate"]
        & result["v4_mate_in"].notna()
    )

    result["v4_label_valid"] = (
        valid_cp | valid_mate
    )

    # -----------------------------------------------------
    # Label change analysis.
    # -----------------------------------------------------

    result["v4_absolute_change_cp"] = None

    comparable = (
        result["v4_verified"]
        & ~result["v4_is_mate"]
        & ~result["v3_original_is_mate"]
        & result["v4_score_cp"].notna()
        & result["v3_original_score_cp"].notna()
    )

    result.loc[
        comparable,
        "v4_absolute_change_cp"
    ] = (
        result.loc[comparable, "v4_score_cp"]
        - result.loc[comparable, "v3_original_score_cp"]
    ).abs()

    result["v4_mate_discovered"] = (
        result["v4_verified"]
        & ~result["v3_original_is_mate"]
        & result["v4_is_mate"]
    )

    return result


# =========================================================
# VALIDATION
# =========================================================

def validate_v4_dataset(v3, v4):

    print("\nValidating V4 dataset...")

    assert len(v3) == len(v4), (
        "V4 row count differs from V3."
    )

    assert v3["fen"].equals(v4["fen"]), (
        "FEN order or values changed."
    )

    assert v4["fen"].notna().all(), (
        "Missing FEN values detected."
    )

    # Ensure original features and labels were preserved.
    for column in v3.columns:

        if column not in v4.columns:
            raise AssertionError(
                f"Original V3 column missing: {column}"
            )

        if not v3[column].equals(v4[column]):
            raise AssertionError(
                f"Original V3 column changed: {column}"
            )

    # Every verified position must have an audit source.
    verified = v4["v4_verified"]

    assert v4.loc[
        verified, "v4_label_source"
    ].ne("v3_unverified").all()

    # No row may simultaneously contain a CP and mate label.
    invalid_mate = (
        v4["v4_is_mate"]
        & v4["v4_score_cp"].notna()
    )

    assert not invalid_mate.any(), (
        "Mate rows contain centipawn labels."
    )

    invalid_cp = (
        ~v4["v4_is_mate"]
        & v4["v4_mate_in"].notna()
    )

    assert not invalid_cp.any(), (
        "Centipawn rows contain mate distances."
    )

    print("PASS: V3 row count preserved.")
    print("PASS: FEN values and order preserved.")
    print("PASS: Original V3 columns preserved.")
    print("PASS: Verified labels have audit sources.")
    print("PASS: Mate and centipawn labels are separate.")


# =========================================================
# METADATA
# =========================================================

def create_metadata(v3, audits, selected_audits, v4):

    verified = v4[v4["v4_verified"]]

    cp_rows = v4[
        v4["v4_label_valid"]
        & ~v4["v4_is_mate"]
    ]

    mate_rows = v4[
        v4["v4_label_valid"]
        & v4["v4_is_mate"]
    ]

    verified_cp = verified[
        ~verified["v4_is_mate"]
        & verified["v4_score_cp"].notna()
    ]

    original_cp_to_mate = int(
        v4["v4_mate_discovered"].sum()
    )

    changes = (
        v4["v4_absolute_change_cp"]
        .dropna()
    )

    metadata = {
        "dataset_version": "V4",
        "generation_method": "Reuse existing V3 audit results",
        "stockfish_analysis_performed": False,

        "source_dataset": str(V3_DATASET),

        "audit_sources": {
            name: str(path)
            for name, path in AUDIT_FILES.items()
        },

        "original_rows": int(len(v3)),
        "v4_rows": int(len(v4)),

        "original_unique_fens": int(
            v3["fen"].nunique()
        ),

        "v4_unique_fens": int(
            v4["fen"].nunique()
        ),

        "original_feature_count": 47,

        "total_audit_records": int(len(audits)),

        "unique_audited_fens": int(
            audits["fen"].nunique()
        ),

        "selected_unique_audit_fens": int(
            len(selected_audits)
        ),

        "verified_rows": int(
            v4["v4_verified"].sum()
        ),

        "unverified_rows": int(
            (~v4["v4_verified"]).sum()
        ),

        "valid_rows": int(
            v4["v4_label_valid"].sum()
        ),

        "invalid_rows": int(
            (~v4["v4_label_valid"]).sum()
        ),

        "centipawn_rows": int(len(cp_rows)),

        "mate_rows": int(len(mate_rows)),

        "mate_discoveries_from_cp": original_cp_to_mate,

        "verified_centipawn_rows": int(
            len(verified_cp)
        ),

        "label_change_mean_cp": (
            float(changes.mean())
            if len(changes) else None
        ),

        "label_change_median_cp": (
            float(changes.median())
            if len(changes) else None
        ),

        "verification_source_counts": {
            str(key): int(value)
            for key, value in (
                v4["v4_label_source"]
                .value_counts()
                .items()
            )
        },

        "verification_status_counts": {
            str(key): int(value)
            for key, value in (
                v4["v4_verification_status"]
                .value_counts()
                .items()
            )
        },

        "note": (
            "Unaudited positions retain V3 labels and are explicitly "
            "marked unverified. Audit labels are reused by FEN. "
            "Training must deduplicate FENs before splitting."
        ),
    }

    return metadata


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 65)
    print("STOCKFISH V4 DATASET BUILDER")
    print("=" * 65)

    v3 = load_v3_dataset()

    audits = load_all_audits()

    selected_audits, audit_counts = (
        resolve_duplicate_audits(audits)
    )

    v4 = build_v4_dataset(
        v3,
        selected_audits,
        audit_counts
    )

    validate_v4_dataset(v3, v4)

    # Save dataset.
    OUTPUT_DATASET.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    v4.to_csv(
        OUTPUT_DATASET,
        index=False
    )

    # Save metadata.
    metadata = create_metadata(
        v3,
        audits,
        selected_audits,
        v4
    )

    OUTPUT_METADATA.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_METADATA,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("V4 DATASET SUMMARY")
    print("=" * 65)

    print(f"\nRows: {len(v4):,}")
    print(f"Unique FENs: {v4['fen'].nunique():,}")

    print(
        f"Verified rows: "
        f"{v4['v4_verified'].sum():,}"
    )

    print(
        f"Unverified rows: "
        f"{(~v4['v4_verified']).sum():,}"
    )

    print(
        f"Valid centipawn rows: "
        f"{metadata['centipawn_rows']:,}"
    )

    print(
        f"Valid mate rows: "
        f"{metadata['mate_rows']:,}"
    )

    print(
        f"Mate discoveries from CP: "
        f"{metadata['mate_discoveries_from_cp']:,}"
    )

    print("\nVerification sources:")

    for source, count in metadata[
        "verification_source_counts"
    ].items():

        print(f"  {source}: {count}")

    print("\nLabel changes:")

    print(
        f"  Mean: {metadata['label_change_mean_cp']}"
    )

    print(
        f"  Median: {metadata['label_change_median_cp']}"
    )

    print("\nSaved dataset:")
    print(OUTPUT_DATASET)

    print("\nSaved metadata:")
    print(OUTPUT_METADATA)

    print("\nV4 dataset build completed successfully.")


if __name__ == "__main__":
    main()
