
import io
import chess
import chess.pgn
import zstandard as zstd

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "lichess_2013-01.pgn.zst"
OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "sample_100k.pgn"

TARGET_GAMES = 100_000

VALID_RESULTS = {"1-0", "0-1", "1/2-1/2"}


def extract_games():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    games_seen = 0
    games_saved = 0
    games_skipped = 0

    decompressor = zstd.ZstdDecompressor()

    print("Starting PGN extraction...")
    print(f"Target: {TARGET_GAMES:,} games")

    with open(INPUT_FILE, "rb") as compressed_file:
        with decompressor.stream_reader(compressed_file) as reader:
            with io.TextIOWrapper(reader, encoding="utf-8") as pgn_stream:
                with open(OUTPUT_FILE, "w", encoding="utf-8") as output:

                    while games_saved < TARGET_GAMES:
                        try:
                            game = chess.pgn.read_game(pgn_stream)
                        except Exception as error:
                            print(f"PGN parsing error: {error}")
                            break

                        if game is None:
                            break

                        games_seen += 1

                        result = game.headers.get("Result", "*")
                        variant = game.headers.get("Variant", "Standard")

                        # Keep standard chess games with a known result.
                        if (
                            result not in VALID_RESULTS
                            or variant not in ("Standard", "")
                            or "FEN" in game.headers
                            or game.errors
                        ):
                            games_skipped += 1
                            continue

                        exporter = chess.pgn.StringExporter(
                            headers=True,
                            variations=False,
                            comments=False
                        )

                        output.write(game.accept(exporter))
                        output.write("\n\n")

                        games_saved += 1

                        if games_saved % 10_000 == 0:
                            print(
                                f"Saved {games_saved:,} games "
                                f"| Read {games_seen:,} "
                                f"| Skipped {games_skipped:,}"
                            )

    print("\nExtraction complete.")
    print(f"Games read:    {games_seen:,}")
    print(f"Games saved:   {games_saved:,}")
    print(f"Games skipped: {games_skipped:,}")
    print(f"Output: {OUTPUT_FILE}")

    if games_saved < TARGET_GAMES:
        print(
            f"WARNING: Only {games_saved:,} eligible games were found."
        )


if __name__ == "__main__":
    extract_games()
