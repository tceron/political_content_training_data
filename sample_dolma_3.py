import json
import os
import random
import sqlite3
import requests
import zstandard
import io
from datetime import datetime
from typing import List


def reservoir_sampling_with_checkpoint(
    url_list: List[str],
    sample_size: int = 100000,
    random_seed: int = 42,
    db_path: str = "reservoir.db",
) -> None:
    """
    Perform reservoir sampling with intermediate results and progress persisted
    to a SQLite database on disk, instead of an in-memory list.

    Each reservoir slot is a row keyed by slot_id; as sampling proceeds, a slot
    is either inserted (reservoir not yet full) or replaced in place (`INSERT
    OR REPLACE`) when a later document is chosen to take its place. Only the
    single document currently being read is ever held in memory - the
    reservoir itself always lives in the database.

    Args:
        url_list: List of URLs to download zstandard-compressed jsonl files from
        sample_size: Number of items to sample (default: 100000)
        random_seed: Random seed for reproducibility (default: 42)
        db_path: Path to the SQLite database file (default: "reservoir.db")
    """
    random.seed(random_seed)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reservoir (
            slot_id INTEGER PRIMARY KEY,
            raw_json TEXT NOT NULL
        )
        """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS progress (
            id INTEGER PRIMARY KEY CHECK (id = 0),
            items_seen INTEGER NOT NULL,
            last_file_idx INTEGER NOT NULL,
            sample_size INTEGER NOT NULL,
            random_seed INTEGER NOT NULL,
            updated_at TEXT NOT NULL
        )
        """)
    conn.commit()

    row = conn.execute(
        "SELECT items_seen, last_file_idx FROM progress WHERE id = 0"
    ).fetchone()
    if row is not None:
        items_seen, last_file_idx = row
        start_idx = last_file_idx + 1
        print(
            f"Resumed from {db_path}: {items_seen} items seen, starting from file index {start_idx}"
        )
    else:
        items_seen = 0
        start_idx = 0
        print("Starting fresh sampling")

    # Process files
    for url_idx in range(start_idx, len(url_list)):
        url = url_list[url_idx]
        print(f"\nProcessing file {url_idx + 1}/{len(url_list)}: {url}")

        temp_filename = f"temp_file_{url_idx}.jsonl.zst"
        file_items = 0

        try:
            # Download file
            print(f"  Downloading...")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(temp_filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Process the zstandard-compressed jsonl file
            print(f"  Processing...")
            dctx = zstandard.ZstdDecompressor()
            with open(temp_filename, "rb") as fh:
                with dctx.stream_reader(fh) as reader:
                    text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                    for raw_line in text_stream:
                        line = raw_line.strip()
                        if not line:  # Skip empty lines
                            continue

                        items_seen += 1
                        file_items += 1

                        # Reservoir sampling algorithm: decide which slot (if
                        # any) this single document should occupy, then
                        # add/replace just that one row in the database.
                        if items_seen <= sample_size:
                            slot = items_seen - 1
                        else:
                            j = random.randint(0, items_seen - 1)
                            slot = j if j < sample_size else None

                        if slot is not None:
                            conn.execute(
                                "INSERT OR REPLACE INTO reservoir (slot_id, raw_json) VALUES (?, ?)",
                                (slot, line),
                            )

                        if file_items % 100000 == 0:
                            print(
                                f"    Processed {file_items} items from this file ({items_seen} total)"
                            )

            print(f"  Finished processing {file_items} items from this file")

        except requests.RequestException as e:
            print(f"  Error downloading {url}: {e}")
        except Exception as e:
            print(f"  Error processing {url}: {e}")
        finally:
            # Clean up: delete the temporary file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
                print(f"  Cleaned up temporary file")

        # Persist progress in the same transaction as the reservoir updates
        # for this file, so the database is never left inconsistent.
        conn.execute(
            """
            INSERT OR REPLACE INTO progress
                (id, items_seen, last_file_idx, sample_size, random_seed, updated_at)
            VALUES (0, ?, ?, ?, ?, ?)
            """,
            (
                items_seen,
                url_idx,
                sample_size,
                random_seed,
                datetime.now().isoformat(),
            ),
        )
        conn.commit()

        reservoir_size = conn.execute("SELECT COUNT(*) FROM reservoir").fetchone()[0]
        print(
            f"  Committed to database ({url_idx + 1}/{len(url_list)} files processed)"
        )
        print(f"  Current reservoir size: {reservoir_size}")
        print(f"  Total items seen: {items_seen}")

    conn.close()
    print(f"\nSampling complete. Reservoir stored in {db_path}")


def export_reservoir_to_jsonl(db_path: str, out_file: str) -> None:
    """
    Stream the sampled reservoir from the SQLite database straight to a jsonl
    file, one row at a time, without ever materializing the full sample (raw
    or parsed) in memory.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT raw_json FROM reservoir ORDER BY slot_id")

    count = 0
    parse_errors = 0
    with open(out_file, "w", encoding="utf-8") as out:
        for (raw_json,) in cursor:
            try:
                json.loads(raw_json)  # validate without keeping the parsed object
            except json.JSONDecodeError as e:
                parse_errors += 1
                print(f"Warning: Failed to parse row {count}: {e}")

            out.write(raw_json + "\n")
            count += 1
            if count % 10000 == 0:
                print(f"  Exported {count} items")

    conn.close()
    print(f"\nExported {count} items to {out_file}")
    if parse_errors > 0:
        print(f"  JSON parse errors: {parse_errors}")


if __name__ == "__main__":
    with open("dolma3_links.txt", "r") as inp:
        url_list = [
            line.strip()
            for line in inp
            if (line.strip() and ("common_crawl" in line or "wiki" in line))
        ]

    reservoir_sampling_with_checkpoint(
        url_list,
        sample_size=500000,
        random_seed=42,
        db_path="reservoir.db",
    )

    export_reservoir_to_jsonl("reservoir.db", "dolma_3_sample.jsonl")
