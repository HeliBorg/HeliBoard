#!/usr/bin/env python3
"""
Builds `app/src/main/assets/dicts/ja/kana_kanji.dat`

starter dictionary + connection-cost matrix for Japanese kana->kanji conversion
(KanaKanjiEngine.kt / LoudsTrie.kt).

Source data: Mozc's OSS dictionary text files and connection matrix
(BSD-3-Clause):
  https://github.com/google/mozc/tree/master/src/data/dictionary_oss
    - dictionary00.txt .. dictionary09.txt
    - connection_single_column.txt

Inspired by:
    https://github.com/google/mozc/blob/master/src/data_manager/gen_connection_data.py

The connection matrix is quantized to 1 byte/cell (cost // RESOLUTION,
capped at 254) to keep the asset a reasonable size; this mirrors Mozc's own
shipped 1-byte-cost quantization scheme.

Usage:
  1. Download `dictionary00.txt`-`dictionary09.txt` and `connection_single_column.txt`
  2. python3 tools/build_ja_dictionary.py <dir> [--max-cost 4000]

Note:
Deduplication key must be (surface, left_id, right_id), as text very often has different
usage for different parts of speech. For example, が is both a case particle (背が高い)
and a conjunctive particle (だが).
"""
import argparse
import glob
import os
import struct
import sys

MAGIC = b"JAKD"
VERSION = 2
RESOLUTION = 64
INVALID_BYTE = 255

# read dictionary.txt files with some de-duplication logic
def load_entries(source_dir: str, max_cost: int) -> dict[str, list[tuple[int, int, int, str]]]:
    """returns reading -> list of (cost, left_id, right_id, surface)"""
    by_reading: dict[str, list[tuple[int, int, int, str]]] = {}
    files = sorted(glob.glob(os.path.join(source_dir, "dictionary*.txt")))
    if not files:
        sys.exit(f"no dictionary*.txt files found in {source_dir}")
    for path in files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 5:
                    continue
                reading, left_id_s, right_id_s, cost_s, surface = parts
                cost = int(cost_s)
                if cost > max_cost:
                    continue
                if not reading or not surface:
                    continue
                left_id, right_id = int(left_id_s), int(right_id_s)
                candidates = by_reading.setdefault(reading, [])
                # de-duplication (surface, left_id, right_id) -> see module doc
                for i, (existing_cost, existing_l, existing_r, existing_surface) in enumerate(candidates):
                    if existing_surface == surface and existing_l == left_id and existing_r == right_id:
                        if cost < existing_cost:
                            candidates[i] = (cost, left_id, right_id, surface)
                        break
                else:
                    candidates.append((cost, left_id, right_id, surface))
    return by_reading

# read connection_single_column.txt
def load_connection_matrix(source_dir: str) -> tuple[int, bytearray]:
    path = os.path.join(source_dir, "connection_single_column.txt")
    if not os.path.isfile(path):
        sys.exit(f"missing {path}")
    with open(path, encoding="ascii") as f:
        pos_size = int(f.readline())
        matrix = bytearray(pos_size * pos_size)
        for i, line in enumerate(f):
            cost = int(line)
            q = cost // RESOLUTION
            matrix[i] = q if q < INVALID_BYTE else INVALID_BYTE - 1
    if len(matrix) != pos_size * pos_size:
        sys.exit(f"connection matrix size mismatch: expected {pos_size * pos_size}, got {len(matrix)}")
    return pos_size, matrix

# output dictionary to binary format
def write_dictionary(by_reading: dict[str, list[tuple[int, int, int, str]]],
                      pos_size: int, matrix: bytearray, out_path: str) -> None:
    readings = sorted(by_reading.keys())
    with open(out_path, "wb") as out:
        out.write(MAGIC)
        out.write(struct.pack(">B", VERSION))
        out.write(struct.pack(">I", len(readings)))
        for reading in readings:
            candidates = sorted(by_reading[reading])[:8]  # keep at most 8 candidates per reading
            reading_bytes = reading.encode("utf-8")
            out.write(struct.pack(">H", len(reading_bytes)))
            out.write(reading_bytes)
            out.write(struct.pack(">H", len(candidates)))
            for cost, left_id, right_id, surface in candidates:
                surface_bytes = surface.encode("utf-8")
                out.write(struct.pack(">iHH", cost, left_id, right_id))
                out.write(struct.pack(">H", len(surface_bytes)))
                out.write(surface_bytes)
        out.write(struct.pack(">HH", pos_size, RESOLUTION))
        out.write(matrix)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", help="directory containing Mozc's dictionary*.txt and connection_single_column.txt")
    parser.add_argument("--max-cost", type=int, default=5000,
                         help="keep only dictionary entries with cost <= this (lower cost = more common word)")
    parser.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "..", "app", "src", "main", "assets", "dicts", "ja", "kana_kanji.dat"))
    args = parser.parse_args()

    by_reading = load_entries(args.source_dir, args.max_cost)
    pos_size, matrix = load_connection_matrix(args.source_dir)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    write_dictionary(by_reading, pos_size, matrix, args.out)
    total_candidates = sum(len(v) for v in by_reading.values())
    size_kb = os.path.getsize(args.out) / 1024
    print(f"wrote {args.out}: {len(by_reading)} readings, {total_candidates} candidates, "
          f"{pos_size}x{pos_size} connection matrix, {size_kb:.1f} KiB")


if __name__ == "__main__":
    main()
