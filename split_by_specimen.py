"""
Split a landmarks CSV into one CSV file per specimen.

Input CSV is expected to have at least a 'specimen' column
(plus, in this case, 'landmark', 'x', 'y' columns), e.g.:

    "specimen","landmark","x","y"
    "LACM ENT 592702_L",1,746.49,263.80
    ...

For each unique value in 'specimen', this writes a separate CSV file
containing only that specimen's rows.

Usage:
    python split_by_specimen.py input.csv output_dir
"""

import csv
import os
import sys


def split_by_specimen(input_csv: str, output_dir: str, specimen_col: str = "specimen") -> None:
    os.makedirs(output_dir, exist_ok=True)

    with open(input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        if fieldnames is None or specimen_col not in fieldnames:
            raise ValueError(
                f"Column '{specimen_col}' not found in input CSV. "
                f"Available columns: {fieldnames}"
            )

        rows_by_specimen = {}
        for row in reader:
            specimen = row[specimen_col]
            rows_by_specimen.setdefault(specimen, []).append(row)

    for specimen, rows in rows_by_specimen.items():
        stem = os.path.splitext(specimen)[0]  # strips .jpg / .png / etc if present in specimen name
        out_path = os.path.join(output_dir, f"{stem}.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as out_f:
            writer = csv.DictWriter(out_f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows -> {out_path}")

    print(f"\nDone. {len(rows_by_specimen)} specimen files written to '{output_dir}'.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python split_by_specimen.py <input.csv> <output_dir>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_dir = sys.argv[2]
    split_by_specimen(input_csv, output_dir)