"""One-off conversion of the IDS2025 workbook to Parquet (M1, Phase 1 step 1).

Reads data/raw/IDS2025.xlsx once (read-only, streamed), strips whitespace from the
column names, adds row_index (0..N-1 in original sheet order) and writes
data/processed/ids2025_raw.parquet. No values are changed and nothing is cleaned:
cleaning is Phase 1 steps 4-7 (§3.2.4).

Type rule (so that no value changes): a column whose cells are all numbers keeps a
numeric dtype (int64 if every cell is an integer, else float64). A column that
contains any text cell (e.g. "Infinity", "NaN" in the rate features) is stored as
text exactly: text cells verbatim and numeric cells as repr(), which round-trips
to the identical float. Phase 1 step 4 parses these columns.

Also writes the source description and the label-run statistics (order check,
Phase 1 step 3; decides LSTM formulation B, plan §4.3) to results/prep_report.json.

The source workbook is never modified (CLAUDE.md): it is opened read-only and its
SHA-256 is checked before and after.

Usage:  python -m nids.prep.convert_xlsx [--config config.yaml]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import statistics
from pathlib import Path

import openpyxl
import pandas as pd
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[3]
LABEL_COL = "newLabel"
WINDOW_SIZES = (5, 10, 20)  # LSTM W grid, Table 3.11 / plan §4.3


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _build_column(values: list) -> tuple[pd.Series, str]:
    """Return the column with a lossless dtype, plus a note of how it was stored."""
    if all(_is_int(v) for v in values):
        return pd.Series(values, dtype="int64"), "int64"
    if all(_is_int(v) or isinstance(v, float) for v in values):
        return pd.Series(values, dtype="float64"), "float64"
    # Mixed/text column: keep text verbatim, numbers as exact repr; empty cells -> null.
    out = [None if v is None else (v if isinstance(v, str) else repr(v)) for v in values]
    return pd.Series(out, dtype="string"), "string (contains text cells)"


def read_sheet(xlsx: Path) -> tuple[pd.DataFrame, dict]:
    """Stream the workbook's only sheet into a DataFrame without altering values."""
    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    try:
        if len(wb.sheetnames) != 1:
            raise ValueError(f"expected one sheet, found {wb.sheetnames}")
        ws = wb[wb.sheetnames[0]]
        rows = ws.iter_rows(values_only=True)
        raw_header = next(rows)
        header = [str(h).strip() for h in raw_header]
        if len(set(header)) != len(header):
            raise ValueError("duplicate column names after stripping whitespace")
        cols: list[list] = [[] for _ in header]
        n_rows = 0
        for row in rows:
            if row is None or all(v is None for v in row):
                continue  # trailing blank rows that some writers emit
            if len(row) > len(header) and any(v is not None for v in row[len(header):]):
                raise ValueError(f"data row {n_rows + 1} has cells beyond the header")
            for i in range(len(header)):
                cols[i].append(row[i] if i < len(row) else None)
            n_rows += 1
        meta = {
            "sheet_names": wb.sheetnames,
            "sheet": ws.title,
            "dimension": ws.calculate_dimension(),
            "excel_data_rows": n_rows,
            "excel_columns": len(header),
            "renamed_by_strip": {r: h for r, h in zip(raw_header, header) if r != h},
        }
    finally:
        wb.close()

    data, dtypes = {}, {}
    for name, values in zip(header, cols):
        data[name], dtypes[name] = _build_column(values)
    df = pd.DataFrame(data)
    df.insert(0, "row_index", pd.RangeIndex(n_rows, dtype="int64"))
    meta["stored_dtypes"] = {k: v for k, v in dtypes.items() if v.startswith("string")}
    meta["text_cells"] = {
        k: df[k][~df[k].str.fullmatch(r"[-+0-9.eE]+").fillna(False)].value_counts().to_dict()
        for k in meta["stored_dtypes"]
    }
    return df, meta


def label_run_stats(labels: pd.Series) -> dict:
    """Run lengths of the label in row order (Phase 1 step 3 order check)."""
    lab = labels.to_numpy()
    n = len(lab)
    runs: list[tuple[str, int]] = []
    for v in lab:
        if runs and runs[-1][0] == v:
            runs[-1] = (v, runs[-1][1] + 1)
        else:
            runs.append((v, 1))
    lengths = [r for _, r in runs]
    per_class = collections.defaultdict(list)
    for v, r in runs:
        per_class[v].append(r)
    return {
        "n_rows": n,
        "n_runs": len(runs),
        "p_next_same_label": round((n - len(runs)) / (n - 1), 6),
        "run_length": {"mean": round(statistics.mean(lengths), 1),
                       "median": statistics.median(lengths), "max": max(lengths)},
        "per_class": {
            c: {"rows": sum(rs), "runs": len(rs), "mean": round(statistics.mean(rs), 1),
                "median": statistics.median(rs), "max": max(rs)}
            for c, rs in sorted(per_class.items(), key=lambda kv: -sum(kv[1]))
        },
        "single_label_window_share": {
            f"W={w}": (round(sum(r - w + 1 for r in lengths if r >= w) / (n - w + 1), 6)
                       if n >= w else None)
            for w in WINDOW_SIZES
        },
        "run_sequence": [[c, r] for c, r in runs],
    }


def update_report(report_path: Path, section: dict) -> None:
    """Merge into results/prep_report.json so later prep steps can add their own keys."""
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    report.update(section)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")


def convert(xlsx: Path, out: Path, report_path: Path) -> dict:
    digest_before = sha256(xlsx)
    df, meta = read_sheet(xlsx)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    # Verify the Parquet file against the Excel counts (+1 column for row_index).
    pf = pq.ParquetFile(out)
    pq_rows, pq_cols = pf.metadata.num_rows, pf.metadata.num_columns
    if pq_rows != meta["excel_data_rows"] or pq_cols != meta["excel_columns"] + 1:
        raise RuntimeError(f"count mismatch: parquet {pq_rows}x{pq_cols}, "
                           f"excel {meta['excel_data_rows']}x{meta['excel_columns']}")
    back = pd.read_parquet(out)
    if not back["row_index"].equals(pd.Series(range(pq_rows), name="row_index")):
        raise RuntimeError("row_index is not 0..N-1 in sheet order")

    if sha256(xlsx) != digest_before:
        raise RuntimeError("source workbook changed during conversion")

    feature_like = [c for c in df.columns if c not in ("row_index", LABEL_COL)]
    summary = {
        "source": {
            "file": str(xlsx.relative_to(ROOT)) if xlsx.is_relative_to(ROOT) else str(xlsx),
            "sha256": digest_before,
            **meta,
            "parquet": str(out.relative_to(ROOT)) if out.is_relative_to(ROOT) else str(out),
            "parquet_rows": pq_rows,
            "parquet_columns": pq_cols,
            "timestamp_present": any("timestamp" in c.lower() for c in df.columns),
            "day_column_present": any(c.lower() in ("day", "source_day", "date") for c in df.columns),
            "identifier_columns_found": [c for c in feature_like if c.lower() in (
                "flow id", "src ip", "source ip", "dst ip", "destination ip",
                "src port", "source port", "dst port", "destination port", "timestamp")],
            "label_column": LABEL_COL,
        },
        "order_check": label_run_stats(back[LABEL_COL]),
    }
    update_report(report_path, summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=ROOT / "config.yaml", type=Path)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    s = convert(ROOT / cfg["paths"]["source_xlsx"], ROOT / cfg["paths"]["raw_parquet"],
                ROOT / "results" / "prep_report.json")
    src, oc = s["source"], s["order_check"]
    print(f"Excel  : {src['excel_data_rows']} rows x {src['excel_columns']} columns "
          f"(sheet '{src['sheet']}', dimension {src['dimension']})")
    print(f"Parquet: {src['parquet_rows']} rows x {src['parquet_columns']} columns "
          f"(= Excel columns + row_index)")
    print(f"Text-bearing columns stored as string: {src['text_cells']}")
    print(f"Timestamp column: {src['timestamp_present']}; day column: {src['day_column_present']}; "
          f"identifiers: {src['identifier_columns_found']}")
    print(f"Label runs: {oc['n_runs']} runs, P(next row same label) = {oc['p_next_same_label']}, "
          f"single-label windows {oc['single_label_window_share']}")
    print(f"SHA-256 unchanged: {src['sha256']}")


if __name__ == "__main__":
    main()
