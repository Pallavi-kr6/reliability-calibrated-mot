#!/usr/bin/env python
"""Build the v0.2 before/after tables from generated result CSVs; missing values stay explicit."""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import pandas as pd


def read(path):
    p = Path(path)
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def cell(value):
    return "NOT YET RUN" if pd.isna(value) else str(value)


def before_snapshot(root):
    """Keep the pre-change generated benchmark table before `main.py report` refreshes README."""
    snapshot = root / "diagnostics" / "before_main_results.csv"
    if snapshot.exists():
        return pd.read_csv(snapshot)
    readme = Path("README.md")
    rows = []
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        start = text.find("#### Benchmark results (public datasets)")
        end = text.find("#### ", start + 5) if start >= 0 else -1
        section = text[start:end if end >= 0 else None] if start >= 0 else ""
        for line in section.splitlines():
            fields = [part.strip() for part in line.strip().strip("|").split("|")]
            if len(fields) >= 8 and fields[0] == "mot17" and fields[1] in {"b0", "b1", "b2", "b3", "rca"}:
                try:
                    rows.append({"dataset": fields[0], "method": fields[1], "HOTA": float(fields[2]),
                                 "AssA": float(fields[4]), "IDF1": float(fields[5]), "IDSW": float(fields[7])})
                except (ValueError, IndexError):
                    pass
    result = pd.DataFrame(rows, columns=["dataset", "method", "HOTA", "AssA", "IDF1", "IDSW"])
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(snapshot, index=False)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/diagnostics/changes_tables.md")
    args = ap.parse_args()
    root = Path(args.results)
    main_df = read(root / "main_results.csv")
    before_df = before_snapshot(root)
    stress = read(root / "stress.csv")
    abl = read(root / "ablations.csv")
    lines = ["## Generated results tables", "", "### Main metrics before vs after", "",
             "| Dataset | Method | Before HOTA | After HOTA | Before AssA | After AssA | Before IDF1 | After IDF1 | Before IDSW | After IDSW |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    merged = before_df.merge(main_df, on=["dataset", "method"], how="outer", suffixes=("_before", "_after")) if not main_df.empty else before_df
    if merged.empty:
        lines.append("| NOT YET RUN | NOT YET RUN | NOT YET RUN | NOT YET RUN | NOT YET RUN | NOT YET RUN | NOT YET RUN | NOT YET RUN | NOT YET RUN | NOT YET RUN |")
    else:
        for row in merged.to_dict("records"):
            values = [row.get("dataset"), row.get("method")]
            for metric in ("HOTA", "AssA", "IDF1", "IDSW"):
                values += [row.get(f"{metric}_before"), row.get(f"{metric}_after")]
            lines.append("| " + " | ".join(cell(v) for v in values) + " |")
    lines += ["", "### S2 matched false-accept rate", "", "| Baseline | Gap frames | Baseline wrong-ID | RCA re-association at matched FAR | Delta |", "|---|---:|---:|---:|---:|"]
    matched = stress[stress.get("table", pd.Series(dtype=str)) == "S2_matched_far"] if not stress.empty else pd.DataFrame()
    if matched.empty:
        lines.append("| NOT YET RUN | NOT YET RUN | NOT YET RUN | NOT YET RUN | NOT YET RUN |")
    else:
        for row in matched.to_dict("records"):
            values = [cell(row.get(k)) for k in ("baseline", "gap_frames", "baseline_wrong_id_rate")]
            # A present matched-FAR row with empty interpolation columns means the
            # baseline FAR is outside the measured RCA curve; preserve that NaN.
            for key in ("rca_reassoc_at_matched_far", "delta_reassoc"):
                value = row.get(key)
                values.append("NaN" if pd.isna(value) else str(value))
            lines.append("| " + " | ".join(values) + " |")
    lines += ["", "### Ablations", "", "| Variant | HOTA | AssA | IDF1 | IDSW |", "|---|---:|---:|---:|---:|"]
    if abl.empty:
        lines.append("| NOT YET RUN | NOT YET RUN | NOT YET RUN | NOT YET RUN | NOT YET RUN |")
    else:
        for row in abl.to_dict("records"):
            lines.append("| " + " | ".join(cell(row.get(k)) for k in ("variant", "HOTA", "AssA", "IDF1", "IDSW")) + " |")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
