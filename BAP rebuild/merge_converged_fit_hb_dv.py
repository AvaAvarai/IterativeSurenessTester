#!/usr/bin/env python3
"""
Concatenate every converged case CSV in a BAP split_* directory, drop duplicate rows,
fit HyperblockClassifierDV once on the union, and write:

  - merged unique cases (tabular CSV)
  - hyperblocks in the same bottom/top format as BAP exports

Example:
  python merge_converged_fit_hb_dv.py results/bap_.../split_1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bap import detect_class_column, hyperblock_edges_tabular_export
from hb_dv import HyperblockClassifierDV


def converged_case_csv_paths(split_dir: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(split_dir.glob("converged_exp_*_seed*.csv")):
        if p.name.endswith("_hyperblocks.csv"):
            continue
        out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge converged subsets and fit hb_dv HBs.")
    ap.add_argument("split_dir", type=Path, help="BAP split_N folder with converged_exp_*.csv")
    ap.add_argument(
        "--hyperblocks-out",
        type=Path,
        default=None,
        help="Default: <split_dir>/merged_all_converged_hb_dv_hyperblocks.csv",
    )
    ap.add_argument(
        "--cases-out",
        type=Path,
        default=None,
        help="Default: <split_dir>/merged_all_converged_cases.csv",
    )
    ap.add_argument("--seed", type=int, default=42, help="DV random_state")
    ap.add_argument(
        "--stats",
        action="store_true",
        help="Also write merged_*_coverage_statistics.txt and *_hyperblock_statistics.csv (needs matplotlib).",
    )
    args = ap.parse_args()

    split_dir = args.split_dir
    paths = converged_case_csv_paths(split_dir)
    if not paths:
        raise SystemExit(f"No converged case CSVs found in {split_dir}")

    merged = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    n_before = len(merged)
    class_col = detect_class_column(merged)
    feat_cols = [c for c in merged.columns if c != class_col]
    merged = merged.drop_duplicates(subset=feat_cols + [class_col], keep="first").reset_index(drop=True)
    n_after = len(merged)

    X = merged.loc[:, feat_cols]
    y = merged[class_col].astype(str)

    clf = HyperblockClassifierDV(random_state=args.seed)
    clf.fit(X, y)
    edges = clf.get_hyperblock_edges()
    edges_df = hyperblock_edges_tabular_export(edges, list(feat_cols))

    hb_path = args.hyperblocks_out or (split_dir / "merged_all_converged_hb_dv_hyperblocks.csv")
    cases_path = args.cases_out or (split_dir / "merged_all_converged_cases.csv")
    hb_path.parent.mkdir(parents=True, exist_ok=True)
    edges_df.to_csv(hb_path, index=False)
    merged.to_csv(cases_path, index=False)

    print(f"Converged case files: {len(paths)}")
    print(f"Rows concatenated: {n_before}, unique after dedupe: {n_after}")
    print(f"HBs from hb_dv: {len(clf.hyperblocks_)}")
    print(f"Wrote {hb_path.resolve()}")
    print(f"Wrote {cases_path.resolve()}")

    if args.stats:
        from report_merged_hb_coverage import run_merged_hb_coverage_report

        txt, csv = run_merged_hb_coverage_report(
            cases_path,
            hb_path,
            out_dir=split_dir,
            stem="merged_all_converged_hb_dv",
        )
        print(f"Wrote {txt.resolve()}")
        print(f"Wrote {csv.resolve()}")


if __name__ == "__main__":
    main()
