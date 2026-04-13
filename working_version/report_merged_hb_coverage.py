#!/usr/bin/env python3
"""
Write coverage_statistics.txt + hyperblock_statistics.csv for arbitrary case + hyperblock CSVs
(e.g. merged_all_converged_cases.csv + merged_all_converged_hb_dv_hyperblocks.csv).

Uses the same train min–max space as BAP split_1 (seed 42, split_id 1 by default).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bap import compute_hb_largest_per_class_coverage

from plot_parallel_hb_coverage import (
    load_cases_csv,
    iris_full_in_bap_train_space,
    parse_hyperblocks_csv,
    split_seed_for_bap,
    write_coverage_statistics,
)


class _HBClf:
    __slots__ = ("hyperblocks_",)

    def __init__(self, hbs):
        self.hyperblocks_ = hbs


def run_merged_hb_coverage_report(
    cases_csv: Path,
    hyperblocks_csv: Path,
    *,
    iris_path: Path | None = None,
    out_dir: Path | None = None,
    stem: str = "merged_all_converged_hb_dv",
    seed: int = 42,
    split_id: int = 1,
    train_ratio: float = 0.8,
) -> tuple[Path, Path]:
    iris = iris_path or (
        Path(__file__).resolve().parent.parent.parent
        / "machine learning datasets"
        / "default"
        / "fisher_iris.csv"
    )
    out = out_dir or cases_csv.parent
    out.mkdir(parents=True, exist_ok=True)

    hbs = parse_hyperblocks_csv(hyperblocks_csv)
    if not hbs:
        raise ValueError(f"No hyperblocks in {hyperblocks_csv}")

    X_sub, y_sub = load_cases_csv(cases_csv)
    test_ratio = 1.0 - train_ratio
    split_seed = split_seed_for_bap(seed, split_id)
    X_full, y_full = iris_full_in_bap_train_space(iris, train_ratio, test_ratio, split_seed)

    n_train = int(round(train_ratio * len(y_full)))
    clf = _HBClf(hbs)
    min_cov, _, per_n, _ = compute_hb_largest_per_class_coverage(clf, X_sub, y_sub)
    y_arr = y_sub.astype(str).to_numpy()
    w_cov = sum(per_n.values()) / len(y_arr) if len(y_arr) else 0.0

    extra_meta: dict[str, object] = {
        "bap_seed": seed,
        "cases_csv": str(cases_csv.resolve()),
        "hb_min_class_coverage": float(min_cov),
        "hb_weighted_coverage": float(w_cov),
        "hyperblocks_csv": str(hyperblocks_csv.resolve()),
        "iris_csv": str(iris.resolve()),
        "n_cases": len(y_sub),
        "n_hbs": len(hbs),
        "n_iris_rows": len(y_full),
        "n_train_rows_bap": n_train,
        "split_id": split_id,
        "stratified_split_seed": split_seed,
        "train_ratio": train_ratio,
    }

    return write_coverage_statistics(
        out,
        stem,
        hbs,
        X_sub,
        y_sub,
        X_full,
        y_full,
        extra_meta,
        hyperblocks_csv,
        subset_scope_name="Merged unique cases (this CSV)",
        full_scope_name="Full Iris (train min–max space)",
    )


def main() -> None:
    p = argparse.ArgumentParser(description="HB coverage stats for merged / custom case+HB exports.")
    p.add_argument("--cases", type=Path, required=True, help="Case CSV (features + class).")
    p.add_argument("--hyperblocks", type=Path, required=True, help="BAP-style *_hyperblocks.csv.")
    p.add_argument(
        "--iris",
        type=Path,
        default=None,
        help="Default: computing/machine learning datasets/default/fisher_iris.csv",
    )
    p.add_argument("--out-dir", type=Path, default=None, help="Default: parent of --cases.")
    p.add_argument("--stem", type=str, default="merged_all_converged_hb_dv", help="Output file name prefix.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-id", type=int, default=1)
    p.add_argument("--train-ratio", type=float, default=0.8)
    args = p.parse_args()

    txt, csv = run_merged_hb_coverage_report(
        args.cases,
        args.hyperblocks,
        iris_path=args.iris,
        out_dir=args.out_dir,
        stem=args.stem,
        seed=args.seed,
        split_id=args.split_id,
        train_ratio=args.train_ratio,
    )
    print(f"Wrote {txt.resolve()}")
    print(f"Wrote {csv.resolve()}")


if __name__ == "__main__":
    main()
