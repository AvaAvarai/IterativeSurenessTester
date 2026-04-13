#!/usr/bin/env python3
"""
1) Min–max normalize Fisher Iris using min/max over **all 150 rows**.
2) Fit HyperblockClassifierDV on that full normalized frame (same hb_dv as BAP).
3) Export hyperblocks (BAP CSV format) and drop raw Iris rows inside **any** HB
   (--minmax-scope all), then run BAP stage-2 on the remainder.
4) Write coverage summaries: stage-1 HBs on full Iris (same norm), stage-2 HBs on
   full Iris in **stage-2 BAP** train min–max space (filtered CSV train slice).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from bap import (
    compute_hb_largest_per_class_coverage,
    hyperblock_edges_tabular_export,
    load_csv,
    split_data,
)
from hb_dv import Hyperblock, HyperblockClassifierDV


class _HBClf:
    __slots__ = ("hyperblocks_",)

    def __init__(self, hbs: list[Hyperblock]) -> None:
        self.hyperblocks_ = hbs


def _inside_any(hbs: list[Hyperblock], x: np.ndarray) -> bool:
    return any(hb.contains(x) for hb in hbs)


def _inside_own_label(hbs: list[Hyperblock], x: np.ndarray, label: str) -> bool:
    for hb in hbs:
        if str(hb.class_) != str(label):
            continue
        if hb.contains(x):
            return True
    return False


def _fmt_scope_lines(
    title: str,
    X: pd.DataFrame,
    y: pd.Series,
    hbs: list[Hyperblock],
) -> list[str]:
    feat_cols = list(X.columns)
    Xa = np.asarray(X[feat_cols].values, dtype=float)
    ys = np.asarray(y.astype(str))
    classes = sorted(np.unique(ys).tolist())
    n = len(X)
    clf = _HBClf(hbs)
    min_cov, per_frac, per_n, n_hb = compute_hb_largest_per_class_coverage(clf, X, y)
    w_cov = sum(per_n.values()) / n if n else 0.0
    n_any = int(sum(1 for i in range(n) if _inside_any(hbs, Xa[i])))
    lines = [
        f"### {title}",
        f"  Rows: {n}",
        f"  Hyperblocks: {n_hb}",
        f"  BAP min (largest HB per class): {min_cov:.6f}",
        f"  BAP weighted coverage: {w_cov:.6f}",
        f"  Points inside any HB: {n_any} ({100 * n_any / n:.2f}% of rows)",
        "  Largest-HB fraction per class: "
        + ", ".join(f"{k}={v:.4f}" for k, v in sorted(per_frac.items())),
        "  Own-label containment (in ≥1 same-class HB):",
    ]
    for c in classes:
        mask = ys == c
        idx = np.where(mask)[0]
        n_c = int(len(idx))
        n_in = int(sum(1 for i in idx if _inside_own_label(hbs, Xa[i], c)))
        lines.append(
            f"    {c}: {n_in}/{n_c} in ({100 * n_in / n_c:.2f}%), outside {n_c - n_in}"
        )
    lines.append("")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="All-data hb_dv → exclude → BAP stage 2 + stats.")
    ap.add_argument(
        "--iris",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent
        / "machine learning datasets"
        / "default"
        / "fisher_iris.csv",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Pipeline output root (default: results/iris_all_data_hb_stage2_<UTC ts>).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-id", type=int, default=1)
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("-n", "--iterations", type=int, default=3000, dest="n")
    ap.add_argument(
        "--inside-rule",
        choices=("any", "own-label"),
        default="own-label",
        help="Passed to iris_exclude_hb_union: any = inside any HB; own-label = inside a same-class HB "
        "(own-label keeps more points for stage 2 when HBs of different classes overlap).",
    )
    ap.add_argument(
        "--skip-bap",
        action="store_true",
        help="Only fit all-data HBs + exclusion + stage-1 stats (no stage-2 BAP).",
    )
    ap.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python for bap.py subprocess.",
    )
    args = ap.parse_args()

    iris = args.iris.resolve()
    if not iris.is_file():
        raise SystemExit(f"Iris CSV not found: {iris}")

    out_root = args.out_dir
    if out_root is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_root = Path(__file__).resolve().parent / "results" / f"iris_all_data_hb_stage2_{ts}"
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).resolve().parent
    X_raw, y = load_csv(str(iris))
    feat_cols = list(X_raw.columns)
    mn_all = X_raw.min()
    mx_all = X_raw.max()
    rng_all = (mx_all - mn_all).replace(0, 1)
    X_norm = (X_raw - mn_all) / rng_all

    clf = HyperblockClassifierDV(random_state=args.seed)
    clf.fit(X_norm, y)
    edges = clf.get_hyperblock_edges()
    edges_df = hyperblock_edges_tabular_export(edges, feat_cols)
    hb_path = out_root / "all_data_hb_dv_hyperblocks.csv"
    edges_df.to_csv(hb_path, index=False)

    cases_raw = X_raw.copy()
    cases_raw["class"] = y.values
    cases_path = out_root / "all_data_hb_dv_cases.csv"
    cases_raw.to_csv(cases_path, index=False)

    (out_root / "normalization.txt").write_text(
        "Hyperblocks and stage-1 stats use min–max from **all** Iris rows (same mn/mx applied to every row).\n"
        f"Features: {feat_cols}\n"
        f"n_hyperblocks: {len(clf.hyperblocks_)}\n",
        encoding="utf-8",
    )

    hbs = list(clf.hyperblocks_)

    stats1 = out_root / "stage1_all_data_hb_full_iris_coverage.txt"
    lines = [
        "Stage 1: hb_dv fit on all 150 rows (full-dataset min–max norm)",
        f"hyperblocks: {hb_path.name}",
        "",
    ]
    lines.extend(_fmt_scope_lines("Full Iris (150), all-data min–max space", X_norm, y, hbs))
    stats1.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {hb_path} ({len(hbs)} HBs)")
    print(f"Wrote {stats1}")

    filtered = out_root / "iris_excluding_all_data_hbs.csv"
    subprocess.run(
        [
            str(args.python),
            str(script_dir / "iris_exclude_hb_union.py"),
            "--iris",
            str(iris),
            "--hyperblocks",
            str(hb_path),
            "--output",
            str(filtered),
            "--seed",
            str(args.seed),
            "--split-id",
            str(args.split_id),
            "--train-ratio",
            str(args.train_ratio),
            "--minmax-scope",
            "all",
            "--inside-rule",
            args.inside_rule,
        ],
        check=True,
        cwd=str(script_dir),
    )

    if args.skip_bap:
        print("Skipped stage-2 BAP (--skip-bap).")
        return

    n_kept = len(pd.read_csv(filtered))
    if n_kept == 0:
        print("No rows left after exclusion; skipping BAP.")
        return

    subprocess.run(
        [
            str(args.python),
            str(script_dir / "bap.py"),
            "--train",
            str(filtered),
            "--testing",
            "split",
            "--split",
            f"{args.train_ratio},{1.0 - args.train_ratio}",
            "--classifier",
            "hb_dv",
            "--k",
            "3",
            "--distance",
            "euclidean",
            "-t",
            "0.95",
            "--hb-coverage-t",
            "0.95",
            "--direction",
            "forward",
            "--splits",
            "1",
            "-n",
            str(args.n),
            "-m",
            "5",
            "--sampling",
            "random",
            "--seed",
            str(args.seed),
            "-o",
            str(out_root),
        ],
        check=True,
        cwd=str(script_dir),
    )

    # Newest bap_* under out_root
    bap_dirs = sorted(out_root.glob("bap_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not bap_dirs:
        raise SystemExit("BAP finished but no bap_* directory found.")
    stage2_dir = bap_dirs[0]
    cfg_txt = stage2_dir / "config.txt"
    train_csv = None
    for line in cfg_txt.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("train:"):
            train_csv = Path(s.split(":", 1)[1].strip())
            break
    if train_csv is None or not train_csv.is_file():
        raise SystemExit(f"Could not resolve stage-2 train CSV from {cfg_txt}")

    # Import here so matplotlib-only users still get stage1 without venv
    from plot_parallel_hb_coverage import (  # pylint: disable=import-outside-toplevel
        parse_hyperblocks_csv,
        pick_best_run,
        split_seed_for_bap,
    )

    split_seed = split_seed_for_bap(args.seed, args.split_id)
    n_train_s2 = int(round(args.train_ratio * len(pd.read_csv(train_csv))))
    s2_split = stage2_dir / f"split_{args.split_id}"
    cases_s2, _, meta = pick_best_run(s2_split, n_train_total=n_train_s2)
    hb_s2_path = cases_s2.with_name(cases_s2.stem + "_hyperblocks.csv")
    hbs_s2 = parse_hyperblocks_csv(hb_s2_path)

    X_train_filt, _ytr, _Xte, _yte = split_data(
        *load_csv(str(train_csv)), args.train_ratio, 1.0 - args.train_ratio, split_seed
    )
    mn_s2 = X_train_filt.min()
    mx_s2 = X_train_filt.max()
    rng_s2 = (mx_s2 - mn_s2).replace(0, 1)
    X_all_raw, y_all = load_csv(str(iris))
    X_full_s2_space = (X_all_raw - mn_s2) / rng_s2

    stats2 = out_root / "stage2_hb_full_iris_coverage_stage2_norm.txt"
    lines2 = [
        "Stage 2: best converged BAP run on remainder after all-data HB exclusion",
        f"stage2_results: {stage2_dir.name}",
        f"best_cases: {cases_s2.name}",
        f"pick_best_run meta: {meta}",
        "",
    ]
    lines2.extend(
        _fmt_scope_lines(
            "Full Fisher Iris (150), min–max from stage-2 train slice of filtered CSV",
            X_full_s2_space,
            y_all,
            hbs_s2,
        )
    )
    stats2.write_text("\n".join(lines2), encoding="utf-8")
    print(f"Wrote {stats2}")
    print(f"Pipeline root: {out_root}")


if __name__ == "__main__":
    main()
