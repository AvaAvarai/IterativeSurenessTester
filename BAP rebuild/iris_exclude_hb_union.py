#!/usr/bin/env python3
"""
Drop all Iris rows that lie inside any of a set of axis-aligned hyperblocks (BAP export).

Hyperblocks are interpreted in the same normalized space as BAP: stratified train split,
then min–max from the training portion applied to every row. Output is raw feature values
so a fresh BAP run can load the CSV and normalize again.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from bap import DEFAULTS, EXPORT_CLASS_COL, load_csv, split_data
from hb_dv import Hyperblock

_HB_CLASS_RE = re.compile(r"^(.+)__HB(\d+)__(bottom|top)$")


def parse_hyperblocks_csv(hb_path: Path) -> list[Hyperblock]:
    df = pd.read_csv(hb_path)
    class_col = EXPORT_CLASS_COL
    feat_cols = [c for c in df.columns if c != class_col]
    by_hb: dict[tuple[int, str], dict[str, pd.Series]] = {}
    for _, row in df.iterrows():
        m = _HB_CLASS_RE.match(str(row[class_col]))
        if not m:
            continue
        base_cls, hid, kind = m.group(1), int(m.group(2)), m.group(3)
        key = (hid, base_cls)
        by_hb.setdefault(key, {})[kind] = row
    hbs: list[Hyperblock] = []
    for (hid, base_cls), parts in sorted(by_hb.items()):
        if "bottom" not in parts or "top" not in parts:
            continue
        bot = parts["bottom"]
        top = parts["top"]
        mins = np.array([float(bot[c]) for c in feat_cols], dtype=float)
        maxs = np.array([float(top[c]) for c in feat_cols], dtype=float)
        hbs.append(Hyperblock(mins, maxs, base_cls))
    return hbs


def inside_any_hyperblock(hbs: list[Hyperblock], x: np.ndarray) -> bool:
    return any(hb.contains(x) for hb in hbs)


def inside_own_class_hyperblock(hbs: list[Hyperblock], x: np.ndarray, label: str) -> bool:
    """True if x lies in at least one HB whose class label matches ``label``."""
    for hb in hbs:
        if str(hb.class_) != str(label):
            continue
        if hb.contains(x):
            return True
    return False


def split_seed_for_bap(config_seed: int, split_id: int) -> int:
    return (config_seed + split_id * DEFAULTS["split_seed_multiplier"]) % DEFAULTS["seed_modulus"]


def main() -> None:
    p = argparse.ArgumentParser(description="Iris CSV minus points inside any listed HB.")
    p.add_argument("--iris", type=Path, required=True, help="Source Iris tabular CSV.")
    p.add_argument("--hyperblocks", type=Path, required=True, help="BAP *_hyperblocks.csv export.")
    p.add_argument("--output", type=Path, required=True, help="Write filtered CSV here.")
    p.add_argument("--seed", type=int, default=42, help="BAP config seed (with split-id).")
    p.add_argument("--split-id", type=int, default=1, help="BAP split_* index for split seed.")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument(
        "--minmax-scope",
        choices=("train", "all"),
        default="train",
        help="train: min–max from stratified train slice (matches default BAP). "
        "all: min–max from every row of the Iris file (use when hyperblocks were "
        "fit in full-dataset normalization space).",
    )
    p.add_argument(
        "--inside-rule",
        choices=("any", "own-label"),
        default="any",
        help="any: drop if inside any HB (union over all classes). "
        "own-label: drop only if inside an HB labeled with that row's class.",
    )
    args = p.parse_args()

    test_ratio = 1.0 - args.train_ratio
    split_seed = split_seed_for_bap(args.seed, args.split_id)

    df = pd.read_csv(args.iris)
    X, y = load_csv(str(args.iris))
    if args.minmax_scope == "train":
        X_train, _yt, _Xte, _yte = split_data(X, y, args.train_ratio, test_ratio, split_seed)
        mn = X_train.min()
        mx = X_train.max()
    else:
        mn = X.min()
        mx = X.max()
    rng = (mx - mn).replace(0, 1)
    X_norm = (X - mn) / rng
    Xa = np.asarray(X_norm.values, dtype=float)

    hbs = parse_hyperblocks_csv(args.hyperblocks)
    if not hbs:
        raise SystemExit(f"No hyperblocks parsed from {args.hyperblocks}")

    if args.inside_rule == "any":
        inside = np.array([inside_any_hyperblock(hbs, Xa[i]) for i in range(len(Xa))], dtype=bool)
    else:
        y_arr = np.asarray(y.astype(str))
        inside = np.array(
            [inside_own_class_hyperblock(hbs, Xa[i], y_arr[i]) for i in range(len(Xa))],
            dtype=bool,
        )
    keep = ~inside
    out = df.loc[keep].reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    dropped = int(np.sum(inside))
    print(f"Hyperblocks: {len(hbs)}  Rows in: {len(df)}  Dropped (inside any HB): {dropped}  Kept: {len(out)}")
    print(f"Wrote {args.output.resolve()}")
    if EXPORT_CLASS_COL in out.columns:
        print("Class counts (kept):", out[EXPORT_CLASS_COL].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
