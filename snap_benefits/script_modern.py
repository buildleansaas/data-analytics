#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modernized labels variant of per‑capita SNAP participation by detailed ancestry
using ACS PUMS 2023 1‑year.

What changes vs. the base script:
  - After attaching Census labels, merge legacy/synonymous labels into a single category
    for reporting clarity. Specifically, exact matches among
    {"Afro American", "Afro-American", "African American", "African-American", "Black", "Negro"}
    are normalized to: "African American".
  - We then aggregate results by this modernized label and recompute the rate.
  - Other labels (e.g., specific African ancestries like "Somali", "Nigerian", etc.) remain distinct.

Inputs (unzipped CSVs from Census "csv_hus.zip" and "csv_pus.zip"):
  - psam_husa.csv, psam_husb.csv  (HOUSEHOLD files; need SERIALNO, FS)
  - psam_pusa.csv, psam_pusb.csv  (PERSON files; need SERIALNO, PWGTP, ANC1P, ANC2P)

Outputs:
  - out/snap_per_capita_ancestry_modern.csv  (ancestry_label|people|people_in_snap_hh|rate)
  - out/snap_per_capita_modern_topN.png      (ranked bar chart)
  - out/snap_per_capita_modern_all.png       (all groups)
"""

import argparse
import os
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import io
import urllib.request


def parse_args():
    ap = argparse.ArgumentParser(description="Per-capita SNAP by ancestry with modernized label categories.")
    ap.add_argument("--household", nargs="+", required=True,
                    help="Paths to household CSV(s), e.g. psam_husa.csv psam_husb.csv")
    ap.add_argument("--person", nargs="+", required=True,
                    help="Paths to person CSV(s), e.g. psam_pusa.csv psam_pusb.csv")
    ap.add_argument("--anc-labels", default="auto",
                    help="Optional CSV with columns value,label for ancestry codes. Use 'auto' to fetch from Census.")
    ap.add_argument("--min-people", type=float, default=5e4,
                    help="Minimum weighted persons to include (default: 50,000). Use 0 for no filter.")
    ap.add_argument("--topn", type=int, default=30,
                    help="Top-N groups to plot (default: 30). Use 0 to plot all groups.")
    ap.add_argument("--outdir", default="out", help="Output directory (default: out)")
    ap.add_argument("--chunksize", type=int, default=500_000,
                    help="CSV chunksize for streaming (default: 500k rows).")
    return ap.parse_args()


H_USECOLS = ["SERIALNO", "FS"]
P_USECOLS = ["SERIALNO", "PWGTP", "ANC1P", "ANC2P"]


def load_household_snap(paths: List[str], chunksize: int) -> pd.Series:
    acc: List[pd.Series] = []
    for p in paths:
        for chunk in pd.read_csv(p, usecols=H_USECOLS, chunksize=chunksize,
                                 dtype={"SERIALNO": "string", "FS": "Int64"}):
            snap = (chunk["FS"] == 1)
            snap.index = chunk["SERIALNO"].astype("string")
            acc.append(snap)
    if not acc:
        raise RuntimeError("No household rows loaded.")
    hh_snap = pd.concat(acc)
    hh_snap = hh_snap[~hh_snap.index.duplicated(keep="first")]
    hh_snap.name = "snap_hh"
    return hh_snap


def ancestry_code_from_row(df: pd.DataFrame) -> np.ndarray:
    a1 = df["ANC1P"].fillna(0).astype("Int64")
    a2 = df["ANC2P"].fillna(0).astype("Int64")
    code = a1.copy()
    code = code.where(code > 0, a2)
    code = code.where(code > 0, -1)
    return code.astype("int32").to_numpy()


def aggregate_people_by_ancestry(person_paths: List[str],
                                 hh_snap: pd.Series,
                                 chunksize: int) -> Tuple[Dict[int, float], Dict[int, float]]:
    people_by_anc: Dict[int, float] = {}
    people_snap_by_anc: Dict[int, float] = {}

    for p in person_paths:
        for chunk in pd.read_csv(p, usecols=P_USECOLS, chunksize=chunksize,
                                 dtype={"SERIALNO": "string", "PWGTP": "float64", "ANC1P": "Int64", "ANC2P": "Int64"}):
            snap = (hh_snap.reindex(chunk["SERIALNO"].astype("string"))
                    .fillna(False).astype(bool).to_numpy())
            anc = ancestry_code_from_row(chunk)
            w = chunk["PWGTP"].to_numpy()
            codes, inv = np.unique(anc, return_inverse=True)
            wt_sum = np.bincount(inv, weights=w).astype(float)
            wt_snap = np.bincount(inv, weights=(w * snap)).astype(float)
            for i, code in enumerate(codes.tolist()):
                people_by_anc[code] = people_by_anc.get(code, 0.0) + wt_sum[i]
                people_snap_by_anc[code] = people_snap_by_anc.get(code, 0.0) + wt_snap[i]
    return people_by_anc, people_snap_by_anc


def load_labels(path: str) -> pd.DataFrame:
    lab = pd.read_csv(path, dtype={"value": "int32", "label": "string"})
    lab = lab.dropna(subset=["value"]).rename(columns=str.lower)
    if "value" not in lab.columns or "label" not in lab.columns:
        raise ValueError("Label file must have columns: value,label")
    return lab[["value", "label"]].drop_duplicates()


def load_labels_auto_2023() -> pd.DataFrame:
    url = "https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2023.csv"
    with urllib.request.urlopen(url) as resp:
        raw = resp.read()
    df = pd.read_csv(io.BytesIO(raw), header=None, dtype=str, keep_default_na=False,
                     engine="python", names=list(range(8)))
    df = df[(df[0] == "VAL") & (df[1].isin(["ANC1P", "ANC2P"]))][[4, 6]].rename(columns={4: "value", 6: "label"})
    df["value"] = df["value"].astype(str).str.replace('"', '', regex=False)
    df["label"] = df["label"].astype(str).str.replace('"', '', regex=False)
    df = df[df["value"].str.len() > 0].copy()
    df["value"] = df["value"].astype(int)
    df = df.drop_duplicates("value")
    df = pd.concat([df, pd.DataFrame({"value": [-1], "label": ["Unclassified (no detailed ancestry)"]})], ignore_index=True)
    return df


def modernize_label(s: str) -> str:
    if s is None:
        return ""
    t = s.strip()
    synonyms = {
        "Afro American",
        "Afro-American",
        "African American",
        "African-American",
        "Black",
        "Negro",
    }
    return "African American" if t in synonyms else t


def plot_bar(df: pd.DataFrame, out_png: str, title: str, topn: int):
    sort = df.sort_values("rate", ascending=False)
    is_all = (topn is None or topn <= 0)
    top = sort if is_all else sort.head(topn)
    n = len(top)
    height = max(8, min(0.28 * n, 180))
    width = 10 if n <= 120 else 12
    fig, ax = plt.subplots(figsize=(width, height))
    plot_df = top.sort_values("rate", ascending=True)
    bars = ax.barh(plot_df["ancestry_label"], plot_df["rate"], color="#4C78A8")
    ax.set_xlabel("Share of people in SNAP-recipient households")
    ax.set_ylabel("Ancestry")
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.grid(axis="x", alpha=0.2)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    max_rate = float(plot_df["rate"].max()) if n else 0.0
    ax.set_xlim(0, min(1.0, max_rate * 1.10 + 1e-6))
    annotate = True if is_all else (n <= 120)
    font_size = 8 if n > 120 else 9
    if annotate:
        for rect, val in zip(bars, plot_df["rate" ].to_list()):
            width = rect.get_width()
            y = rect.get_y() + rect.get_height()/2
            label = f"{val*100:.1f}%"
            if width >= max_rate * 0.25:
                ax.text(width - (max_rate * 0.01), y, label, va="center", ha="right", color="white", fontsize=font_size)
            else:
                ax.text(width + (max_rate * 0.01), y, label, va="center", ha="left", color="#333333", fontsize=font_size)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    hh_snap = load_household_snap(args.household, args.chunksize)
    people_by_anc, people_snap_by_anc = aggregate_people_by_ancestry(args.person, hh_snap, args.chunksize)

    rows = []
    for code, tot in people_by_anc.items():
        if tot <= 0:
            continue
        tot_snap = people_snap_by_anc.get(code, 0.0)
        rows.append((code, tot, tot_snap, (tot_snap / tot)))
    res = pd.DataFrame(rows, columns=["ancestry_code", "people", "people_in_snap_hh", "rate"])

    # Attach labels (auto by default)
    lab = None
    if args.anc_labels and args.anc_labels.lower() != "auto":
        lab = load_labels(args.anc_labels)
    else:
        try:
            lab = load_labels_auto_2023()
        except Exception:
            lab = None
    if lab is not None:
        res = res.merge(lab, left_on="ancestry_code", right_on="value", how="left")
        res["ancestry_label"] = res["label"].fillna(res["ancestry_code"].astype(str))
        res = res.drop(columns=["value", "label"])
    else:
        res["ancestry_label"] = res["ancestry_code"].astype(str)

    # Modernize labels and aggregate by label
    res["ancestry_label"] = res["ancestry_label"].astype(str).map(modernize_label)
    res = res.groupby("ancestry_label", as_index=False).agg({
        "people": "sum",
        "people_in_snap_hh": "sum",
    })
    res["rate"] = res["people_in_snap_hh"] / res["people"]

    # Filter small groups
    res = res[res["people"] >= args.min_people].copy()

    out_csv = os.path.join(args.outdir, "snap_per_capita_ancestry_modern.csv")
    res.sort_values("rate", ascending=False).to_csv(out_csv, index=False)

    plot_bar(res, os.path.join(args.outdir, "snap_per_capita_modern_topN.png"),
             "Per-capita SNAP by ancestry (modernized labels) — Top N", args.topn)
    plot_bar(res, os.path.join(args.outdir, "snap_per_capita_modern_all.png"),
             "Per-capita SNAP by ancestry (modernized labels) — All", topn=0)

    print(f"Saved: {out_csv}")
    print(res.sort_values('rate', ascending=False).head(10).assign(rate_pct=lambda d: (d['rate']*100).round(1)))


if __name__ == "__main__":
    main()

