#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute PER-CAPITA SNAP participation by detailed ancestry using ACS PUMS 2023 1-year.

Inputs (unzipped CSVs from Census "csv_hus.zip" and "csv_pus.zip"):
  - psam_husa.csv, psam_husb.csv  (HOUSEHOLD files; need SERIALNO, FS)
  - psam_pusa.csv, psam_pusb.csv  (PERSON files; need SERIALNO, PWGTP, ANC1P, ANC2P)

Definition (person-level, i.e., PER CAPITA):
  rate(ancestry) = SUM(PWGTP for persons of ancestry living in a HH that received SNAP in last 12mo)
                   / SUM(PWGTP for persons of that ancestry)

Outputs:
  - out/snap_per_capita_ancestry.csv  (ancestry_code|ancestry_label|people|people_in_snap_hh|rate)
  - out/snap_per_capita_topN.png      (ranked bar chart, highest per-capita rates)
  - out/snap_per_capita_selected.png  (optional, for --focus "Afghan,Somali,...", if labels are present)

Notes:
  - FS (household SNAP past 12 months): 1 = yes, 2 = no
  - ANC1P/ANC2P are coded integers; script can optionally map to labels via:
      --anc-labels PATH   (CSV with columns: value,label)
    If omitted, output uses numeric codes. You can build a label file from the
    Census PUMS value labels (2023) when convenient.
"""

import argparse
import os
from typing import Dict, Iterable, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import io
import urllib.request

# -------- CLI --------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Compute per-capita SNAP participation by detailed ancestry from ACS PUMS (2023 1-year)."
    )
    ap.add_argument("--household", nargs="+", required=True,
                    help="Paths to household CSV(s), e.g. psam_husa.csv psam_husb.csv")
    ap.add_argument("--person", nargs="+", required=True,
                    help="Paths to person CSV(s), e.g. psam_pusa.csv psam_pusb.csv")
    ap.add_argument("--anc-labels", default=None,
                    help="Optional CSV with columns value,label for ancestry codes. Use 'auto' to fetch labels from Census.")
    ap.add_argument("--min-people", type=float, default=5e4,
                    help="Minimum weighted persons to include in results (default: 50,000).")
    ap.add_argument("--topn", type=int, default=30,
                    help="Top-N groups to plot by per-capita rate (default: 30). Use 0 to plot all groups.")
    ap.add_argument("--outdir", default="out", help="Output directory (default: out)")
    ap.add_argument("--chunksize", type=int, default=500_000,
                    help="CSV chunksize for streaming (default: 500k rows).")
    ap.add_argument("--focus", default=None,
                    help='Comma-separated ancestry labels to highlight in a separate plot (labels loaded via --anc-labels or auto).')
    return ap.parse_args()

# -------- IO helpers --------

H_USECOLS = ["SERIALNO", "FS"]
P_USECOLS = ["SERIALNO", "PWGTP", "ANC1P", "ANC2P"]

def load_household_snap(paths: List[str], chunksize: int) -> pd.Series:
    """
    Stream household files and return a boolean Series indexed by SERIALNO indicating SNAP receipt.
    """
    acc: List[pd.Series] = []
    for p in paths:
        for chunk in pd.read_csv(
            p, usecols=H_USECOLS, chunksize=chunksize, dtype={"SERIALNO": "string", "FS": "Int64"}
        ):
            snap = (chunk["FS"] == 1)
            # SERIALNO in 2023 ACS PUMS can be alphanumeric (e.g., contains 'GQ').
            # Keep as string consistently for joining with person records.
            snap.index = chunk["SERIALNO"].astype("string")
            acc.append(snap)
    if not acc:
        raise RuntimeError("No household rows loaded.")
    hh_snap = pd.concat(acc)
    # If duplicates, keep first (they should be unique per file; concat may stack state slices)
    hh_snap = hh_snap[~hh_snap.index.duplicated(keep="first")]
    hh_snap.name = "snap_hh"
    return hh_snap

def ancestry_code_from_row(df: pd.DataFrame) -> np.ndarray:
    """
    Use ANC1P if >0 else ANC2P if >0 else -1 (Unclassified).
    """
    a1 = df["ANC1P"].fillna(0).astype("Int64")
    a2 = df["ANC2P"].fillna(0).astype("Int64")
    code = a1.copy()
    code = code.where(code > 0, a2)
    code = code.where(code > 0, -1)
    return code.astype("int32").to_numpy()

def aggregate_people_by_ancestry(person_paths: List[str],
                                 hh_snap: pd.Series,
                                 chunksize: int) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Stream person files, left-join household SNAP flag by SERIALNO,
    and accumulate weighted totals per ancestry code.
    Returns (people_by_anc, people_snap_by_anc) dicts.
    """
    people_by_anc: Dict[int, float] = {}
    people_snap_by_anc: Dict[int, float] = {}

    # hh_snap may contain NA where FS is missing; keep as-is and fill during join

    for p in person_paths:
        for chunk in pd.read_csv(
            p, usecols=P_USECOLS, chunksize=chunksize,
            dtype={"SERIALNO": "string", "PWGTP": "float64", "ANC1P": "Int64", "ANC2P": "Int64"},
        ):
            # Join HH SNAP status by SERIALNO (string key)
            snap = (
                hh_snap.reindex(chunk["SERIALNO"].astype("string"))
                .fillna(False)
                .astype(bool)
                .to_numpy()
            )
            anc = ancestry_code_from_row(chunk)
            w = chunk["PWGTP"].to_numpy()

            # Accumulate totals (vectorized pass -> then bincount style)
            # Map ancestry codes to contiguous indices using factorization per chunk
            codes, inv = np.unique(anc, return_inverse=True)
            wt_sum = np.bincount(inv, weights=w).astype(float)
            wt_snap = np.bincount(inv, weights=(w * snap)).astype(float)

            for i, code in enumerate(codes.tolist()):
                people_by_anc[code] = people_by_anc.get(code, 0.0) + wt_sum[i]
                people_snap_by_anc[code] = people_snap_by_anc.get(code, 0.0) + wt_snap[i]

    return people_by_anc, people_snap_by_anc

def load_labels(path: str) -> pd.DataFrame:
    """
    Expect a two-column CSV: value,label (value=int ancestry code).
    """
    lab = pd.read_csv(path, dtype={"value": "int32", "label": "string"})
    lab = lab.dropna(subset=["value"])
    lab = lab.rename(columns=str.lower)
    if "value" not in lab.columns or "label" not in lab.columns:
        raise ValueError("Label file must have columns: value,label")
    return lab[["value", "label"]].drop_duplicates()

def load_labels_auto_2023() -> pd.DataFrame:
    """
    Download and parse the ACS PUMS 2023 data dictionary to extract labels for ANC1P/ANC2P.
    Returns DataFrame with columns value(int), label(str).
    """
    url = "https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2023.csv"
    with urllib.request.urlopen(url) as resp:
        raw = resp.read()
    df = pd.read_csv(
        io.BytesIO(raw), header=None, dtype=str, keep_default_na=False,
        engine="python", names=list(range(7))
    )
    df = df[(df[0] == "VAL") & (df[1].isin(["ANC1P", "ANC2P"]))]
    df = df[[4, 6]].rename(columns={4: "value", 6: "label"})
    df["value"] = df["value"].astype(str).str.strip().str.replace("\"", "", regex=False)
    df["label"] = df["label"].astype(str).str.strip().str.replace("\"", "", regex=False)
    df = df[df["value"].str.len() > 0].copy()
    df["value"] = df["value"].astype(int)
    df = df.drop_duplicates("value")
    df = pd.concat([
        df,
        pd.DataFrame({"value": [-1], "label": ["Unclassified (no detailed ancestry)"]}),
    ], ignore_index=True)
    return df

# -------- Plotting --------

def plot_topn(df: pd.DataFrame, out_png: str, topn: int):
    sort = df.sort_values("rate", ascending=False)
    is_all = (topn is None or topn <= 0)
    top = sort if is_all else sort.head(topn)
    n = len(top)
    # Scale height with bar count; increase width slightly for very long charts
    height = max(8, min(0.28 * n, 180))
    width = 10 if n <= 120 else 12
    fig, ax = plt.subplots(figsize=(width, height))

    # Plot in ascending order so the highest is at the top
    plot_df = top.sort_values("rate", ascending=True)
    bars = ax.barh(plot_df["ancestry_label"], plot_df["rate"], color="#4C78A8")

    # Axes formatting
    ax.set_xlabel("Share of people in SNAP-recipient households")
    ax.set_ylabel("Ancestry")
    ax.set_title("Per-capita SNAP participation by detailed ancestry (ACS PUMS 2023, 1-year)")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.grid(axis="x", alpha=0.2)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # Leave a bit of space on the right for labels
    max_rate = float(plot_df["rate"].max()) if n else 0.0
    ax.set_xlim(0, min(1.0, max_rate * 1.10 + 1e-6))

    # Annotate percent labels beside bars
    # For the ALL-groups chart, always annotate; otherwise annotate when manageable
    annotate = True if is_all else (n <= 120)
    # Slightly reduce font for very long charts
    font_size = 8 if n > 120 else 9
    if annotate:
        for rect, val in zip(bars, plot_df["rate" ].to_list()):
            width = rect.get_width()
            y = rect.get_y() + rect.get_height() / 2
            label = f"{val*100:.1f}%"
            # Place inside for wider bars, otherwise outside to the right
            if width >= max_rate * 0.25:
                ax.text(width - (max_rate * 0.01), y, label,
                        va="center", ha="right", color="white", fontsize=font_size)
            else:
                ax.text(width + (max_rate * 0.01), y, label,
                        va="center", ha="left", color="#333333", fontsize=font_size)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def plot_focus(df: pd.DataFrame, focus_labels: List[str], out_png: str):
    sub = df[df["ancestry_label"].isin([s.strip() for s in focus_labels])]
    if sub.empty:
        return
    sub = sub.sort_values("rate", ascending=True)
    height = max(3, 0.5 * len(sub))
    fig, ax = plt.subplots(figsize=(8, height))
    bars = ax.barh(sub["ancestry_label"], sub["rate"], color="#4C78A8")
    ax.set_xlabel("Share of people in SNAP-recipient households")
    ax.set_ylabel("Ancestry (focus)")
    ax.set_title("Per-capita SNAP by selected ancestries")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.grid(axis="x", alpha=0.2)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    max_rate = float(sub["rate"].max()) if len(sub) else 0.0
    ax.set_xlim(0, min(1.0, max_rate * 1.10 + 1e-6))
    for rect, val in zip(bars, sub["rate" ].to_list()):
        width = rect.get_width()
        y = rect.get_y() + rect.get_height() / 2
        label = f"{val*100:.1f}%"
        if width >= max_rate * 0.25:
            ax.text(width - (max_rate * 0.01), y, label, va="center", ha="right", color="white", fontsize=9)
        else:
            ax.text(width + (max_rate * 0.01), y, label, va="center", ha="left", color="#333333", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

# -------- Main --------

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # 1) Load household SNAP indicator as Series indexed by SERIALNO
    hh_snap = load_household_snap(args.household, args.chunksize)

    # 2) Aggregate person-weighted totals by ancestry
    people_by_anc, people_snap_by_anc = aggregate_people_by_ancestry(
        args.person, hh_snap, args.chunksize
    )

    # 3) Build results DataFrame
    rows = []
    for code, tot in people_by_anc.items():
        if tot <= 0:
            continue
        tot_snap = people_snap_by_anc.get(code, 0.0)
        rows.append((code, tot, tot_snap, (tot_snap / tot)))
    res = pd.DataFrame(rows, columns=["ancestry_code", "people", "people_in_snap_hh", "rate"])

    # 4) Filter small groups
    res = res[res["people"] >= args.min_people].copy()

    # 5) Attach labels if provided; otherwise try auto-download from Census
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
        # Fallback to codes as labels
        res["ancestry_label"] = res["ancestry_code"].astype(str)

    # 6) Save CSV
    out_csv = os.path.join(args.outdir, "snap_per_capita_ancestry.csv")
    res.sort_values("rate", ascending=False).to_csv(out_csv, index=False)

    # 7) Plots
    plot_topn(res, os.path.join(args.outdir, "snap_per_capita_topN.png"), args.topn)
    # Also save an "all groups" plot (sorted by rate)
    plot_topn(res, os.path.join(args.outdir, "snap_per_capita_all.png"), topn=0)

    if args.focus:
        focus_list = [x.strip() for x in args.focus.split(",") if x.strip()]
        plot_focus(res, focus_list, os.path.join(args.outdir, "snap_per_capita_selected.png"))

    # 8) Quick stdout summary
    print(f"Saved: {out_csv}")
    print(res.sort_values('rate', ascending=False).head(10).assign(rate_pct=lambda d: (d['rate']*100).round(1)))


if __name__ == "__main__":
    main()
