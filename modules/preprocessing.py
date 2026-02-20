from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Helpers
# -----------------------------

def _to_datetime_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=False)

def _clean_numeric(s: pd.Series) -> pd.Series:
    """Convert strings like '$1,234.50' or '1,234' to float."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)

    def parse_one(x):
        if pd.isna(x):
            return np.nan
        x = str(x).strip()
        x = re.sub(r"[^0-9\.\-]", "", x)  # keep digits, dot, minus
        if x in ("", "-", "."):
            return np.nan
        try:
            return float(x)
        except:
            return np.nan

    return s.map(parse_one).astype(float)

def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    b = b.replace({0: np.nan})
    return a / b

def _fmt(x: Optional[float], ndigits: int = 4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return round(float(x), ndigits)

def _top_bottom(df: pd.DataFrame, metric: str, k: int = 10) -> Dict[str, List[Dict]]:
    cols = ["Campaign_ID", metric]
    extra = [c for c in ["Company", "Channel_Used", "Customer_Segment", "Target_Audience"] if c in df.columns]
    cols = cols + extra

    tmp = df[cols].dropna(subset=[metric]).copy()
    top = tmp.sort_values(metric, ascending=False).head(k).to_dict(orient="records")
    bottom = tmp.sort_values(metric, ascending=True).head(k).to_dict(orient="records")
    return {"top": top, "bottom": bottom}

def _group_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    agg = {
        "Clicks": "sum",
        "Impressions": "sum",
        "CTR": "mean",
        "ROI": "mean",
        "Conversion_Rate": "mean",
        "Acquisition_Cost": "mean",
        "Engagement_Score": "mean" if "Engagement_Score" in df.columns else "mean",
    }
    # keep only existing columns
    agg = {k: v for k, v in agg.items() if k in df.columns}
    out = df.groupby(group_col, as_index=False).agg(agg)

    # add share of impressions if possible
    if "Impressions" in out.columns and out["Impressions"].notna().any():
        total_impr = out["Impressions"].sum()
        out["impressions_share"] = np.where(total_impr > 0, out["Impressions"] / total_impr, np.nan)

    # rank by ROI if exists otherwise CTR
    rank_col = "ROI" if "ROI" in out.columns else "CTR"
    if rank_col in out.columns:
        out = out.sort_values(rank_col, ascending=False)

    return out

def _add_outlier_flags(df: pd.DataFrame, metric: str, z_thresh: float = 2.5) -> pd.DataFrame:
    s = df[metric].astype(float)
    mu = s.mean()
    sigma = s.std(ddof=0)
    if sigma == 0 or np.isnan(sigma):
        df[f"{metric}_z"] = np.nan
        df[f"{metric}_outlier"] = False
        return df
    z = (s - mu) / sigma
    df[f"{metric}_z"] = z
    df[f"{metric}_outlier"] = z.abs() >= z_thresh
    return df


# -----------------------------
# Load & Prepare
# -----------------------------

REQUIRED_COLS = {"Campaign_ID", "Clicks", "Impressions", "ROI", "Conversion_Rate", "Acquisition_Cost"}
OPTIONAL_COLS = {
    "Company", "Channel_Used", "Customer_Segment", "Target_Audience",
    "Campaign_Type", "Campaign_Goal", "Location", "Language", "Engagement_Score", "Date"
}

def load_and_prepare(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    # Normalize types
    df["Campaign_ID"] = _clean_numeric(df["Campaign_ID"]).astype("Int64")
    df["Clicks"] = _clean_numeric(df["Clicks"])
    df["Impressions"] = _clean_numeric(df["Impressions"])
    df["ROI"] = _clean_numeric(df["ROI"])
    df["Conversion_Rate"] = _clean_numeric(df["Conversion_Rate"])
    df["Acquisition_Cost"] = _clean_numeric(df["Acquisition_Cost"])

    # Optional date
    if "Date" in df.columns:
        df["Date"] = _to_datetime_series(df["Date"])

    # Derived
    df["CTR"] = _safe_div(df["Clicks"], df["Impressions"])

    # Clean categoricals (optional)
    for c in ["Company", "Channel_Used", "Customer_Segment", "Target_Audience"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # Drop broken rows
    df = df.dropna(subset=["Campaign_ID"]).copy()
    df["Campaign_ID"] = df["Campaign_ID"].astype(int)

    return df


# -----------------------------
# Signal Extraction (Way 1)
# -----------------------------

def build_signals(df: pd.DataFrame, dataset_name: str) -> Dict:
    """
    Build report-ready signals without relying on time series:
    - overall KPIs
    - top/bottom campaigns by ROI/CTR/Conversion
    - group summaries (Channel, Company, Segment)
    - outliers
    """
    signals: Dict = {"dataset": dataset_name}

    # Overall KPIs
    overall = {
        "n_rows": int(len(df)),
        "n_campaigns": int(df["Campaign_ID"].nunique()),
        "clicks_total": _fmt(df["Clicks"].sum(), 2),
        "impressions_total": _fmt(df["Impressions"].sum(), 2),
        "ctr_mean": _fmt(df["CTR"].mean(), 6),
        "roi_mean": _fmt(df["ROI"].mean(), 4),
        "conversion_rate_mean": _fmt(df["Conversion_Rate"].mean(), 6),
        "acquisition_cost_mean": _fmt(df["Acquisition_Cost"].mean(), 4),
    }
    signals["overall_kpis"] = overall

    # Top/Bottom campaigns
    signals["top_bottom"] = {
        "ROI": _top_bottom(df, "ROI", k=10),
        "CTR": _top_bottom(df, "CTR", k=10),
        "Conversion_Rate": _top_bottom(df, "Conversion_Rate", k=10),
    }

    # Group summaries
    group_blocks = {}
    for group_col in ["Channel_Used", "Company", "Customer_Segment", "Target_Audience"]:
        if group_col in df.columns:
            summary_df = _group_summary(df, group_col)
            group_blocks[group_col] = {
                "summary_top10": summary_df.head(10).to_dict(orient="records"),
                "summary_bottom10": summary_df.tail(10).to_dict(orient="records"),
            }
    signals["group_summaries"] = group_blocks

    # Outliers (ROI + CTR)
    outliers = {}
    tmp = df[["Campaign_ID", "ROI", "CTR"] + [c for c in ["Company", "Channel_Used"] if c in df.columns]].copy()
    tmp = _add_outlier_flags(tmp, "ROI", z_thresh=2.5)
    tmp = _add_outlier_flags(tmp, "CTR", z_thresh=2.5)

    roi_out = tmp[tmp["ROI_outlier"]].sort_values("ROI_z", ascending=False).head(15)
    ctr_out = tmp[tmp["CTR_outlier"]].sort_values("CTR_z", ascending=False).head(15)

    outliers["ROI_outliers"] = roi_out.to_dict(orient="records")
    outliers["CTR_outliers"] = ctr_out.to_dict(orient="records")
    signals["outliers"] = outliers

    # Quick “insights” draft (rules-based)
    insights = []

    # Best channel by ROI
    if "Channel_Used" in df.columns:
        ch = _group_summary(df, "Channel_Used")
        if "ROI" in ch.columns and len(ch) > 0:
            best = ch.iloc[0]
            worst = ch.iloc[-1]
            insights.append({
                "type": "channel_roi",
                "message": f"Top channel by ROI: {best['Channel_Used']} (avg ROI={_fmt(best['ROI'], 4)}). "
                           f"Lowest: {worst['Channel_Used']} (avg ROI={_fmt(worst['ROI'], 4)})."
            })

    # Low CTR overall
    if overall["ctr_mean"] is not None and overall["ctr_mean"] < 0.01:
        insights.append({
            "type": "low_ctr",
            "message": f"Overall CTR is low (mean CTR={overall['ctr_mean']}). Consider creative refresh and targeting refinement."
        })

    signals["rule_based_insights"] = insights

    return signals


# -----------------------------
# Saving utilities
# -----------------------------

def save_signals(signals: Dict, out_dir: str, base_name: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{base_name}_signals.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(signals, f, ensure_ascii=False, indent=2)
    return out_path

def save_group_csvs(df: pd.DataFrame, out_dir: str, base_name: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for group_col in ["Channel_Used", "Company", "Customer_Segment", "Target_Audience"]:
        if group_col in df.columns:
            g = _group_summary(df, group_col)
            p = os.path.join(out_dir, f"{base_name}_by_{group_col}.csv")
            g.to_csv(p, index=False)
            saved.append(p)
    return saved
