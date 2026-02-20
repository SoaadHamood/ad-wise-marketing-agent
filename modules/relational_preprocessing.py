from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# -----------------------------
# Helpers
# -----------------------------

def _to_datetime_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")

def _fmt(x: Optional[float], ndigits: int = 4):
    if x is None:
        return None
    try:
        if np.isnan(x):
            return None
    except Exception:
        pass
    return round(float(x), ndigits)

def _safe_div(a, b):
    b = 0 if b is None else b
    if b == 0:
        return None
    return a / b

def _value_counts_table(df: pd.DataFrame, col: str, topk: int = 10) -> List[Dict]:
    if col not in df.columns:
        return []
    vc = df[col].value_counts(dropna=False).head(topk)
    out = [{"value": str(idx), "count": int(cnt)} for idx, cnt in vc.items()]
    return out

def save_json(obj: Dict, out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return out_path


# -----------------------------
# Load & Validate
# -----------------------------

REQUIRED_EVENTS = {"event_id", "ad_id", "user_id", "timestamp", "event_type"}
REQUIRED_ADS = {"ad_id", "campaign_id", "ad_platform"}
REQUIRED_CAMPAIGNS = {"campaign_id", "start_date", "end_date", "total_budget"}
REQUIRED_USERS = {"user_id", "age_group", "country"}

def load_relational_data(
    events_path: str,
    ads_path: str,
    campaigns_path: str,
    users_path: str,
) -> Dict[str, pd.DataFrame]:
    events = pd.read_csv(events_path, low_memory=False)
    ads = pd.read_csv(ads_path, low_memory=False)
    campaigns = pd.read_csv(campaigns_path, low_memory=False)
    users = pd.read_csv(users_path, low_memory=False)

    miss = REQUIRED_EVENTS - set(events.columns)
    if miss:
        raise ValueError(f"ad_events missing columns: {sorted(miss)}")
    miss = REQUIRED_ADS - set(ads.columns)
    if miss:
        raise ValueError(f"ads missing columns: {sorted(miss)}")
    miss = REQUIRED_CAMPAIGNS - set(campaigns.columns)
    if miss:
        raise ValueError(f"campaigns missing columns: {sorted(miss)}")
    miss = REQUIRED_USERS - set(users.columns)
    if miss:
        raise ValueError(f"users missing columns: {sorted(miss)}")

    # types
    events["timestamp"] = _to_datetime_series(events["timestamp"])
    campaigns["start_date"] = _to_datetime_series(campaigns["start_date"])
    campaigns["end_date"] = _to_datetime_series(campaigns["end_date"])

    # normalize ids
    events["ad_id"] = pd.to_numeric(events["ad_id"], errors="coerce").astype("Int64")
    ads["ad_id"] = pd.to_numeric(ads["ad_id"], errors="coerce").astype("Int64")
    ads["campaign_id"] = pd.to_numeric(ads["campaign_id"], errors="coerce").astype("Int64")
    campaigns["campaign_id"] = pd.to_numeric(campaigns["campaign_id"], errors="coerce").astype("Int64")

    # total_budget numeric
    campaigns["total_budget"] = pd.to_numeric(campaigns["total_budget"], errors="coerce")

    # drop broken rows (keep dataset robust)
    events = events.dropna(subset=["ad_id", "user_id", "timestamp", "event_type"]).copy()
    ads = ads.dropna(subset=["ad_id", "campaign_id", "ad_platform"]).copy()
    campaigns = campaigns.dropna(subset=["campaign_id", "start_date", "end_date"]).copy()
    users = users.dropna(subset=["user_id"]).copy()

    return {"events": events, "ads": ads, "campaigns": campaigns, "users": users}


# -----------------------------
# Join → Fact Table
# -----------------------------

def build_fact_table(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    events = dfs["events"]
    ads = dfs["ads"]
    campaigns = dfs["campaigns"]
    users = dfs["users"]

    # events -> ads (add campaign_id + platform + targeting fields if present)
    fact = events.merge(ads, on="ad_id", how="left", suffixes=("", "_ad"))
    # fact -> campaigns (add budget + dates)
    fact = fact.merge(campaigns, on="campaign_id", how="left", suffixes=("", "_camp"))
    # fact -> users (add demographics)
    fact = fact.merge(users, on="user_id", how="left", suffixes=("", "_user"))

    # derived time buckets
    fact["date"] = fact["timestamp"].dt.date.astype(str)
    fact["hour"] = fact["timestamp"].dt.hour

    # normalize event_type
    fact["event_type"] = fact["event_type"].astype(str).str.strip().str.lower()

    return fact


# -----------------------------
# Metrics / Signals (Way-1 style)
# -----------------------------

EVENT_IMPRESSION = "impression"
EVENT_LIKE = "like"
EVENT_SHARE = "share"
EVENT_CLICK = "click"  # if exists, otherwise it will be 0

def _event_counts(df: pd.DataFrame) -> Dict[str, int]:
    vc = df["event_type"].value_counts()
    return {
        "impressions": int(vc.get(EVENT_IMPRESSION, 0)),
        "likes": int(vc.get(EVENT_LIKE, 0)),
        "shares": int(vc.get(EVENT_SHARE, 0)),
        "clicks": int(vc.get(EVENT_CLICK, 0)),
        "events_total": int(len(df)),
    }

def _add_rates(row: pd.Series) -> pd.Series:
    impr = row.get("impressions", 0)
    likes = row.get("likes", 0)
    shares = row.get("shares", 0)
    clicks = row.get("clicks", 0)

    row["engagements"] = likes + shares
    row["engagement_rate"] = (likes + shares) / impr if impr and impr > 0 else np.nan
    row["ctr"] = clicks / impr if impr and impr > 0 else np.nan
    return row

def summarize_by_group(fact: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    # count event types per group
    pivot = (
        fact.pivot_table(
            index=group_cols,
            columns="event_type",
            values="event_id",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
    )

    # map possible columns to standard names
    rename = {}
    for col in pivot.columns:
        if col == EVENT_IMPRESSION:
            rename[col] = "impressions"
        elif col == EVENT_LIKE:
            rename[col] = "likes"
        elif col == EVENT_SHARE:
            rename[col] = "shares"
        elif col == EVENT_CLICK:
            rename[col] = "clicks"
    pivot = pivot.rename(columns=rename)

    # ensure missing standard columns exist
    for c in ["impressions", "likes", "shares", "clicks"]:
        if c not in pivot.columns:
            pivot[c] = 0

    # add derived rates
    pivot = pivot.apply(_add_rates, axis=1)

    # add budget info if campaign_id in group
    if "campaign_id" in group_cols and "total_budget" in fact.columns:
        bud = fact.groupby(group_cols, as_index=False)["total_budget"].mean()
        pivot = pivot.merge(bud, on=group_cols, how="left")
        # cost per engagement & per impression (rough)
        pivot["cpe"] = pivot["total_budget"] / pivot["engagements"].replace({0: np.nan})
        pivot["cpm_like"] = (pivot["total_budget"] / pivot["impressions"].replace({0: np.nan})) * 1000

    return pivot

def top_bottom(df: pd.DataFrame, metric: str, group_cols: List[str], k: int = 10) -> Dict[str, List[Dict]]:
    tmp = df[group_cols + [metric]].dropna(subset=[metric]).copy()
    top = tmp.sort_values(metric, ascending=False).head(k).to_dict(orient="records")
    bottom = tmp.sort_values(metric, ascending=True).head(k).to_dict(orient="records")
    return {"top": top, "bottom": bottom}

def detect_outliers(df: pd.DataFrame, metric: str, group_cols: List[str], z_thresh: float = 2.5, topk: int = 15) -> List[Dict]:
    s = df[metric].astype(float)
    mu = s.mean()
    sigma = s.std(ddof=0)
    if sigma == 0 or np.isnan(sigma):
        return []
    z = (s - mu) / sigma
    tmp = df[group_cols + [metric]].copy()
    tmp[f"{metric}_z"] = z
    out = tmp[z.abs() >= z_thresh].sort_values(f"{metric}_z", ascending=False).head(topk)
    return out.to_dict(orient="records")

def build_relational_signals(fact: pd.DataFrame, dataset_name: str = "relational_ads") -> Dict:
    signals: Dict = {"dataset": dataset_name}

    # Overall counts
    counts = _event_counts(fact)
    signals["overall_counts"] = counts
    signals["overall_kpis"] = {
        "n_events": int(len(fact)),
        "n_users": int(fact["user_id"].nunique()),
        "n_ads": int(fact["ad_id"].nunique()),
        "n_campaigns": int(fact["campaign_id"].nunique()) if "campaign_id" in fact.columns else None,
        "impressions": counts["impressions"],
        "likes": counts["likes"],
        "shares": counts["shares"],
        "clicks": counts["clicks"],
        "engagement_rate": _fmt(_safe_div(counts["likes"] + counts["shares"], counts["impressions"]), 6),
        "ctr": _fmt(_safe_div(counts["clicks"], counts["impressions"]), 6),
    }

    # basic distribution debug (helps report-writing / sanity)
    signals["distributions"] = {
        "event_type_top": _value_counts_table(fact, "event_type", topk=10),
        "platform_top": _value_counts_table(fact, "ad_platform", topk=10) if "ad_platform" in fact.columns else [],
        "age_group_top": _value_counts_table(fact, "age_group", topk=10) if "age_group" in fact.columns else [],
        "country_top": _value_counts_table(fact, "country", topk=10) if "country" in fact.columns else [],
    }

    # Group summaries (platform, age_group, campaign, platform x campaign)
    group_blocks: Dict = {}

    if "ad_platform" in fact.columns:
        by_platform = summarize_by_group(fact, ["ad_platform"])
        group_blocks["by_platform"] = {
            "top10_by_engagement_rate": by_platform.sort_values("engagement_rate", ascending=False).head(10).to_dict(orient="records"),
            "top10_by_impressions": by_platform.sort_values("impressions", ascending=False).head(10).to_dict(orient="records"),
        }

    if "age_group" in fact.columns:
        by_age = summarize_by_group(fact, ["age_group"])
        group_blocks["by_age_group"] = {
            "top10_by_engagement_rate": by_age.sort_values("engagement_rate", ascending=False).head(10).to_dict(orient="records"),
            "top10_by_impressions": by_age.sort_values("impressions", ascending=False).head(10).to_dict(orient="records"),
        }

    if "campaign_id" in fact.columns:
        by_campaign = summarize_by_group(fact, ["campaign_id"])
        group_blocks["by_campaign"] = {
            "top10_by_engagement_rate": by_campaign.sort_values("engagement_rate", ascending=False).head(10).to_dict(orient="records"),
            "top10_by_impressions": by_campaign.sort_values("impressions", ascending=False).head(10).to_dict(orient="records"),
        }

    if "campaign_id" in fact.columns and "ad_platform" in fact.columns:
        by_campaign_platform = summarize_by_group(fact, ["campaign_id", "ad_platform"])
        group_blocks["by_campaign_platform"] = {
            "top10_by_engagement_rate": by_campaign_platform.sort_values("engagement_rate", ascending=False).head(10).to_dict(orient="records"),
            "bottom10_by_engagement_rate": by_campaign_platform.sort_values("engagement_rate", ascending=True).head(10).to_dict(orient="records"),
        }

    signals["group_summaries"] = group_blocks

    # Top/bottom blocks (for report)
    top_bottom_blocks = {}
    if "campaign_id" in fact.columns:
        by_campaign = summarize_by_group(fact, ["campaign_id"])
        top_bottom_blocks["campaign_engagement_rate"] = top_bottom(by_campaign, "engagement_rate", ["campaign_id"], k=10)
        top_bottom_blocks["campaign_impressions"] = top_bottom(by_campaign, "impressions", ["campaign_id"], k=10)

    if "ad_platform" in fact.columns:
        by_platform = summarize_by_group(fact, ["ad_platform"])
        top_bottom_blocks["platform_engagement_rate"] = top_bottom(by_platform, "engagement_rate", ["ad_platform"], k=10)

    signals["top_bottom"] = top_bottom_blocks

    # Outliers (engagement_rate & ctr at campaign_platform level if exists)
    outliers = {}
    if "campaign_id" in fact.columns and "ad_platform" in fact.columns:
        lvl = summarize_by_group(fact, ["campaign_id", "ad_platform"])
        outliers["engagement_rate_outliers"] = detect_outliers(lvl, "engagement_rate", ["campaign_id", "ad_platform"])
        outliers["ctr_outliers"] = detect_outliers(lvl, "ctr", ["campaign_id", "ad_platform"])
    signals["outliers"] = outliers

    # Rule-based insights draft (simple)
    insights = []
    if "ad_platform" in fact.columns:
        by_platform = summarize_by_group(fact, ["ad_platform"]).sort_values("engagement_rate", ascending=False)
        if len(by_platform) > 0:
            best = by_platform.iloc[0]
            worst = by_platform.iloc[-1]
            insights.append({
                "type": "platform_engagement",
                "message": f"Top platform by engagement rate: {best['ad_platform']} (eng_rate={_fmt(best['engagement_rate'], 4)}). "
                           f"Lowest: {worst['ad_platform']} (eng_rate={_fmt(worst['engagement_rate'], 4)})."
            })

    if signals["overall_kpis"]["engagement_rate"] is not None and signals["overall_kpis"]["engagement_rate"] < 0.02:
        insights.append({
            "type": "low_engagement",
            "message": f"Overall engagement rate is low ({signals['overall_kpis']['engagement_rate']}). Consider creative refresh and tighter targeting."
        })

    signals["rule_based_insights"] = insights

    return signals


# -----------------------------
# Save CSV summaries (optional)
# -----------------------------

def save_group_csvs(fact: pd.DataFrame, out_dir: str, base_name: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    saved = []

    def _save(df: pd.DataFrame, name: str):
        p = os.path.join(out_dir, f"{base_name}_{name}.csv")
        df.to_csv(p, index=False)
        saved.append(p)

    if "ad_platform" in fact.columns:
        _save(summarize_by_group(fact, ["ad_platform"]), "by_platform")

    if "age_group" in fact.columns:
        _save(summarize_by_group(fact, ["age_group"]), "by_age_group")

    if "campaign_id" in fact.columns:
        _save(summarize_by_group(fact, ["campaign_id"]), "by_campaign")

    if "campaign_id" in fact.columns and "ad_platform" in fact.columns:
        _save(summarize_by_group(fact, ["campaign_id", "ad_platform"]), "by_campaign_platform")

    return saved
