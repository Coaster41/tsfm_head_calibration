import argparse
import os
import re
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd


def quant_label(q: float, nd: int = 6) -> str:
    """
    Canonical string label for a quantile.
    Rounds to 'nd' decimals, formats to fixed decimals, then strips trailing zeros and dot.
    Ensures 0.1, 0.1000000000001, (1-s)/2 all produce the exact same label.
    """
    # Round first (avoids tiny float noise), format fixed decimals, then strip
    s = f"{round(float(q), nd):.{nd}f}"
    s = s.rstrip("0").rstrip(".")
    # Edge case: 0 -> "0"
    return s if s != "" else "0"


def _read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['ds'])
    # Drop any unnamed index column
    for col in list(df.columns):
        if str(col).lower().startswith("unnamed"):
            df = df.drop(columns=[col])
    return df


def load_observations(path: str) -> pd.DataFrame:
    """
    Observed CSV: columns must include ["unique_id","ds","y"]
    """
    y_df = _read_csv(path)
    # Canonicalize column names
    cols_lower = [c.lower() for c in y_df.columns]
    rename_map = {}
    for i, c in enumerate(y_df.columns):
        if cols_lower[i] == "unique_id":
            rename_map[c] = "unique_id"
        elif cols_lower[i] == "ds":
            rename_map[c] = "ds"
        elif cols_lower[i] == "y":
            rename_map[c] = "y"
    y_df = y_df.rename(columns=rename_map)
    # Parse ds as datetime
    y_df["ds"] = pd.to_datetime(y_df["ds"], utc=False)
    # Sanity
    expected = {"unique_id", "ds", "y"}
    missing = expected - set(y_df.columns)
    if missing:
        raise ValueError(f"Observations CSV missing columns: {missing}")
    return y_df[["unique_id", "ds", "y"]]


def read_forecast_wide(path: str) -> pd.DataFrame:
    """
    Forecast CSV: columns must include ["unique_id","ds"] + horizon columns ["1","2",...].
    ds is the timestamp of last observed value; horizons predict after ds.
    """
    df = _read_csv(path)
    # Canonicalize column names
    cols_lower = [c.lower() for c in df.columns]
    rename_map = {}
    for i, c in enumerate(df.columns):
        if cols_lower[i] == "unique_id":
            rename_map[c] = "unique_id"
        elif cols_lower[i] == "ds":
            rename_map[c] = "ds"
    df = df.rename(columns=rename_map)
    # Parse ds as datetime
    df["ds"] = pd.to_datetime(df["ds"], utc=False)
    # Identify horizon columns (numeric names)
    horizon_cols = [c for c in df.columns if str(c).isdigit()]
    if not horizon_cols:
        # In case the columns are integers, convert to str
        horizon_cols = [c for c in df.columns if isinstance(c, (int, np.integer))]
        if horizon_cols:
            df = df.rename(columns={c: str(c) for c in horizon_cols})
            horizon_cols = [str(c) for c in horizon_cols]
    if not horizon_cols:
        raise ValueError(f"No horizon columns found in forecast file {path}. Expected columns '1','2',...")

    return df[["unique_id", "ds"] + horizon_cols]


def melt_forecast_to_long(
    forecast_wide: pd.DataFrame,
    freq: Union[str, pd.DateOffset],
    value_name: str = "yhat",
) -> pd.DataFrame:
    """
    Convert wide forecast table (one row per last observed ds with columns 1..H) into long:
    columns: unique_id, ds_last, h, target_ds, yhat
    """
    if isinstance(freq, str):
        offset = pd.tseries.frequencies.to_offset(freq)
    else:
        offset = freq

    horizon_cols = [c for c in forecast_wide.columns if str(c).isdigit()]
    long_df = forecast_wide.melt(
        id_vars=["unique_id", "ds"],
        value_vars=horizon_cols,
        var_name="h",
        value_name=value_name,
    )
    long_df = long_df.rename(columns={"ds": "ds_last"})
    # Ensure ds_last is datetime64[ns]
    if not np.issubdtype(long_df["ds_last"].dtype, np.datetime64):
        long_df["ds_last"] = pd.to_datetime(long_df["ds_last"], utc=False)

    long_df["h"] = long_df["h"].astype(int)

    # Vectorized target_ds computation
    # For fixed-frequency offsets (Tick), offset.delta is a Python timedelta
    if isinstance(offset, pd.tseries.offsets.Tick) and offset.delta is not None:
        step = pd.to_timedelta(offset.delta)  # pandas Timedelta
        # h * step yields a Timedelta series; addition stays datetime64[ns]
        long_df["target_ds"] = long_df["ds_last"] + (long_df["h"] * step)
    else:
        # Fallback for non-fixed offsets (e.g., MonthEnd); slower but correct
        long_df["target_ds"] = long_df["ds_last"] + long_df["h"].map(lambda k: k * offset)

    # Ensure target_ds is datetime64[ns] (guards against object dtype)
    long_df["target_ds"] = pd.to_datetime(long_df["target_ds"], utc=False)

    return long_df[["unique_id", "ds_last", "h", "target_ds", value_name]]


def align_with_truth(
    preds_long: pd.DataFrame, y_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge predictions (long) with truth on (unique_id, target_ds).
    Returns a DataFrame with columns from preds_long plus 'y'.
    """
    preds = preds_long.copy()
    # Force target_ds to datetime64[ns]
    preds["target_ds"] = pd.to_datetime(preds["target_ds"], utc=False)

    y_keyed = y_df.rename(columns={"ds": "target_ds"}).copy()
    y_keyed["target_ds"] = pd.to_datetime(y_keyed["target_ds"], utc=False)

    merged = preds.merge(
        y_keyed[["unique_id", "target_ds", "y"]],
        on=["unique_id", "target_ds"],
        how="inner",
    )
    return merged


def compute_naive_mae(y_df: pd.DataFrame, freq: Union[str, pd.DateOffset]) -> float:
    """
    Naive MAE based on in-sample one-step-ahead persistence:
    naive yhat_t = y_{t-1}, where t-1 is exactly one freq unit before t.
    Uses a single self-merge to ensure exact offset alignment (handles irregular spacing).
    """
    if isinstance(freq, str):
        offset = pd.tseries.frequencies.to_offset(freq)
    else:
        offset = freq

    prev_df = y_df[["unique_id", "ds", "y"]].copy()
    prev_df["ds"] = prev_df["ds"] + offset
    prev_df = prev_df.rename(columns={"y": "y_prev"})

    merged = y_df.merge(prev_df, on=["unique_id", "ds"], how="inner")
    naive_mae = np.mean(np.abs(merged["y"] - merged["y_prev"])) if len(merged) else np.nan
    return naive_mae


def build_quantile_long(
    quantile_paths: Dict[float, str], freq: Union[str, pd.DateOffset]
) -> pd.DataFrame:
    """
    Returns a single long DataFrame with columns:
    unique_id, ds_last, h, target_ds, q, yhat
    """
    parts = []
    for q, path in sorted(quantile_paths.items()):
        fw = read_forecast_wide(path)
        long_df = melt_forecast_to_long(fw, freq=freq, value_name="yhat")
        long_df["q"] = float(q)
        parts.append(long_df)
    q_long = pd.concat(parts, ignore_index=True)
    return q_long[["unique_id", "ds_last", "h", "target_ds", "q", "yhat"]]


def build_lower_upper_panel(
    q_aligned: pd.DataFrame, q_low: float, q_high: float
) -> pd.DataFrame:
    """
    From the aligned quantile-long df, build lower/upper panel:
    columns: unique_id, target_ds, h, lower, upper, y
    Uses canonical string quantile labels to avoid float-equality issues.
    """
    df = q_aligned.copy()
    df["q_label"] = df["q"].apply(quant_label)
    low_lab = quant_label(q_low)
    high_lab = quant_label(q_high)

    slc = df[df["q_label"].isin([low_lab, high_lab])].copy()

    # 'y' reference per (unique_id, target_ds, h)
    y_ref = (
        slc[["unique_id", "target_ds", "h", "y"]]
        .drop_duplicates(["unique_id", "target_ds", "h"])
        .set_index(["unique_id", "target_ds", "h"])
    )

    # Pivot predictions by label
    pv = slc.pivot_table(
        index=["unique_id", "target_ds", "h"], columns="q_label", values="yhat", aggfunc="mean"
    )

    rename_cols = {}
    if low_lab in pv.columns:
        rename_cols[low_lab] = "lower"
    if high_lab in pv.columns:
        rename_cols[high_lab] = "upper"
    pv = pv.rename(columns=rename_cols)

    panel = pv.join(y_ref, how="inner")

    if not {"lower", "upper"}.issubset(set(panel.columns)):
        raise ValueError(
            f"Missing lower or upper quantile columns after pivot. "
            f"Looked for labels {low_lab} and {high_lab}. Available: {list(pv.columns)}"
        )

    return panel.reset_index()


def build_quantile_pivot(q_aligned: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot all quantiles so we can form multiple intervals.
    Index: (unique_id, target_ds, h)
    Columns: canonical string labels for quantiles + 'y'
    """
    df = q_aligned.copy()
    df["q_label"] = df["q"].apply(quant_label)

    y_ref = (
        df[["unique_id", "target_ds", "h", "y"]]
        .drop_duplicates(["unique_id", "target_ds", "h"])
        .set_index(["unique_id", "target_ds", "h"])
    )

    pv = df.pivot_table(
        index=["unique_id", "target_ds", "h"], columns="q_label", values="yhat", aggfunc="mean"
    )
    pv = pv.join(y_ref, how="inner")
    return pv


def metric_mae_and_mase(
    median_aligned: pd.DataFrame, naive_mae: float
) -> Tuple[pd.Series, float, pd.Series, float]:
    """
    Returns:
    - MAE per-horizon (Series indexed by h),
    - MAE_mean_over_h,
    - MASE per-horizon (Series),
    - MASE_mean_over_h
    """
    err = np.abs(median_aligned["y"] - median_aligned["yhat"])
    mae_h = err.groupby(median_aligned["h"]).mean().sort_index()
    mae_mean = mae_h.mean() if len(mae_h) else np.nan

    if np.isnan(naive_mae) or naive_mae == 0:
        mase_h = mae_h * np.nan
        mase_mean = np.nan
    else:
        mase_h = mae_h / naive_mae
        mase_mean = mase_h.mean()
    return mae_h, mae_mean, mase_h, mase_mean


def metric_mae_and_mase_by_series(
    median_aligned: pd.DataFrame, naive_mae: float
) -> Tuple[pd.Series, pd.Series]:
    """
    Returns:
      - MAE per (unique_id, h)
      - MASE per (unique_id, h) scaled by global naive_mae for comparability
    """
    err = np.abs(median_aligned["y"] - median_aligned["yhat"])
    mae_uh = err.groupby([median_aligned["unique_id"], median_aligned["h"]]).mean().sort_index()
    if np.isnan(naive_mae) or naive_mae == 0:
        mase_uh = mae_uh * np.nan
    else:
        mase_uh = mae_uh / naive_mae
    return mae_uh, mase_uh


def metric_pce(q_aligned: pd.DataFrame) -> Tuple[pd.Series, float]:
    """
    Probabilistic Calibration Error per-horizon and mean over horizons.
    PCE_h = mean_q | q - mean( 1[y <= yhat_q] | h, q ) |
    """
    # Empirical CDF at predicted quantiles
    indicators = (q_aligned["y"] <= q_aligned["yhat"]).astype(float)
    grp = q_aligned.assign(ind=indicators).groupby(["h", "q"])["ind"].mean()
    # Absolute difference from nominal q
    pce_hq = (grp.index.get_level_values("q") - grp.values)
    pce_hq = np.abs(pce_hq)
    # Average across q per h
    # Build a Series indexed by (h,q), then group by h
    pce_hq = pd.Series(pce_hq, index=grp.index)
    pce_h = pce_hq.groupby(level="h").mean().sort_index()
    return pce_h, pce_h.mean()


def metric_pce_by_series(q_aligned: pd.DataFrame) -> pd.Series:
    """
    PCE per (unique_id, h): average over q of |q - mean(1[y <= yhat_q])|
    """
    indicators = (q_aligned["y"] <= q_aligned["yhat"]).astype(float)
    grp = q_aligned.assign(ind=indicators).groupby(["unique_id", "h", "q"])["ind"].mean()
    pce_uhq = pd.Series(np.abs(grp.index.get_level_values("q") - grp.values), index=grp.index)
    pce_uh = pce_uhq.groupby(level=["unique_id", "h"]).mean().sort_index()
    return pce_uh


def metric_tpce(panel_lu: pd.DataFrame, confidence: float) -> Tuple[pd.Series, float]:
    """
    Tailed PCE per-horizon and mean over horizons:
    outside_ratio = (1 - confidence)/2
    TPCE_h = 0.5*( |outside_ratio - mean(y > upper)| + |outside_ratio - mean(y < lower)| )
    """
    outside_ratio = (1.0 - confidence) / 2.0
    by_h = panel_lu.groupby("h")
    upper_out = by_h.apply(lambda g: (g["y"] > g["upper"]).mean())
    lower_out = by_h.apply(lambda g: (g["y"] < g["lower"]).mean())
    tpce_h = 0.5 * (np.abs(outside_ratio - upper_out) + np.abs(outside_ratio - lower_out))
    tpce_h = tpce_h.sort_index()
    return tpce_h, tpce_h.mean()


def metric_tpce_by_series(panel_lu: pd.DataFrame, confidence: float) -> pd.Series:
    outside_ratio = (1.0 - confidence) / 2.0
    by_uh = panel_lu.groupby(["unique_id", "h"])
    upper_out = by_uh.apply(lambda g: (g["y"] > g["upper"]).mean())
    lower_out = by_uh.apply(lambda g: (g["y"] < g["lower"]).mean())
    tpce_uh = 0.5 * (np.abs(outside_ratio - upper_out) + np.abs(outside_ratio - lower_out))
    return tpce_uh.sort_index()


def metric_tcce_from_pivot(q_pivot: pd.DataFrame, confidence: float) -> Tuple[pd.Series, float]:
    """
    Tailed Centered Calibration Error for a single interval with s=confidence.
    s = q_high - q_low with q_low=(1-s)/2, q_high=1-q_low
    """
    s = confidence
    q_low = (1.0 - s) / 2.0
    q_high = 1.0 - q_low
    low_lab = quant_label(q_low)
    high_lab = quant_label(q_high)
    if low_lab not in q_pivot.columns or high_lab not in q_pivot.columns:
        raise ValueError(f"Quantile labels '{low_lab}' and/or '{high_lab}' not found for TCCE. "
                         f"Available: {list(q_pivot.columns)}")
    inside = (q_pivot[low_lab] <= q_pivot["y"]) & (q_pivot["y"] <= q_pivot[high_lab])
    cov_h = inside.groupby(level="h").mean().sort_index()
    tcce_h = s - cov_h
    return tcce_h, tcce_h.mean()


def metric_tcce_by_series_from_pivot(q_pivot: pd.DataFrame, confidence: float) -> pd.Series:
    s = confidence
    q_low = (1.0 - s) / 2.0
    q_high = 1.0 - q_low
    low_lab = quant_label(q_low)
    high_lab = quant_label(q_high)
    if low_lab not in q_pivot.columns or high_lab not in q_pivot.columns:
        raise ValueError(f"Quantile labels '{low_lab}' and/or '{high_lab}' not found for TCCE. "
                         f"Available: {list(q_pivot.columns)}")
    inside = (q_pivot[low_lab] <= q_pivot["y"]) & (q_pivot["y"] <= q_pivot[high_lab])
    cov_uh = inside.groupby(level=["unique_id", "h"]).mean().sort_index()
    tcce_uh = s - cov_uh
    return tcce_uh


def metric_cce_multi_from_pivot(
    q_pivot: pd.DataFrame, S: List[float]
) -> Tuple[pd.Series, pd.DataFrame, float]:
    """
    Centered Calibration Error averaged over multiple symmetric intervals S.
    For each s in S: q_low=(1-s)/2, q_high=1-q_low, CCE_s(h) = s - mean(1[lower<=y<=upper])
    Returns:
      - CCE_h_mean: average CCE over S per horizon (signed)
      - CCE_h_by_s: DataFrame with columns per s
      - CCE_mean_over_h: overall mean of CCE_h_mean
    """
    cce_per_s = {}
    for s in S:
        q_low = (1.0 - s) / 2.0
        q_high = 1.0 - q_low
        low_lab = quant_label(q_low)
        high_lab = quant_label(q_high)
        if low_lab not in q_pivot.columns or high_lab not in q_pivot.columns:
            raise ValueError(f"Quantile labels '{low_lab}'/'{high_lab}' not found for CCE s={s}. "
                             f"Available: {list(q_pivot.columns)}")
        inside = (q_pivot[low_lab] <= q_pivot["y"]) & (q_pivot["y"] <= q_pivot[high_lab])
        cov_h = inside.groupby(level="h").mean().sort_index()
        cce_per_s[s] = s - cov_h
    cce_h_by_s = pd.DataFrame(cce_per_s).sort_index()
    cce_h_mean = cce_h_by_s.mean(axis=1)
    return cce_h_mean, cce_h_by_s, cce_h_mean.mean()


def metric_cce_multi_by_series_from_pivot(
    q_pivot: pd.DataFrame, S: List[float]
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Returns:
      - CCE_uh_mean: average across S per (unique_id,h)
      - CCE_uh_by_s: DataFrame with MultiIndex (unique_id,h) and columns S
    """
    cce_per_s = {}
    for s in S:
        q_low = (1.0 - s) / 2.0
        q_high = 1.0 - q_low
        low_lab = quant_label(q_low)
        high_lab = quant_label(q_high)
        if low_lab not in q_pivot.columns or high_lab not in q_pivot.columns:
            raise ValueError(f"Quantile labels '{low_lab}'/'{high_lab}' not found for CCE s={s}. "
                             f"Available: {list(q_pivot.columns)}")
        inside = (q_pivot[low_lab] <= q_pivot["y"]) & (q_pivot["y"] <= q_pivot[high_lab])
        cov_uh = inside.groupby(level=["unique_id", "h"]).mean().sort_index()
        cce_per_s[s] = s - cov_uh
    cce_uh_by_s = pd.DataFrame(cce_per_s).sort_index()
    cce_uh_mean = cce_uh_by_s.mean(axis=1)
    return cce_uh_mean, cce_uh_by_s


def metric_siw(panel_lu: pd.DataFrame, y_df: pd.DataFrame, q_low: float, q_high: float) -> Tuple[pd.Series, float]:
    """
    Scaled Interval Width per-horizon:
    SIW_h = mean(upper - lower) / (Q_y(q_high) - Q_y(q_low)) over all y
    """
    y_q_low = np.quantile(y_df["y"], q_low)
    y_q_high = np.quantile(y_df["y"], q_high)
    denom = (y_q_high - y_q_low)
    width = (panel_lu["upper"] - panel_lu["lower"])
    siw_h_raw = width.groupby(panel_lu["h"]).mean().sort_index()
    if denom == 0:
        siw_h = siw_h_raw * np.nan
    else:
        siw_h = siw_h_raw / denom
    return siw_h, siw_h.mean()


def metric_siw_by_series(panel_lu: pd.DataFrame, y_df: pd.DataFrame, q_low: float, q_high: float) -> pd.Series:
    y_q_low = np.quantile(y_df["y"], q_low)
    y_q_high = np.quantile(y_df["y"], q_high)
    denom = (y_q_high - y_q_low)
    width = (panel_lu["upper"] - panel_lu["lower"])
    siw_uh_raw = width.groupby([panel_lu["unique_id"], panel_lu["h"]]).mean().sort_index()
    siw_uh = siw_uh_raw / denom if denom != 0 else siw_uh_raw * np.nan
    return siw_uh


def metric_siw_multi_from_pivot(
    q_pivot: pd.DataFrame, y_df: pd.DataFrame, S: List[float]
) -> Tuple[pd.Series, pd.DataFrame, float]:
    """
    Symmetric Interval Width (SIW) averaged across multiple symmetric intervals S.
    For each s in S, with q_low=(1-s)/2 and q_high=1-q_low:
      width_s(h) = mean over rows at horizon h of (upper_s - lower_s)
      denom_s = Q_y(q_high) - Q_y(q_low) computed globally over y_df["y"]
      SIW_s(h) = width_s(h) / denom_s
    Returns:
      - SIW_h_mean: average SIW over S per horizon (Series indexed by h)
      - SIW_h_by_s: DataFrame with columns per s (index=h)
      - SIW_mean_over_h: overall mean of SIW_h_mean
    """
    siw_per_s = {}
    for s in S:
        q_low = (1.0 - s) / 2.0
        q_high = 1.0 - q_low
        low_lab = quant_label(q_low)
        high_lab = quant_label(q_high)
        if low_lab not in q_pivot.columns or high_lab not in q_pivot.columns:
            raise ValueError(
                f"Quantile labels '{low_lab}'/'{high_lab}' not found for SIW s={s}. "
                f"Available: {list(q_pivot.columns)}"
            )
        # Forecast interval width per row
        width_row = q_pivot[high_lab] - q_pivot[low_lab]
        # Aggregate by horizon
        width_h = width_row.groupby(level="h").mean().sort_index()
        # Global denominator based on observed y distribution at the same (q_low,q_high)
        y_q_low = np.quantile(y_df["y"], q_low)
        y_q_high = np.quantile(y_df["y"], q_high)
        denom = (y_q_high - y_q_low)
        siw_s_h = width_h / denom if denom != 0 else width_h * np.nan
        siw_per_s[s] = siw_s_h

    siw_h_by_s = pd.DataFrame(siw_per_s).sort_index()  # index: h, columns: s
    siw_h_mean = siw_h_by_s.mean(axis=1)
    return siw_h_mean, siw_h_by_s, siw_h_mean.mean()


def metric_siw_multi_by_series_from_pivot(
    q_pivot: pd.DataFrame, y_df: pd.DataFrame, S: List[float]
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Per-series SIW averaged across S.
    Returns:
      - SIW_uh_mean: Series indexed by (unique_id,h) averaging SIW across S
      - SIW_uh_by_s: DataFrame with MultiIndex (unique_id,h) and columns S
    Uses a global denominator per s (same as overall SIW) for cross-series comparability.
    """
    siw_per_s = {}
    for s in S:
        q_low = (1.0 - s) / 2.0
        q_high = 1.0 - q_low
        low_lab = quant_label(q_low)
        high_lab = quant_label(q_high)
        if low_lab not in q_pivot.columns or high_lab not in q_pivot.columns:
            raise ValueError(
                f"Quantile labels '{low_lab}'/'{high_lab}' not found for SIW s={s}. "
                f"Available: {list(q_pivot.columns)}"
            )
        width_row = q_pivot[high_lab] - q_pivot[low_lab]
        width_uh = width_row.groupby(level=["unique_id", "h"]).mean().sort_index()

        y_q_low = np.quantile(y_df["y"], q_low)
        y_q_high = np.quantile(y_df["y"], q_high)
        denom = (y_q_high - y_q_low)
        siw_s_uh = width_uh / denom if denom != 0 else width_uh * np.nan
        siw_per_s[s] = siw_s_uh

    siw_uh_by_s = pd.DataFrame(siw_per_s).sort_index()  # index: (unique_id,h)
    siw_uh_mean = siw_uh_by_s.mean(axis=1)
    return siw_uh_mean, siw_uh_by_s


def metric_wql(
    q_aligned: pd.DataFrame,
    reduce: str = "mean",            # "mean" or "sum" across (unique_id, ds_last) per horizon
    scale_mode: str = "auto",        # "auto", "mean_y", "sum_y", or "none"
) -> Tuple[pd.Series, float, pd.Series]:
    """
    Weighted Quantile Loss with configurable reduction and scaling.
    Steps:
      1) Row-level per-quantile loss:
         ql = 2 * [ q*(y - yhat) if y >= yhat else (1-q)*(yhat - y) ]
      2) Aggregate across quantiles to get per-forecast-origin loss:
         group by (h, unique_id, ds_last, target_ds), sum ql -> ql_row; keep y_row (first)
      3) Reduce across forecast-origins for each horizon:
         - reduce="sum": WQL_h_num = sum(ql_row)
         - reduce="mean": WQL_h_num = mean(ql_row)
      4) Scale per horizon:
         - scale_mode="sum_y": divide by sum of y_row (or sum of unique target y if preferred)
         - scale_mode="mean_y": divide by mean of y_row
         - scale_mode="none": no scaling
         - scale_mode="auto": "mean_y" if reduce="mean", else "sum_y"

    Returns:
      - WQL_h (Series per horizon, normalized)
      - WQL_total (sum over h if reduce='sum', else mean over h if reduce='mean')
      - Optional detail: per-(h,q) normalized sum if you need it (kept for compatibility; still uses sum reduction over rows)
    """
    y = q_aligned["y"].values
    yhat = q_aligned["yhat"].values
    qvals = q_aligned["q"].values
    ge_mask = (y >= yhat)
    ql = np.where(
        ge_mask,
        2.0 * qvals * (y - yhat),
        2.0 * (1.0 - qvals) * (yhat - y),
    )
    df = q_aligned.copy()
    df["ql"] = ql

    # Row-level (one per forecast origin for a given horizon)
    row = (
        df.groupby(["h", "unique_id", "ds_last", "target_ds"], as_index=False)
          .agg(ql=("ql", "sum"), y=("y", "first"))
    )

    # Reduction across forecast origins for each horizon
    if reduce not in ("mean", "sum"):
        raise ValueError("reduce must be 'mean' or 'sum'")
    if reduce == "mean":
        wql_h_num = row.groupby("h")["ql"].mean().sort_index()
    else:
        wql_h_num = row.groupby("h")["ql"].sum().sort_index()

    # Scaling
    if scale_mode == "auto":
        scale_mode = "mean_y" if reduce == "mean" else "sum_y"

    if scale_mode == "mean_y":
        scale_h = row.groupby("h")["y"].mean().sort_index()
    elif scale_mode == "sum_y":
        # Option A: sum over rows
        scale_h = row.groupby("h")["y"].sum().sort_index()
        # Option B (alternative): sum over unique targets per h
        # uniq = row.drop_duplicates(subset=["h", "unique_id", "target_ds"])
        # scale_h = uniq.groupby("h")["y"].sum().sort_index()
    elif scale_mode == "none":
        scale_h = pd.Series(1.0, index=wql_h_num.index)
    else:
        raise ValueError("scale_mode must be 'auto','mean_y','sum_y', or 'none'")

    eps = 1e-12
    wql_h = wql_h_num / np.maximum(scale_h, eps)

    # Total: make the aggregate consistent with reduction type
    wql_total = wql_h.mean() if reduce == "mean" else wql_h.sum()

    # Optional per-(h,q) detail (still sum across rows, then scaled by sum_y for backward compat)
    ql_hq_sum = df.groupby(["h", "q"])["ql"].sum().sort_index()
    uniq_targets = df.drop_duplicates(subset=["unique_id", "target_ds"])
    scale_sum_y = uniq_targets.groupby("h")["y"].sum().reindex(ql_hq_sum.index.get_level_values("h")).values
    wql_hq = ql_hq_sum / np.maximum(scale_sum_y, eps)

    return wql_h, wql_total, wql_hq


def metric_wql_by_series(
    q_aligned: pd.DataFrame,
    reduce: str = "mean",
    scale_mode: str = "sum_y",
) -> pd.Series:
    """
    WQL per (unique_id, h) with same semantics as metric_wql, but computed within each series.
    """
    y = q_aligned["y"].values
    yhat = q_aligned["yhat"].values
    qvals = q_aligned["q"].values
    ge_mask = (y >= yhat)
    ql = np.where(
        ge_mask,
        2.0 * qvals * (y - yhat),
        2.0 * (1.0 - qvals) * (yhat - y),
    )
    df = q_aligned.copy()
    df["ql"] = ql

    # Row-level within series
    row = (
        df.groupby(["unique_id", "h", "ds_last", "target_ds"], as_index=False)
          .agg(ql=("ql", "sum"), y=("y", "first"))
    )

    # Reduce across origins per (unique_id,h)
    if reduce not in ("mean", "sum"):
        raise ValueError("reduce must be 'mean' or 'sum'")
    if reduce == "mean":
        wql_uh_num = row.groupby(["unique_id", "h"])["ql"].mean().sort_index()
    else:
        wql_uh_num = row.groupby(["unique_id", "h"])["ql"].sum().sort_index()

    # Scaling per (unique_id,h)
    if scale_mode == "auto":
        scale_mode = "mean_y" if reduce == "mean" else "sum_y"
    if scale_mode == "mean_y":
        scale_uh = row.groupby(["unique_id", "h"])["y"].mean().sort_index()
    elif scale_mode == "sum_y":
        scale_uh = row.groupby(["unique_id", "h"])["y"].sum().sort_index()
    elif scale_mode == "none":
        scale_uh = pd.Series(1.0, index=wql_uh_num.index)
    else:
        raise ValueError("scale_mode must be 'auto','mean_y','sum_y', or 'none'")

    eps = 1e-12
    wql_uh = wql_uh_num / np.maximum(scale_uh, eps)
    return wql_uh



def metric_msis(panel_lu: pd.DataFrame, confidence: float, naive_mae: float) -> Tuple[pd.Series, float]:
    """
    Mean Scaled Interval Score:
    MIS per row = (U-L) + 2/(1-conf)*(L - y)*1[y<L] + 2/(1-conf)*(y - U)*1[y>U]
    MSIS_h = mean(MIS_row) / naive_mae
    """
    if np.isnan(naive_mae) or naive_mae == 0:
        # Cannot scale properly
        mis_h = panel_lu.groupby("h").apply(lambda g: np.nan)
        return mis_h, np.nan

    L = panel_lu["lower"].values
    U = panel_lu["upper"].values
    y = panel_lu["y"].values
    alpha = 1.0 - confidence

    width = U - L
    below = (y < L)
    above = (y > U)
    penalty = (2.0 / alpha) * ((L - y) * below + (y - U) * above)
    mis_row = width + penalty
    panel_lu = panel_lu.copy()
    panel_lu["mis"] = mis_row

    mis_h = panel_lu.groupby("h")["mis"].mean().sort_index()
    msis_h = mis_h / naive_mae
    return msis_h, msis_h.mean()


def metric_msis_by_series(panel_lu: pd.DataFrame, confidence: float, naive_mae: float) -> pd.Series:
    """
    MSIS per (unique_id,h), scaled by global naive_mae to keep comparability across series.
    """
    L = panel_lu["lower"].values
    U = panel_lu["upper"].values
    y = panel_lu["y"].values
    alpha = 1.0 - confidence

    width = U - L
    below = (y < L)
    above = (y > U)
    penalty = (2.0 / alpha) * ((L - y) * below + (y - U) * above)
    mis_row = width + penalty
    df = panel_lu.copy()
    df["mis"] = mis_row

    mis_uh = df.groupby(["unique_id", "h"])["mis"].mean().sort_index()
    if np.isnan(naive_mae) or naive_mae == 0:
        return mis_uh * np.nan
    return mis_uh / naive_mae


def compute_all_metrics(
    obs_csv: str,
    median_csv: str,
    quantile_csvs: Dict[float, str],
    freq: Union[str, pd.DateOffset],
    q_low: float = 0.1,
    q_high: float = 0.9,
    confidence: float = 0.8,
    cce_S: List[float] = (0.8, 0.6, 0.4, 0.2),
    wql_reduce: str = "mean",
    wql_scale_mode: str = "sum_y",
):
    """
    Compute metrics overall (per-horizon and aggregated) and per-series breakdowns.
    Returns:
      summary: dict
      per_h_df: DataFrame with per-h metrics (overall)
      per_series_per_h_df: DataFrame with MultiIndex (unique_id, h), columns = metrics
      per_series_summary_df: DataFrame per series (mean across horizons)
    """
    # Load data
    y_df = load_observations(obs_csv)
    fw_median = read_forecast_wide(median_csv)
    median_long = melt_forecast_to_long(fw_median, freq=freq, value_name="yhat")
    median_aligned = align_with_truth(median_long, y_df)

    q_long = build_quantile_long(quantile_csvs, freq=freq)
    q_aligned = align_with_truth(q_long, y_df)

    # Panels
    panel_lu = build_lower_upper_panel(q_aligned, q_low=q_low, q_high=q_high)
    q_pivot = build_quantile_pivot(q_aligned)

    # Naive MAE
    naive_mae = compute_naive_mae(y_df, freq=freq)

    # Overall metrics per horizon
    mae_h, mae_mean, mase_h, mase_mean = metric_mae_and_mase(median_aligned, naive_mae)
    pce_h, pce_mean = metric_pce(q_aligned)
    tpce_h, tpce_mean = metric_tpce(panel_lu, confidence=confidence)
    tcce_h, tcce_mean = metric_tcce_from_pivot(q_pivot, confidence=confidence)
    cce_h, cce_h_by_s, cce_mean = metric_cce_multi_from_pivot(q_pivot, list(cce_S))
    siw_h, siw_h_by_s, siw_mean = metric_siw_multi_from_pivot(q_pivot, y_df, list(cce_S))
    wql_h, wql_total, wql_hq = metric_wql(q_aligned, reduce=wql_reduce, scale_mode=wql_scale_mode)
    msis_h, msis_mean = metric_msis(panel_lu, confidence=confidence, naive_mae=naive_mae)

    # Per-series per horizon
    mae_uh, mase_uh = metric_mae_and_mase_by_series(median_aligned, naive_mae)
    pce_uh = metric_pce_by_series(q_aligned)
    tpce_uh = metric_tpce_by_series(panel_lu, confidence=confidence)
    tcce_uh = metric_tcce_by_series_from_pivot(q_pivot, confidence=confidence)
    cce_uh, cce_uh_by_s = metric_cce_multi_by_series_from_pivot(q_pivot, list(cce_S))
    siw_uh, siw_uh_by_s = metric_siw_multi_by_series_from_pivot(q_pivot, y_df, list(cce_S))
    wql_uh = metric_wql_by_series(q_aligned, reduce=wql_reduce, scale_mode=wql_scale_mode)
    msis_uh = metric_msis_by_series(panel_lu, confidence=confidence, naive_mae=naive_mae)

    # Assemble overall per-h
    all_horizons = sorted(set(median_aligned["h"]))
    per_h = pd.DataFrame(index=all_horizons)
    def _add(series: pd.Series, name: str):
        s = series.reindex(per_h.index)
        per_h[name] = s.values

    _add(mae_h, "MAE")
    _add(mase_h, "MASE")
    _add(pce_h, "PCE")
    _add(tpce_h, "TPCE")
    _add(cce_h, "CCE")      # signed, averaged over S
    _add(tcce_h, "TCCE")    # signed, single interval
    _add(siw_h, "SIW")
    _add(wql_h, "WQL")
    _add(msis_h, "MSIS")

    # Assemble per-series per-h
    idx = sorted(set(mae_uh.index))  # (unique_id, h)
    per_series_per_h = pd.DataFrame(index=pd.MultiIndex.from_tuples(idx, names=["unique_id", "h"]))
    def _add_uh(series: pd.Series, name: str):
        per_series_per_h[name] = series.reindex(per_series_per_h.index)

    _add_uh(mae_uh, "MAE")
    _add_uh(mase_uh, "MASE")
    _add_uh(pce_uh, "PCE")
    _add_uh(tpce_uh, "TPCE")
    _add_uh(cce_uh, "CCE")
    _add_uh(tcce_uh, "TCCE")
    _add_uh(siw_uh, "SIW")
    _add_uh(wql_uh, "WQL")
    _add_uh(msis_uh, "MSIS")

    # Per-series summary (mean over horizons)
    per_series_summary = (
        per_series_per_h.groupby(level="unique_id").mean(numeric_only=True).sort_index()
    )

    summary = {
        "MAE": {"per_h": mae_h, "mean_over_h": mae_mean},
        "MASE": {"per_h": mase_h, "mean_over_h": mase_mean, "naive_mae": naive_mae},
        "PCE": {"per_h": pce_h, "mean_over_h": pce_mean},
        "TPCE": {"per_h": tpce_h, "mean_over_h": tpce_mean},
        "CCE": {
            "per_h": cce_h,
            "per_h_by_s": cce_h_by_s,
            "S": list(cce_S),
            "mean_over_h": cce_mean,
            "signed": True
        },
        "TCCE": {"per_h": tcce_h, "mean_over_h": tcce_mean, "signed": True, "confidence": confidence},
        "SIW": {
            "per_h": siw_h,
            "per_h_by_s": siw_h_by_s,
            "mean_over_h": siw_mean,
            "S": list(cce_S),
        },
        "WQL": {
            "per_h": wql_h,
            "aggregate": wql_total,       # mean over h if reduce='mean', else sum
            "reduce": wql_reduce,
            "scale_mode": wql_scale_mode,
            "per_h_per_q": wql_hq,
        },
        "MSIS": {"per_h": msis_h, "mean_over_h": msis_mean, "confidence": confidence},
        "PER_SERIES": {
            "per_h": per_series_per_h,        # MultiIndex (unique_id,h)
            "summary": per_series_summary,     # one row per unique_id
            "S": list(cce_S),
        }
    }

    return summary, per_h.reset_index().rename(columns={"index": "h"}), per_series_per_h, per_series_summary


def collect_model_head_files(results_root: str, model_name: str, head_name: str) -> Tuple[str, Dict[float, str]]:
    """
    From directory layout:
      results_root/model_name/head_name/
        10_preds.csv, 20_preds.csv, ..., 90_preds.csv, median_preds.csv
    Returns:
      median_csv_path, quantiles_map {0.1: path_to_10_preds.csv, ..., 0.9: path}
    Accepts any NN_preds.csv with NN in 1..99 -> q=NN/100; ignores out-of-range.
    """
    dir_path = os.path.join(results_root, model_name, head_name)
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    files = os.listdir(dir_path)
    # Find median
    median_csv = None
    for fn in files:
        if fn.lower() == "median_preds.csv":
            median_csv = os.path.join(dir_path, fn)
            break
    if median_csv is None:
        raise FileNotFoundError(f"median_preds.csv not found in {dir_path}")

    # Find quantiles like "10_preds.csv"
    qmap: Dict[float, str] = {}
    pat = re.compile(r"^(\d+)_preds\.csv$", flags=re.IGNORECASE)
    for fn in files:
        m = pat.match(fn)
        if not m:
            continue
        n = int(m.group(1))
        q = n / 100.0
        if 0.0 < q < 1.0:
            qmap[q] = os.path.join(dir_path, fn)

    # Helpful check: ensure we have at least lower and upper bounds for intervals
    if not any(np.isclose(list(qmap.keys()), 0.1)) or not any(np.isclose(list(qmap.keys()), 0.9)):
        # Not fatal, but warn
        print("Warning: Did not find both 10_preds.csv and 90_preds.csv. Interval metrics may fail if missing.")

    return median_csv, qmap


def main():
    parser = argparse.ArgumentParser(description="Compute TS forecasting metrics (MASE, SIW, PCE, TPCE, CCE, TCCE, WQL, MSIS) overall, per-horizon, and per-series.")
    parser.add_argument("--obs", required=True, help="Path to observations CSV (columns: unique_id, ds, y)")
    parser.add_argument("--results_root", required=True, help="Root directory containing model_results")
    parser.add_argument("--model_name", required=True, help="Model name subdirectory under results_root")
    parser.add_argument("--head_name", required=True, help="Head name subdirectory under model_name")
    parser.add_argument("--freq", required=True, help="Pandas offset alias like 'D','H','15min','W'")
    parser.add_argument("--q_low", type=float, default=0.1, help="Lower quantile for single-interval metrics (default 0.1)")
    parser.add_argument("--q_high", type=float, default=0.9, help="Upper quantile for single-interval metrics (default 0.9)")
    parser.add_argument("--confidence", type=float, default=0.8, help="Confidence used by TPCE/TCCE/SIW/MSIS (default 0.8)")
    parser.add_argument("--cce_S", default="0.8,0.6,0.4,0.2", help="Comma-separated list of symmetric interval coverages for CCE (e.g., 0.8,0.6,0.4,0.2)")
    parser.add_argument("--wql_reduce", default="mean", choices=["mean","sum"], help="Reduce WQL across forecast origins per horizon")
    parser.add_argument("--wql_scale_mode", default="sum_y", choices=["auto","mean_y","sum_y","none"], help="Scaling mode for WQL per horizon/series")
    parser.add_argument("--out_per_h", default="", help="Optional path to write overall per-horizon metrics CSV")
    parser.add_argument("--out_summary", default="", help="Optional path to write overall summary (JSON)")
    parser.add_argument("--out_per_h_by_series", default="", help="Optional path to write per-series per-horizon metrics CSV")
    parser.add_argument("--out_series_summary", default="", help="Optional path to write per-series summary (CSV)")

    args = parser.parse_args()

    # Resolve files from directory layout
    median_csv, qmap = collect_model_head_files(args.results_root, args.model_name, args.head_name)

    S = [float(x) for x in args.cce_S.split(",") if x.strip()]

    summary, per_h_df, per_series_per_h_df, per_series_summary_df = compute_all_metrics(
        obs_csv=args.obs,
        median_csv=median_csv,
        quantile_csvs=qmap,
        freq=args.freq,
        q_low=args.q_low,
        q_high=args.q_high,
        confidence=args.confidence,
        cce_S=S,
        wql_reduce=args.wql_reduce,
        wql_scale_mode=args.wql_scale_mode,
    )

    # Print concise overall summary
    print("=== Aggregated (mean over horizons unless noted) ===")
    print(f"Model={args.model_name} | Head={args.head_name}")
    print(f"MAE:   {summary['MAE']['mean_over_h']:.6f}")
    print(f"MASE:  {summary['MASE']['mean_over_h']:.6f} (naive MAE={summary['MASE']['naive_mae']:.6f})")
    print(f"PCE:   {summary['PCE']['mean_over_h']:.6f}")
    print(f"TPCE:  {summary['TPCE']['mean_over_h']:.6f}")
    print(f"CCE(S={summary['CCE']['S']}): {summary['CCE']['mean_over_h']:.6f} (signed, averaged over S)")
    print(f"TCCE(conf={summary['TCCE']['confidence']}): {summary['TCCE']['mean_over_h']:.6f} (signed, single interval)")
    print(f"SIW:   {summary['SIW']['mean_over_h']:.6f} (averaged over S={summary['SIW']['S']})")
    print(f"WQL:   aggregate={summary['WQL']['aggregate']:.6f} (reduce={summary['WQL']['reduce']}, scale={summary['WQL']['scale_mode']})")
    print(f"MSIS:  {summary['MSIS']['mean_over_h']:.6f} (confidence={summary['MSIS']['confidence']})")

    # Save outputs
    if args.out_per_h:
        per_h_df.to_csv(args.out_per_h, index=False)
        print(f"Per-horizon metrics written to: {args.out_per_h}")

    if args.out_per_h_by_series:
        # Flatten MultiIndex to columns for CSV
        out = per_series_per_h_df.reset_index()
        out.to_csv(args.out_per_h_by_series, index=False)
        print(f"Per-series per-horizon metrics written to: {args.out_per_h_by_series}")

    if args.out_series_summary:
        per_series_summary_df.reset_index().to_csv(args.out_series_summary, index=False)
        print(f"Per-series summary written to: {args.out_series_summary}")

    if args.out_summary:
        import json

        def ser_to_dict(ser: pd.Series):
            # Works for index that are ints (h)
            return {int(k): (None if pd.isna(v) else float(v)) for k, v in ser.items()}

        out = {
            "MAE": {"mean_over_h": summary["MAE"]["mean_over_h"], "per_h": ser_to_dict(summary["MAE"]["per_h"])},
            "MASE": {
                "mean_over_h": summary["MASE"]["mean_over_h"],
                "naive_mae": summary["MASE"]["naive_mae"],
                "per_h": ser_to_dict(summary["MASE"]["per_h"]),
            },
            "PCE": {"mean_over_h": summary["PCE"]["mean_over_h"], "per_h": ser_to_dict(summary["PCE"]["per_h"])},
            "TPCE": {"mean_over_h": summary["TPCE"]["mean_over_h"], "per_h": ser_to_dict(summary["TPCE"]["per_h"])},
            "CCE": {
                "S": summary["CCE"]["S"],
                "mean_over_h": summary["CCE"]["mean_over_h"],
                "per_h": ser_to_dict(summary["CCE"]["per_h"]),
            },
            "TCCE": {"mean_over_h": summary["TCCE"]["mean_over_h"], "per_h": ser_to_dict(summary["TCCE"]["per_h"])},
            "SIW": {
                "S": summary['SIW']['S'],
                "mean_over_h": summary["SIW"]["mean_over_h"],
                "per_h": ser_to_dict(summary["SIW"]["per_h"]),
            },
            "WQL": {
                "aggregate": summary["WQL"]["aggregate"],
                "reduce": summary["WQL"]["reduce"],
                "scale_mode": summary["WQL"]["scale_mode"],
                "per_h": ser_to_dict(summary["WQL"]["per_h"]),
            },
            "MSIS": {"mean_over_h": summary["MSIS"]["mean_over_h"], "per_h": ser_to_dict(summary["MSIS"]["per_h"])},
        }
        with open(args.out_summary, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Summary written to: {args.out_summary}")


if __name__ == "__main__":
    main()