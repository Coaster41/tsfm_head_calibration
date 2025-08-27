#!/usr/bin/env python3
"""
Run rolling-window forecasting for a backbone model and saved heads, and export
per-head predictions to CSVs.

- Loads a long-format CSV with columns [unique_id, ds, y]
- Builds a GluonTS PandasDataset and constructs rolling windows
- Uses forecaster.HeadForecaster to load backbone + head checkpoints
- For each head, collects median and quantile predictions and writes CSVs:
    save_dir/{head}/{median or q}_preds.csv

Notes:
- This script mirrors your notebook behavior, including using "item_id" in both
  the 'unique_id' and 'ds' columns in the output CSV (as in the provided code).
- If you'd like ds to be forecast start timestamps instead, change the
  'start_date_col' extraction to use an appropriate timestamp field.
"""

import argparse
import os
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from gluonts.itertools import batcher

# Your local module that wraps loading backbones and head checkpoints
from forecaster import ForecastConfig, HeadForecaster
QUANTILES = [10,20,30,40,50,60,70,80,90]

def load_test_data(
    pred_length: int,
    context: int,
    dataset_path: str,
    forecast_date: str,
    stride: int = 1,
) -> Tuple[object, str, str, object]:
    """
    Build a GluonTS rolling-window test dataset from a long-format CSV.

    Returns:
        test_data: generate_instances(...) result
        freq: inferred frequency string (e.g., '1H')
        unit: base unit (e.g., 'H', 'D', 'M')
        freq_delta: pandas DateOffset/Timedelta representing one 'freq' step
    """
    PDT = pred_length
    CTX = context

    # Load dataframe and GluonTS dataset
    df = pd.read_csv(dataset_path, index_col=0, parse_dates=["ds"])
    df["y"] = df["y"].astype(np.float32)

    ds = PandasDataset.from_long_dataframe(
        df, target="y", item_id="unique_id", timestamp="ds"
    )
    freq = ds.freq
    unit = "".join(char for char in freq if not char.isdigit())
    print(f"freq: {freq}, unit: {unit}")
    unit_str = "".join(filter(str.isdigit, freq))
    unit_num = int(unit_str) if unit_str != "" else 1

    if unit == "M":
        freq_delta = pd.DateOffset(months=unit_num)
    else:
        freq_delta = pd.Timedelta(unit_num, unit)

    # Determine forecast start date (if blank, start after context length)
    if forecast_date == "":
        forecast_date_ts = min(df["ds"]) + freq_delta * CTX
    else:
        forecast_date_ts = pd.Timestamp(forecast_date)

    end_date = max(df["ds"])
    if unit == "M":
        total_forecast_length = (
            (end_date.to_period(unit) - forecast_date_ts.to_period(unit)).n // unit_num + 1
        )
    else:
        total_forecast_length = (end_date - forecast_date_ts) // freq_delta

    # Split to get test template starting at forecast_date_ts
    _, test_template = split(ds, date=pd.Period(forecast_date_ts, freq=freq))

    # Build rolling-window instances
    test_data = test_template.generate_instances(
        prediction_length=PDT,                 # steps to predict
        windows=(total_forecast_length - PDT) // stride,   # number of windows
        distance=stride,                       # step between windows
        max_history=CTX,
    )
    return test_data, freq, unit, freq_delta


def parse_args():
    parser = argparse.ArgumentParser(description="Rolling-window forecasting with saved heads.")
    # Data/rolling window
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to CSV with columns [unique_id, ds, y].")
    parser.add_argument("--pred-length", type=int, default=64,
                        help="Prediction horizon.")
    parser.add_argument("--context-len", type=int, default=512,
                        help="Context length.")
    parser.add_argument("--forecast-date", type=str, default="2021-01-31 23:00:00",
                        help="Forecast start datetime. Empty string '' uses min(ds) + context steps.")
    parser.add_argument("--stride", type=int, default=1,
                        help="Stride between rolling windows.")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Batch size for rolling eval.")
    # Model and heads
    parser.add_argument("--model-name", type=str, default="timesfm",
                        choices=["timesfm", "moirai2", "chronos_bolt", "tirex", "yinglong"],
                        help="Backbone model name for HeadForecaster.")
    parser.add_argument("--ckpt-dir", type=str, default="models",
                        help="Directory containing head checkpoints (e.g., models/{model}_{head}_head.pt).")
    parser.add_argument("--heads", type=str, nargs="+",
                        default=["quantiles", "studentst", "gaussian", "mixture"],
                        help="Heads to evaluate; 'backbone' is added automatically.")
    parser.add_argument("--horizon-len", type=int, required=False,
                        default=None,
                        help="Sets a strict forecast horizon length")
    # Redundant...
    parser.add_argument("--include-backbone", action="store_true", default=True, 
                        help="Include 'backbone' as a pseudo-head for baseline forecasts.")
    # Output
    parser.add_argument("--save-dir", type=str, required=True,
                        help="Directory where predictions will be saved as CSVs.")
    # Device
    parser.add_argument("--device", type=str, default=None,
                        help="Device string for torch (e.g., 'cuda', 'cpu'). If omitted, uses cuda if available.")
    return parser.parse_args()


def main():
    args = parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    # 1) Build rolling test set
    test_data, freq, unit, freq_delta = load_test_data(
        pred_length=args.pred_length,
        context=args.context_len,
        dataset_path=args.dataset,
        forecast_date=args.forecast_date,
        stride=args.stride,
    )

    # 2) Build forecaster and load heads
    cfg = ForecastConfig(
        model_name=args.model_name,
        context_len=args.context_len,
        horizon_len=args.horizon_len,
        ckpt_dir=args.ckpt_dir,
        device=device_str,
    )
    forecaster = HeadForecaster(cfg)

    heads: List[str] = list(args.heads)
    loaded_heads = forecaster.load_heads(heads)
    print("Loaded heads:", loaded_heads)

    if args.include_backbone and "backbone" not in heads:
        heads.append("backbone")

    # 3) Iterate over rolling windows in batches and forecast
    os.makedirs(args.save_dir, exist_ok=True)
    start_time = time.time()

    item_id_col_batches = []
    start_date_col_batches = []
    forecasts = {h: [] for h in heads}

    for i, batch in enumerate(batcher(test_data.input, batch_size=args.batch_size)):
        t0 = time.time()
        # context tensor shape [B, T]
        context = torch.stack([torch.tensor(entry["target"]) for entry in batch])
        out = forecaster.forecast(context)  # dict: head -> {'median': [B,H], 'quantiles': [Q,B,H]}

        print(f"batch {i}: single iteration: {time.time() - t0:.3f}s | total: {time.time() - start_time:.3f}s")

        # The notebook stores item_id twice (both unique_id and ds). We keep this behavior for fidelity.
        item_id_col_batches.append(np.array([entry["item_id"] for entry in batch]))
        start_date_col_batches.append(np.array([entry["start"] for entry in batch]) + args.context_len - 1)

        for head in heads:
            # Stack median as the first row and then quantiles [Q,B,H] beneath => [1+Q, B, H]
            # Ensure all numpy arrays for saving
            median_np = np.asarray(out[head]["median"])
            quants_np = np.asarray(out[head]["quantiles"])
            combined = np.concatenate([median_np[None, ...], quants_np], axis=0)
            forecasts[head].append(combined)

    # 4) Concatenate all batches and write per-head CSVs
    item_id_col = np.concatenate(item_id_col_batches, axis=0)
    start_date_col = np.concatenate(start_date_col_batches, axis=0)

    for head in heads:
        head_dir = os.path.join(args.save_dir, head)
        os.makedirs(head_dir, exist_ok=True)

        # forecasts[head] is a list of [1+Q, B, H] -> concat on axis=1 -> [1+Q, total_B, H]
        head_forecasts = np.concatenate(forecasts[head], axis=1)

        # Build and save one CSV per "prediction type": median + each quantile
        pred_types = ["median"] + [str(q) for q in QUANTILES]
        if head_forecasts.shape[0] != len(pred_types):
            print(
                f"Warning: number of rows in head_forecasts[0] ({head_forecasts.shape[0]}) "
                f"!= 1 + len(quantiles) ({1 + len(QUANTILES)}). "
                f"Proceeding with min alignment."
            )
            # Align lengths conservatively
            min_len = min(head_forecasts.shape[0], len(pred_types))
            pred_types = pred_types[:min_len]
            head_forecasts = head_forecasts[:min_len]

        for pred_type_ind, pred_type in enumerate(pred_types):
            data_dict = {
                "unique_id": item_id_col,
                "ds": start_date_col,
            }
            # horizon = head_forecasts.shape[2]
            horizon = args.pred_length
            for step in range(horizon):
                data_dict[str(step + 1)] = head_forecasts[pred_type_ind, :, step]
            df_out = pd.DataFrame(data_dict)
            out_path = os.path.join(head_dir, f"{pred_type}_preds.csv")
            df_out.to_csv(out_path, index=False)
            print(f"Saved: {out_path}")

    print("All done.")


if __name__ == "__main__":
    main()