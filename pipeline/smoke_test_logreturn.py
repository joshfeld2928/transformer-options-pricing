"""
Quick smoke-test training + inference for all four forecasting models, targeting
AAPL daily log return.

Pulls the trailing ~6 months of AAPL underlying data plus macro indicators,
trains each model (Autoformer, FEDformer, Informer, Pyraformer) with a short
full-batch loop on sliding windows of that history, then forecasts the next
PRED_LEN trading days of log returns. One CSV per model is written to
`prediction CSVs/log_return_<mon><day>/`.
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
from torch import nn, optim

from utils.company_data import _get_underlying_history
from utils.macro_indicators import get_macro_data
from inference.autoformer_inference import build_autoformer
from inference.fedformer_inference import build_fedformer
from inference.informer_inference import build_informer
from inference.pyraformer_inference import build_pyraformer

TICKER = "AAPL"
HISTORY_MONTHS = 6
SEQ_LEN = 60
LABEL_LEN = 30
PRED_LEN = 10
EPOCHS = 8
LR = 1e-3
SEED = 42

TARGET = "log_return"
FEATURE_COLUMNS = [
    "log_return",
    "underlying_close",
    "realized_vol",
    "VIX",
    "10Y_Treasury_Yield",
    "Daily_Breakeven_Inflation",
    "Official_Inflation_YoY",
]

# Small dims so the CPU smoke-test trains in seconds.
SMALL_STANDARD_KWARGS = dict(d_model=64, n_heads=4, e_layers=1, d_layers=1, d_ff=128, dropout=0.05, freq="d")
SMALL_PYRAFORMER_KWARGS = dict(d_model=64, d_inner_hid=64, d_bottleneck=32, d_k=32, d_v=32,
                               n_head=4, n_layer=2, window_size=[3, 3, 3])

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "prediction CSVs",
                          f"log_return_{datetime.today().strftime('%b').lower()}{datetime.today().day}")


def build_feature_panel():
    end_date = datetime.today()
    window_start = end_date - timedelta(days=30 * HISTORY_MONTHS)
    # Lead-in so the 21-day realized-vol window isn't NaN at window_start.
    pull_start = window_start - timedelta(days=45)
    start_str, end_str = pull_start.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

    underlying = _get_underlying_history(TICKER, start_str, end_str)
    macro = get_macro_data(start_date=start_str, end_date=end_str)

    panel = underlying.join(macro, how="left").ffill().dropna()
    panel = panel.loc[panel.index >= window_start]
    return panel[FEATURE_COLUMNS]


def time_features(dates, n_cols=3):
    """Normalized (month, day, weekday[, hour=0]) features; Pyraformer needs 4 columns."""
    dates = pd.DatetimeIndex(dates)
    cols = [dates.month.values / 12.0 - 0.5, dates.day.values / 31.0 - 0.5, dates.weekday.values / 6.0 - 0.5]
    if n_cols == 4:
        cols.append(np.zeros(len(dates)))
    return np.stack(cols, axis=-1).astype(np.float32)


def make_windows(values, marks):
    n_features = values.shape[1]
    enc_x, enc_mark, dec_x, dec_mark, target = [], [], [], [], []
    for i in range(len(values) - SEQ_LEN - PRED_LEN + 1):
        enc_x.append(values[i:i + SEQ_LEN])
        enc_mark.append(marks[i:i + SEQ_LEN])
        dec_hist = values[i + SEQ_LEN - LABEL_LEN:i + SEQ_LEN]
        dec_x.append(np.concatenate([dec_hist, np.zeros((PRED_LEN, n_features), np.float32)]))
        dec_mark.append(marks[i + SEQ_LEN - LABEL_LEN:i + SEQ_LEN + PRED_LEN])
        target.append(values[i + SEQ_LEN:i + SEQ_LEN + PRED_LEN])
    return [torch.from_numpy(np.stack(a)) for a in (enc_x, enc_mark, dec_x, dec_mark, target)]


def make_windows_pyraformer(values, marks4):
    """Encoder-only windows with the FC head's +1 zero predict token."""
    n_features = values.shape[1]
    enc_x, enc_mark, target = [], [], []
    for i in range(len(values) - SEQ_LEN - PRED_LEN + 1):
        enc_x.append(np.concatenate([values[i:i + SEQ_LEN], np.zeros((1, n_features), np.float32)]))
        enc_mark.append(marks4[i:i + SEQ_LEN + 1])
        target.append(values[i + SEQ_LEN:i + SEQ_LEN + PRED_LEN])
    enc_x, enc_mark, target = (torch.from_numpy(np.stack(a)) for a in (enc_x, enc_mark, target))
    dec_x = torch.zeros(len(enc_x), PRED_LEN, n_features)
    dec_mark = torch.zeros(len(enc_x), PRED_LEN, marks4.shape[1])
    return enc_x, enc_mark, dec_x, dec_mark, target


def forecast_inputs(values, marks, future_marks):
    n_features = values.shape[1]
    enc_x = values[-SEQ_LEN:]
    enc_mark = marks[-SEQ_LEN:]
    dec_x = np.concatenate([values[-LABEL_LEN:], np.zeros((PRED_LEN, n_features), np.float32)])
    dec_mark = np.concatenate([marks[-LABEL_LEN:], future_marks])
    return [torch.from_numpy(a[None, ...]) for a in (enc_x, enc_mark, dec_x, dec_mark)]


def forecast_inputs_pyraformer(values, marks4, future_marks4):
    n_features = values.shape[1]
    enc_x = np.concatenate([values[-SEQ_LEN:], np.zeros((1, n_features), np.float32)])
    enc_mark = np.concatenate([marks4[-SEQ_LEN:], future_marks4[:1]])
    return [torch.from_numpy(enc_x[None, ...]), torch.from_numpy(enc_mark[None, ...]),
            torch.zeros(1, PRED_LEN, n_features), torch.zeros(1, PRED_LEN, marks4.shape[1])]


def train(model, name, enc_x, enc_mark, dec_x, dec_mark, target):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        out = model(enc_x, enc_mark, dec_x, dec_mark)
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()
        print(f"  [{name}] epoch {epoch}/{EPOCHS} - loss {loss.item():.6f}")


def predict(model, inputs, future_dates, mean, std, last_close):
    model.eval()
    with torch.no_grad():
        pred = model(*inputs).squeeze(0).numpy()
    idx = FEATURE_COLUMNS.index(TARGET)
    log_ret = pred[:, idx] * std[idx] + mean[idx]
    return pd.DataFrame({
        "date": future_dates,
        "predicted_log_return": log_ret,
        "implied_close": last_close * np.exp(np.cumsum(log_ret)),
    })


def main():
    torch.manual_seed(SEED)
    panel = build_feature_panel()
    values = panel.values.astype(np.float32)
    mean, std = values.mean(axis=0), values.std(axis=0)
    std[std == 0] = 1.0
    scaled = ((values - mean) / std).astype(np.float32)
    n_features = scaled.shape[1]

    last_date = panel.index[-1]
    last_close = float(panel["underlying_close"].iloc[-1])
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=PRED_LEN)

    marks3 = time_features(panel.index)
    marks4 = time_features(panel.index, n_cols=4)
    future_marks3 = time_features(future_dates)
    future_marks4 = time_features(future_dates, n_cols=4)

    print(f"Loaded {len(panel)} daily rows for {TICKER} ({panel.index[0].date()} - {last_date.date()})")

    dims = dict(enc_in=n_features, dec_in=n_features, c_out=n_features, seq_len=SEQ_LEN, label_len=LABEL_LEN)
    standard_windows = make_windows(scaled, marks3)
    standard_inputs = forecast_inputs(scaled, marks3, future_marks3)

    runs = {
        "autoformer": (build_autoformer(**dims, pred_len=PRED_LEN, **SMALL_STANDARD_KWARGS),
                       standard_windows, standard_inputs),
        "fedformer": (build_fedformer(**dims, pred_len=PRED_LEN, modes=16, **SMALL_STANDARD_KWARGS),
                      standard_windows, standard_inputs),
        "informer": (build_informer(**dims, out_len=PRED_LEN, **SMALL_STANDARD_KWARGS),
                     standard_windows, standard_inputs),
        "pyraformer": (build_pyraformer(enc_in=n_features, c_out=n_features, input_size=SEQ_LEN,
                                        predict_step=PRED_LEN, device="cpu", **SMALL_PYRAFORMER_KWARGS),
                       make_windows_pyraformer(scaled, marks4),
                       forecast_inputs_pyraformer(scaled, marks4, future_marks4)),
    }
    print(f"Built {len(standard_windows[0])} training windows (seq_len={SEQ_LEN}, pred_len={PRED_LEN})")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for name, (model, windows, inputs) in runs.items():
        print(f"Training {name}...")
        train(model, name, *windows)
        out_path = os.path.join(OUTPUT_DIR, f"{name}_predictions.csv")
        predict(model, inputs, future_dates, mean, std, last_close).to_csv(out_path, index=False)
        print(f"Saved {name} predictions to {out_path}")



if __name__ == "__main__":
    main()
