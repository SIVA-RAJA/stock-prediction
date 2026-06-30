import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.metrics import (f1_score, precision_score, recall_score, classification_report)
from .lstm_config import DEVICE, CHECKPOINT_DIR
from data.config import TICKER_TO_ID, INTERVAL_TO_ID
from data.scaler import load_scaler


log = logging.getLogger(__name__)
RESULTS_DIR = CHECKPOINT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


ID_TO_TICKER = {v: k for k, v in TICKER_TO_ID.items()}
ID_TO_INTERVAL = {v: k for k, v in INTERVAL_TO_ID.items()}

def _inverse_close(scaled_values: np.ndarray, ticker: str, interval: str,) -> np.ndarray:

    bundle = load_scaler(ticker, interval)
    if bundle is None:
        log.warning(f"No scaler for {ticker} @ {interval} - returning scaled values")
        return scaled_values

    scaler, cols = bundle
    if "close" not in cols:
        return scaled_values

    close_idx = cols.index("close")

    dummy = np.zeros((len(scaled_values), len(cols)))
    dummy[:, close_idx] = scaled_values
    inv = scaler.inverse_transform(dummy)
    return inv[:, close_idx]

def _inverse_close_all(scaled_values: np.ndarray, ticker_ids: np.ndarray, interval_ids: np.ndarray) -> np.ndarray:

    out = np.empty_like(scaled_values, dtype=np.float64)
    pairs = np.stack([ticker_ids, interval_ids], axis=1)
    unique_pairs = np.unique(pairs, axis=0)

    for ticker_id, interval_id in unique_pairs:
        ticker = ID_TO_TICKER.get(int(ticker_id))
        interval = ID_TO_INTERVAL.get(int(interval_id))
        mask = (ticker_ids == ticker_id) & (interval_ids == interval_id)

        if ticker is None or interval is None:
            log.warning(f"Unknown ticker_id={ticker_id}/interval_id={interval_id}; leaving {mask.sum()} samples scaled")
            out[mask] = scaled_values[mask]
            continue

        out[mask] = _inverse_close(scaled_values[mask], ticker, interval)

    return out


def evaluate(model: nn.Module, test_loader: DataLoader, run_name: str="eval", ) -> dict:
    model.eval()
    model.to(DEVICE)

    all_price_pred, all_price_true = [], []
    all_dir_pred, all_dir_true = [], []
    all_attn = []
    all_emb = []

    with torch.no_grad():
        for x_num, x_emb, y_price, y_dir in test_loader:
            x_num = x_num.to(DEVICE)
            x_emb = x_emb.to(DEVICE)

            price_pred, dir_pred, attn = model(x_num, x_emb)

            dir_pred = torch.sigmoid(dir_pred)

            all_price_pred.append(price_pred.squeeze(-1).cpu().numpy())
            all_price_true.append(y_price.numpy())
            all_dir_pred.append(dir_pred.squeeze(-1).cpu().numpy())
            all_dir_true.append(y_dir.numpy())
            all_attn.append(attn.cpu().numpy())
            all_emb.append(x_emb.cpu().numpy())

    price_pred_sc = np.concatenate(all_price_pred)
    price_true_sc = np.concatenate(all_price_true)
    dir_pred_raw = np.concatenate(all_dir_pred)
    dir_true = np.concatenate(all_dir_true)
    attn_all = np.concatenate(all_attn)
    emb_all = np.concatenate(all_emb)
    ticker_ids, interval_ids = emb_all[:, 0], emb_all[:, 3]

    log.info("Inverse-scaling predictions per ticker before computing metrics...")
    price_pred_sc = _inverse_close_all(price_pred_sc, ticker_ids, interval_ids)
    price_true_sc = _inverse_close_all(price_true_sc, ticker_ids, interval_ids)

    mae = np.mean(np.abs(price_pred_sc - price_true_sc))
    rmse = np.sqrt(np.mean((price_pred_sc - price_true_sc) ** 2))
    mape = np.mean(np.abs((price_pred_sc - price_true_sc) / np.where(price_true_sc == 0, 1e-8, price_true_sc))) * 100

    dir_binary = (dir_pred_raw >= 0.5).astype(int)
    dir_true_i = dir_true.astype(int)
    dir_acc = (dir_binary == dir_true_i).mean()
    f1 = f1_score(dir_true_i, dir_binary, zero_division=0)
    precision = precision_score(dir_true_i, dir_binary, zero_division=0)
    recall = recall_score(dir_true_i, dir_binary, zero_division=0)

    returns = np.diff(price_true_sc)
    signals = dir_binary[:-1] * 2 - 1
    strat_ret = signals * returns
    sharpe_proxy = (strat_ret.mean() / (strat_ret.std() + 1e-8) * np.sqrt(252))

    metrices = {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "MAPE_%": round(float(mape), 4),
        "Dir_Accuracy": round(float(dir_acc), 4),
        "F1": round(float(f1), 4),
        "Precision": round(float(precision), 4),
        "Recall": round(float(recall), 4),
        "Sharpe_proxy": round(float(sharpe_proxy), 4),
    }

    log.info("\n ----- Evaluation Results ---------------------------")
    for k, v in metrices.items():
        log.info(f"{k:<18}: {v}")
    log.info(classification_report(dir_true_i, dir_binary, target_names=["DOWN", "UP"], zero_division=0))

    pred_df = pd.DataFrame({
        "true_price": price_true_sc,
        "pred_price": price_pred_sc,
        "true_dir": dir_true_i,
        "pred_dir": dir_binary,
        "dir_prob": dir_pred_raw,
    })

    csv_path = RESULTS_DIR / f"{run_name}_predictions.csv"
    pred_df.to_csv(csv_path, index=False)
    log.info(f"Predictions saved -> {csv_path}")

    fig, axes = plt.subplots(2, 1, figsize=(14,8))

    axes[0].plot(price_true_sc, label="True Price", alpha=0.8, linewidth=1.2)
    axes[0].plot(price_pred_sc, label="Pred Price", alpha=0.8, linewidth=1.0, linestyle="--")
    axes[0].set_title(f"Price Prediction (Test Set)")
    axes[0].set_ylabel("Price")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(dir_pred_raw, label="Up Probability", alpha=0.7)
    axes[1].axhline(0.5, color="red", linestyle="--", linewidth=0.8)
    axes[1].set_title("Direction Probability")
    axes[1].set_ylabel("P(up)")
    axes[1].set_xlabel("Sample")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = RESULTS_DIR / f"{run_name}_predictions.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    log.info(f"Plot saved -> {plot_path}")

    mean_attn = attn_all.mean(axis=0)
    fig2, ax2 = plt.subplots(figsize=(12, 2))
    ax2.bar(range(len(mean_attn)), mean_attn, color="steelblue")
    ax2.set_xlabel("Timestep (0 = oldest, -1 = most recent)")
    ax2.set_ylabel("Attention Weight")
    ax2.set_title("Mean Attention Over Test Set")
    plt.tight_layout()
    attn_path = RESULTS_DIR / f"{run_name}_attention.png"
    plt.savefig(attn_path, dpi=150)
    plt.close()
    log.info(f"Attention plot ssaved ->{attn_path}")

    return metrices
