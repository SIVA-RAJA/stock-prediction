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
from data.config import ID_TO_INTERVAL, ID_TO_TICKER
from data.scaler import load_scaler


log = logging.getLogger(__name__)
RESULTS_DIR = CHECKPOINT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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


def attention_ablation_test(model, test_loader):

    model.eval()
    model.to(DEVICE)
    normal_preds = []
    ablated_preds = []

    def zero_context_hook(module, input, output):
        context, attn_weights = output
        return torch.zeros_like(context), attn_weights

    handle = model.attention.register_forward_hook(zero_context_hook)

    with torch.no_grad():
        for x_num, x_emb, y_price, y_dir in test_loader:
            x_num = x_num.to(DEVICE)
            x_emb = x_emb.to(DEVICE)

            pp_norm, _, _ = model(x_num, x_emb)
            normal_preds.append(pp_norm.cpu().numpy())

            if len(normal_preds) > 10:
                break

    handle.remove()

    normal = np.concatenate(normal_preds)
    diff = np.mean(np.abs(normal))

    log.info("------------ Attention Ablation Test ----------------")
    if diff < 0.001:
        log.warning("Model ignoring attention - predictions unchanged")
    else:
        log.info(f"Model using attention - mean changed : {diff:.6f})")

def plot_correlation(price_pred_sc, price_true_sc, run_name):

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(price_true_sc[:2000], price_pred_sc[:2000], alpha=0.3, s=10, color="steelblue")
    mn = min(price_true_sc.min(), price_pred_sc.min())
    mx = max(price_true_sc.max(), price_pred_sc.max())
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=2, label="Perfect Prediction")

    ss_res = np.sum((price_true_sc - price_pred_sc) ** 2)
    ss_tot = np.sum((price_true_sc - np.mean(price_true_sc)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    ax.set_xlabel("True Price (normalized)")
    ax.set_ylabel("Predicted Price (normalized)")
    ax.set_title(f"Predicted Vs. True | R²={r2:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = RESULTS_DIR / f"{run_name}_correlation.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    log.info(f"Correlation plot saved -> {plot_path}")

    log.info(f"R² score: {r2:.4f}")
    if r2 > 0.85:
        log.info("Model shows strong correlation - model learning well")
    elif r2 > 0.60:
        log.info("Model shows moderate correlation - model learning somewhat")
    else:
        log.warning("Model shows weak correlation - model not learning well")




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

    same_group = (ticker_ids[1:] == ticker_ids[:-1]) & (interval_ids[1:] == interval_ids[:-1])
    returns = np.diff(price_true_sc)[same_group]
    signals = (dir_binary[:-1] * 2 - 1)[same_group]
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

    naive_pred = np.roll(price_true_sc, 1)
    naive_pred[0] = price_true_sc[0]

    naive_mae = float(np.mean(np.abs(naive_pred - price_true_sc)))
    naive_rmse = float(np.sqrt(np.mean((naive_pred - price_true_sc) ** 2)))

    model_mae = float(np.mean(np.abs(price_pred_sc - price_true_sc)))
    model_rmse = float(np.sqrt(np.mean((price_pred_sc - price_true_sc) ** 2)))

    log.info(f"---------BASELINE COMPARISON----------------")
    log.info(f"Naive MAE: {naive_mae:.5f}")
    log.info(f"Model MAE: {model_mae:.5f}")
    log.info(f"Improvement: {((naive_mae - model_mae) / naive_mae * 100):.1f}%")
    log.info(f"Naive RMSE: {naive_rmse:.5f}")
    log.info(f"Model RMSE: {model_rmse:.5f}")

    if model_mae >= naive_mae:
        log.warning("Model is worse than naive baseline!")
        log.warning("Model is just copying the last value.")
    else:
        log.info("Model is beats naive baseline.")


    random_acc = 0.5
    always_up_acc = float(dir_true.mean())
    always_down_acc = float(1 - dir_true.mean())

    pred_up_ratio = float((dir_binary == 1).mean())

    log.info(f"---------DIRECTION ANALYSIS----------------")
    log.info(f"Model direction accuracy: {dir_acc:.4f}")
    log.info(f"Random direction accuracy: {random_acc:.4f}")
    log.info(f"Always UP direction accuracy: {always_up_acc:.4f}")
    log.info(f"Always DOWN direction accuracy: {always_down_acc:.4f}")
    log.info(f"Model predicted UP ratio: {pred_up_ratio:.1%} of time")

    if abs(pred_up_ratio - 0.5) > 0.3:
        log.warning("Model heavily biased towards one direction.")
        log.warning("Increase LAMBDA_DIR or check class balance.")
    elif dir_acc <= 0.51:
        log.warning("Model is not performing significantly better than random.")
    else:
        log.info("Model shows real directional skill.")

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

    plot_correlation(price_pred_sc, price_true_sc, run_name)

    ss_res = np.sum((price_true_sc - price_pred_sc) ** 2)
    ss_tot = np.sum((price_true_sc - np.mean(price_true_sc)) **2)
    r2 = float(1 - (ss_res / ss_tot))

    attention_ablation_test(model, test_loader)

    _print_scorecard(metrices=metrices, naive_mae=naive_mae, model_mae=model_mae, dir_acc=dir_acc, pred_up_ratio=pred_up_ratio, r2=r2)

    return metrices

def _print_scorecard(metrices, naive_mae, model_mae, dir_acc, pred_up_ratio, r2):
    log.info("\n" + "="*60)
    log.info("--------------MODEL SCORECARD----------------------")
    log.info("="*60)
    checks = {
        "Beats naive MAE baseline": model_mae < naive_mae,
        "Direction accuracy > 52%": dir_acc > 0.52,
        "Direction accuracy > 55%": dir_acc > 0.55,
        "Not biased (predicts both UP and DOWN)": 0.3 < pred_up_ratio < 0.7,
        "F1 score > 0.40": metrices.get("F1") > 0.40,
        "R² Correlation > 0.60": r2 > 0.60,
        "Positive Sharpe proxy": metrices.get("Sharpe_proxy") > 0.0,
        "MAPE < 5%": metrices.get("MAPE_%") < 5.0,
    }

    passed = 0
    for check, result in checks.items():
        status = "PASS" if result else "FAIL"
        log.info(f"{check}: {status}")
        if result:
            passed += 1
    log.info(f"\n Score: {passed}/{len(checks)} checks passed.")

    if passed >= 7:
        log.info("EXCELLENT MODEL PERFORMANCE - model is genuinely learning.")
    elif passed >= 5:
        log.info("GOOD MODEL PERFORMANCE - model shows real skill")
    elif passed >= 3:
        log.info("FAIR MODEL PERFORMANCE - needs more data or tuning")
    else:
        log.info("POOR MODEL PERFORMANCE - model is not learning meaningfully")

    log.info("=" * 60)
