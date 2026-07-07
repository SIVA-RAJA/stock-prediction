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
from .lstm_config import DEVICE, EVAL_DIR
from data.config import ID_TO_INTERVAL, ID_TO_TICKER
from data.scaler import load_scaler


log = logging.getLogger(__name__)
RESULTS_DIR = EVAL_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)



def _inverse_close_all(scaled_values: np.ndarray, market_ids: np.ndarray, interval_ids: np.ndarray, col_name) -> np.ndarray:

    out = np.empty_like(scaled_values, dtype=np.float64)
    pairs = np.stack([market_ids, interval_ids], axis=1)
    unique_pairs = np.unique(pairs, axis=0)

    for market_id, interval_id in unique_pairs:
        market = ID_TO_TICKER.get(int(market_id))
        interval = ID_TO_INTERVAL.get(int(interval_id))
        mask = (market_ids == market_id) & (interval_ids == interval_id)

        if market is None or interval is None:
            log.warning(f"Unknown market_id={market_id}/interval_id={interval_id}; leaving {mask.sum()} samples scaled")
            out[mask] = scaled_values[mask]
            continue

        bundle = load_scaler(market, interval)
        if bundle is None or col_name not in bundle[1]:
            log.warning(f"No scaler for {market} @ {interval} or column '{col_name}' not found; leaving {mask.sum()} samples scaled")
            out[mask] = scaled_values[mask]
            continue

        scaler, cols = bundle
        idx = cols.index(col_name)
        dummy = np.zeros((mask.sum(), len(cols)))
        dummy[:, idx] = scaled_values[mask]

        out[mask] = scaler.inverse_transform(dummy)[:, idx]

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
        for x_num, x_emb, y_dir, _ in test_loader:
            x_num = x_num.to(DEVICE)
            x_emb = x_emb.to(DEVICE)

            dir_out, _ = model(x_num, x_emb)
            normal_preds.append(dir_out.cpu().numpy())

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



def evaluate(model: nn.Module, test_loader: DataLoader, run_name: str="eval", ) -> dict:
    model.eval()
    model.to(DEVICE)

    all_dir_pred, all_dir_true = [], []
    all_attn = []
    all_emb = []
    all_last_close = []

    with torch.no_grad():
        for x_num, x_emb, y_dir, last_close in test_loader:
            x_num = x_num.to(DEVICE)
            x_emb = x_emb.to(DEVICE)

            dir_pred, attn = model(x_num, x_emb)

            dir_pred = torch.sigmoid(dir_pred)

            all_last_close.append(last_close.numpy())
            all_dir_pred.append(dir_pred.squeeze(-1).cpu().numpy())
            all_dir_true.append(y_dir.numpy())
            all_attn.append(attn.cpu().numpy())
            all_emb.append(x_emb.cpu().numpy())


    last_close_sc = np.concatenate(all_last_close)
    dir_pred_raw = np.concatenate(all_dir_pred)
    dir_true = np.concatenate(all_dir_true)
    attn_all = np.concatenate(all_attn)
    emb_all = np.concatenate(all_emb)
    market_ids, interval_ids, ticker_ids = emb_all[:, 0], emb_all[:, 2], emb_all[:, 3]

    log.info("Inverse-scaling predictions per ticker before computing metrics...")
    raw_last_close = _inverse_close_all(last_close_sc, market_ids, interval_ids, col_name="close")

    dir_binary = (dir_pred_raw >= 0.5).astype(int)
    dir_true_i = dir_true.astype(int)
    dir_acc = (dir_binary == dir_true_i).mean()
    f1 = f1_score(dir_true_i, dir_binary, zero_division=0)
    precision = precision_score(dir_true_i, dir_binary, zero_division=0)
    recall = recall_score(dir_true_i, dir_binary, zero_division=0)

    same_group = (ticker_ids[1:] == ticker_ids[:-1]) & (interval_ids[1:] == interval_ids[:-1])
    returns = np.diff(raw_last_close)[same_group]
    signals = (dir_binary[:-1] * 2 - 1)[same_group]
    strat_ret = signals * returns
    sharpe_proxy = (strat_ret.mean() / (strat_ret.std() + 1e-8) * np.sqrt(252))

    metrices = {
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
        "true_dir": dir_true_i,
        "pred_dir": dir_binary,
        "dir_prob": dir_pred_raw,
    })

    csv_path = RESULTS_DIR / f"{run_name}_predictions.csv"
    pred_df.to_csv(csv_path, index=False)
    log.info(f"Predictions saved -> {csv_path}")

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(dir_pred_raw, label="Up Probability", alpha=0.7)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8)
    ax.set_title("Direction Probability")
    ax.set_ylabel("P(up)")
    ax.set_xlabel("Sample")
    ax.legend()
    ax.grid(True, alpha=0.3)

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
    log.info(f"Attention plot saved -> {attn_path}")

    attention_ablation_test(model, test_loader)

    _print_scorecard(metrices=metrices, dir_acc=dir_acc, pred_up_ratio=pred_up_ratio)

    return metrices

def _print_scorecard(metrices, dir_acc, pred_up_ratio):
    log.info("\n" + "="*60)
    log.info("--------------MODEL SCORECARD----------------------")
    log.info("="*60)
    checks = {
        "Direction accuracy > 52%": dir_acc > 0.52,
        "Direction accuracy > 55%": dir_acc > 0.55,
        "Not biased (predicts both UP and DOWN)": 0.3 < pred_up_ratio < 0.7,
        "F1 score > 0.40": metrices.get("F1") > 0.40,
        "Positive Sharpe proxy": metrices.get("Sharpe_proxy") > 0.0,
    }

    passed = 0
    for check, result in checks.items():
        status = "PASS" if result else "FAIL"
        log.info(f"{check}: {status}")
        if result:
            passed += 1
    log.info(f"\n Score: {passed}/{len(checks)} checks passed.")

    if passed >= 5:
        log.info("EXCELLENT MODEL PERFORMANCE - model is genuinely learning.")
    elif passed >= 3:
        log.info("GOOD MODEL PERFORMANCE - model shows real skill")
    else:
        log.info("POOR MODEL PERFORMANCE - model is not learning meaningfully")

    log.info("=" * 60)
