import logging
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score

from lstm_config import (
    DEVICE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, LAMBDA_PRICE, LAMBDA_DIR, GRAD_CLIP, PATIENCE,
    MIN_DELTA, SCHEDULER_TO, SCHEDULER_T_MULT, BEST_CKPT, LAST_CKPT, LOG_DIR,
)

log = logging.getLogger(__name__)

class MultiTaskLoss(nn.Module):
    def __init__(self, lambda_price=LAMBDA_PRICE, lambda_dir=LAMBDA_DIR):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()
        self.lp = lambda_price
        self.ld = lambda_dir

    def forward(self, price_pred, price_true, dir_pred, dir_true) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        loss_price = self.mse(price_pred.squeeze(), price_true)
        loss_dir = self.bce(dir_pred.squeeze(), dir_true)
        total = self.lp * loss_price + self.ld * loss_dir
        return total, loss_price, loss_dir

def _compute_metrics(price_preds: np.ndarray, price_trues: np.ndarray, dir_preds: np.ndarray, dir_trues: np.ndarray) -> dict:

    mae = np.mean(np.abs(price_preds - price_trues))
    rmse = np.sqrt(np.mean((price_preds - price_trues) ** 2))
    dir_binary = (dir_preds >= 0.5).astype(int)
    dir_acc = (dir_binary == dir_trues.astype(int)).mean()
    f1 = f1_score(dir_trues.astype(int), dir_binary, zero_division=0)

    return {
        "mae": mae,
        "rmse": rmse,
        "dir_acc": dir_acc,
        "f1": f1,
    }

def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: MultiTaskLoss,
    optimizer: torch.optim.Optimizer | None,
    is_train: bool,
) -> tuple[float, float, float, dict]:

    model.train() if is_train else model.eval()

    total_loss = price_loss_sum = dir_loss_sum = 0.0
    all_price_pred, all_price_true = [], []
    all_dir_pred, all_dir_true = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for x_num, x_emb, y_price, y_dir in loader:
            x_num = x_num.to(DEVICE)
            x_emb = x_emb.to(DEVICE)
            y_price = y_price.to(DEVICE)
            y_dir = y_dir.to(DEVICE)

            price_pred, dir_pred, _ = model(x_num, x_emb)
            loss, lp, ld = criterion(price_pred, y_price, dir_pred, y_dir)

            if is_train:
                assert optimizer is not None, "optimizer must be provided when is_train=True"
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            total_loss += loss.item()
            price_loss_sum += lp.item()
            dir_loss_sum += ld.item()

            all_price_pred.append(price_pred.squeeze().detach().cpu().numpy())
            all_price_true.append(y_price.detach().cpu().numpy())
            all_dir_pred.append(dir_pred.squeeze().detach().cpu().numpy())
            all_dir_true.append(y_dir.detach().cpu().numpy())

    n = len(loader)
    metrics = _compute_metrics(
        np.concatenate(all_price_pred),
        np.concatenate(all_price_true),
        np.concatenate(all_dir_pred),
        np.concatenate(all_dir_true),
    )

    return total_loss / n, price_loss_sum / n, dir_loss_sum / n, metrics


class EarlyStopping:
    def __init__(self, patience=PATIENCE, min_delta=MIN_DELTA):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def _save_checkpoint(model, optimizer,scheduler, epoch,val_loss, path):

    torch.save({
        "epoch": epoch,
        "val_loss": val_loss,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, path)
    log.info(f"Checkpoint saved at {path.name} (val_loss={val_loss:.6f})")

def load_checkpoint(model, optimizer=None, scheduler=None, path=BEST_CKPT):

    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler:
        scheduler.load_state_dict(ckpt["scheduler"])
    log.info(f"Loaded checkpoint from epoch {ckpt['epoch']} val_loss={ckpt['val_loss']:.6f}")

    return ckpt["epoch"], ckpt["val_loss"]

def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, run_name: str="run") -> nn.Module:

    model.to(DEVICE)
    log.info(f"Training on {DEVICE}")
    log.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = MultiTaskLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=SCHEDULER_TO, T_mult=SCHEDULER_T_MULT
    )
    early_stop = EarlyStopping()
    writer = SummaryWriter(log_dir=str(LOG_DIR / run_name))

    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):

        t0 = time.time()
        tr_loss, tr_lp, tr_ld, tr_m = _run_epoch(model, train_loader, criterion, optimizer, is_train=True)
        val_loss, val_lp, val_ld, val_m = _run_epoch(model, val_loader, criterion, None, is_train=False)

        scheduler.step()

        log.info(f"Epoch {epoch}/{NUM_EPOCHS} | "
                 f"Train Loss: {tr_loss:.6f} (Price: {tr_lp:.6f}, Dir: {tr_ld:.6f}) | "
                 f"Val Loss: {val_loss:.6f} (Price: {val_lp:.6f}, Dir: {val_ld:.6f}) | "
                 f"Time: {time.time() - t0:.2f}s")

        writer.add_scalars("Loss/total", {"train": tr_loss, "val": val_loss}, epoch)
        writer.add_scalars("Loss/price", {"train": tr_lp, "val": val_lp}, epoch)
        writer.add_scalars("Loss/direction", {"train": tr_ld, "val": val_ld}, epoch)
        writer.add_scalars("Metrics/MAE", {"train": tr_m["mae"], "val": val_m["mae"]}, epoch)
        writer.add_scalars("Metrics/RMSE", {"train": tr_m["rmse"], "val": val_m["rmse"]}, epoch)
        writer.add_scalars("Metrics/Direction Accuracy", {"train": tr_m["dir_acc"], "val": val_m["dir_acc"]}, epoch)
        writer.add_scalars("Metrics/F1 Score", {"train": tr_m["f1"], "val": val_m["f1"]}, epoch)
        writer.add_scalars("Learning Rate", {"lr": optimizer.param_groups[0]["lr"]}, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(model, optimizer, scheduler, epoch, val_loss, BEST_CKPT)
            log.info(f"New best model saved at epoch {epoch} with val_loss={val_loss:.6f}")
        _save_checkpoint(model, optimizer, scheduler, epoch, val_loss, LAST_CKPT)

        if early_stop.step(val_loss):
            log.info(f"Early stopping triggered at epoch {epoch}")
            break

    writer.close()

    load_checkpoint(model, path=BEST_CKPT)
    log.info("Training completed. Best weights loaded")

    return model
