import logging
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score

from .lstm_config import (
    DEVICE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, LAMBDA_ATTN, GRAD_CLIP, PATIENCE,
    MIN_DELTA,BEST_CKPT, RESUME_CKPT, LOG_DIR, USE_AMP
)

log = logging.getLogger(__name__)

class MultiTaskLoss(nn.Module):
    def __init__(self, lambda_attn=LAMBDA_ATTN):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.1]).to(DEVICE))
        self.la = lambda_attn

    def forward(self, dir_pred, dir_true, attn_weights) -> tuple[torch.Tensor, torch.Tensor]:

        loss_dir = self.bce(dir_pred.squeeze(-1), dir_true)

        eps = 1e-8
        entropy =  -(attn_weights * torch.log(attn_weights + eps)).sum(dim=1).mean()
        l_attn = -entropy

        total = loss_dir + self.la * l_attn
        return total, loss_dir

def _compute_metrics(dir_preds: np.ndarray, dir_trues: np.ndarray) -> dict:

    dir_binary = (dir_preds >= 0.5).astype(int)
    dir_acc = (dir_binary == dir_trues.astype(int)).mean()
    f1 = f1_score(dir_trues.astype(int), dir_binary, zero_division=0)

    return {
        "dir_acc": dir_acc,
        "f1": f1,
    }

def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: MultiTaskLoss,
    optimizer: torch.optim.Optimizer | None,
    scaler_amp,
    is_train: bool,
) -> tuple[float, float, dict]:

    model.train() if is_train else model.eval()

    total_loss = dir_loss_sum = 0.0
    all_dir_pred, all_dir_true = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        n_batches = 0
        for x_num, x_emb, y_dir, _ in loader:
            x_num = x_num.to(DEVICE, non_blocking=True)
            x_emb = x_emb.to(DEVICE, non_blocking=True)
            y_dir = y_dir.to(DEVICE, non_blocking=True)

            try:
                autocast_device = "cuda" if DEVICE == "cuda" else "cpu"
                with torch.autocast(device_type=autocast_device, enabled=(USE_AMP and DEVICE=="cuda")):
                    dir_pred, attn_weights = model(x_num, x_emb)
                    loss, ld = criterion(dir_pred, y_dir, attn_weights)

                if is_train:
                    assert optimizer is not None, "optimizer must be provided when is_train=True"
                    optimizer.zero_grad(set_to_none=True)

                    if USE_AMP and DEVICE == "cuda":
                        scaler_amp.scale(loss).backward()
                        scaler_amp.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                        scaler_amp.step(optimizer)
                        scaler_amp.update()
                    else:
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                        optimizer.step()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    log.error(f"OOM at batch — skipping: {e}")
                    if is_train and optimizer is not None:
                        optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    continue
                raise

            total_loss += loss.item()
            dir_loss_sum += ld.item()
            n_batches += 1

            dir_pred = torch.sigmoid(dir_pred)

            all_dir_pred.append(dir_pred.squeeze(-1).detach().cpu().numpy())
            all_dir_true.append(y_dir.detach().cpu().numpy())

    n = max(n_batches, 1)

    if not all_dir_pred:
        log.warning("No valid batches were processed in this epoch.")

    metrics = _compute_metrics(
        np.concatenate(all_dir_pred),
        np.concatenate(all_dir_true),
    )

    return total_loss / n, dir_loss_sum / n, metrics


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


def _save_checkpoint(model, optimizer,scheduler, scaler_amp, epoch, val_loss, path, best_val_loss=None, early_stop_counter=0):

    tmp_path = Path(str(path) + ".tmp")

    torch.save({
        "epoch": epoch,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "early_stop_counter": early_stop_counter,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler_amp": scaler_amp.state_dict() if (USE_AMP and DEVICE == "cuda") else None,
    }, tmp_path)
    tmp_path.replace(path)
    log.info(f"Checkpoint saved at {Path(path).name} (val_loss={val_loss:.6f})")

def load_checkpoint(model, optimizer=None, scheduler=None, path=BEST_CKPT):

    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    if optimizer:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler:
        scheduler.load_state_dict(ckpt["scheduler"])

    val_loss = ckpt.get('val_loss')
    log.info(f"Loaded checkpoint from epoch {ckpt['epoch']} val_loss={val_loss if val_loss is None else f'{val_loss:.6f}'}")

    return ckpt["epoch"], ckpt["val_loss"]

def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, run_name: str="run") -> nn.Module:

    model.to(DEVICE)
    log.info(f"Training on {DEVICE}")
    log.info(f"AMP : {USE_AMP}")
    log.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = MultiTaskLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scaler_amp = torch.amp.grad_scaler.GradScaler(device=DEVICE, enabled=(USE_AMP and DEVICE == "cuda"))
    early_stop = EarlyStopping()
    writer = SummaryWriter(log_dir=str(LOG_DIR / run_name))
    best_val_loss = float("inf")
    start_epoch = 1

    if RESUME_CKPT.exists():
        log.info(f"Resuming training from checkpoint {RESUME_CKPT.name}")
        ckpt = torch.load(RESUME_CKPT, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])

        if ckpt.get("scaler_amp") and USE_AMP and DEVICE == "cuda":
            scaler_amp.load_state_dict(ckpt["scaler_amp"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        early_stop.best_loss = best_val_loss
        early_stop.counter = ckpt.get("early_stop_counter", 0)

    for epoch in range(start_epoch, NUM_EPOCHS + 1):

        t0 = time.time()
        try:
         tr_loss,tr_ld, tr_m = _run_epoch(model, train_loader, criterion, optimizer, scaler_amp, is_train=True)
        except RuntimeError as e:
            log.error(f"Training crashed: {e}")
            torch.cuda.empty_cache()
            raise
        val_loss,val_ld, val_m = _run_epoch(model, val_loader, criterion, None, scaler_amp, is_train=False)

        scheduler.step(val_loss)

        elapsed = time.time() - t0

        if DEVICE == "cuda":
            dev_idx = torch.cuda.current_device()
            mem_used = torch.cuda.memory_reserved(dev_idx) / 1e9
            mem_total = torch.cuda.get_device_properties(dev_idx).total_memory / 1e9
            gpu_str = f" GPU Memory: {mem_used:.2f}GB / {mem_total:.2f}GB"
        else:
            gpu_str = ""


        log.info(f"Epoch {epoch}/{NUM_EPOCHS} |"
                 f" LR: {optimizer.param_groups[0]['lr']:.8f} | "
                 f"Time: {elapsed:.2f}s {gpu_str} | "
                 f"Gap: {val_loss - tr_loss:.6f} | "
                 f"Best Val Loss: {best_val_loss:.6f} | "
                 f"Dir Acc: {val_m['dir_acc']:.6f}, F1: {val_m['f1']:.6f}")

        writer.add_scalars("Loss/total", {"train": tr_loss, "val": val_loss}, epoch)
        writer.add_scalars("Loss/direction", {"train": tr_ld, "val": val_ld}, epoch)
        writer.add_scalars("Metrics/Direction Accuracy", {"train": tr_m["dir_acc"], "val": val_m["dir_acc"]}, epoch)
        writer.add_scalars("Metrics/F1 Score", {"train": tr_m["f1"], "val": val_m["f1"]}, epoch)
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], epoch)

        if DEVICE == "cuda":
            writer.add_scalar("GPU Memory (GB)", torch.cuda.memory_reserved((torch.cuda.current_device())) / 1e9, epoch)

        should_stop = early_stop.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(model, optimizer,scheduler, scaler_amp, epoch, val_loss, BEST_CKPT, best_val_loss, early_stop.counter)
            log.info(f"New best model saved at epoch {epoch} with val_loss={val_loss:.6f}")
        _save_checkpoint(model, optimizer,scheduler, scaler_amp, epoch, val_loss, RESUME_CKPT, best_val_loss=best_val_loss, early_stop_counter=early_stop.counter)

        if should_stop:
            log.info(f"Early stopping triggered at epoch {epoch}")
            break

        gap = tr_loss - val_loss
        if epoch > 10:
            if val_loss > best_val_loss * 1.05:
                log.warning(f"Val loss {val_loss:.5f} is 5% above best val loss {best_val_loss:.5f} - Possible overfitting.")
            if gap < -0.1:
                log.warning(f"Large gap between train and val loss: {gap:.5f} - model memorizing training data.")

    writer.close()

    load_checkpoint(model, path=BEST_CKPT)
    log.info("Training completed. Best weights loaded")

    return model
