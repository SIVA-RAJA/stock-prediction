import logging
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score

from lstm_config import (
    DEVICE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, LAMBDA_PRICE, LAMBDA_DIR, GRAD_CLIP, PATIENCE, MIN_DELTA, SCHEDULER_TO, SCHEDULER_T_MULT, BEST_CKPT, LAST_CKPT, LOG_DIR,
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


