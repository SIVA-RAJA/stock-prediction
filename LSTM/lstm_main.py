import argparse
import logging
from multiprocessing.spawn import import_main_path
import torch

from data.config import TICKER_TO_ID, MARKET_TO_ID, REGION_TO_ID
from lstm_config import DEVICE, BEST_CKPT
from lstm_dataset import make_dataloaders
from lstm_model import MarketLSTM
from lstm_trainer import train, load_checkpoint
from lstm_evaluate import evaluate
from lstm_export import export_onnx, verify_onnx

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger(__name__)

def build_model(num_features: int) -> MarketLSTM:
    return MarketLSTM(
        num_features=num_features,
        num_tickers=len(TICKER_TO_ID),
        num_markets=len(MARKET_TO_ID),
        num_regions=len(REGION_TO_ID),
    )
    
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", type=str, default=None)
    parser.add_argument("--inteval", type=str, default="1d")
    parser.add_argument("--region", type=str, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()
    
    run_name = f"{args.market or 'ALL'}_{args.interval}_{args.region or 'ALL'}"
    
    log.info(f"Loading data for: {run_name}")
    train_loader, val_loader, test_loader, num_features = make_dataloaders(
        market=args.market,
        interval=args.interval,
        region=args.region,
    )
    model = build_model(num_features)
    log.info(f"Model built: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    if args.eval_only:
        if not BEST_CKPT.exists():
            raise FileNotFoundError(f"No checkpoint found at {BEST_CKPT}")
        load_checkpoint(model, path=BEST_CKPT)
        log.info("Loaded best checkpoint for evaluation")
    else:
        log.info("Starting training...")
        model = train(model, train_loader, val_loader, run_name=run_name)
    
    log.info("Evaluating on testm set...")
    metrices = evaluate(
        model, test_loader,
        ticker=args.market or 'ALL',
        interval=args.interval,
        run_name=run_name,
    )
    
    if args.export:
        log.info("Exporting to INNX...")
        export_onnx(model, num_features)
        verify_onnx(num_features)
        
    return model, metrices

if __name__ == "__main__":
    main()
