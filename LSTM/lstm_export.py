import torch
import logging
from lstm_config import CHECKPOINT_DIR, SEQ_LEN

log = logging.getLogger(__name__)
ONNX_PATH = CHECKPOINT_DIR / "market_lstm.onnx"

def export_onnx(model, num_features: int):

    model.eval()
    model.to("cpu")

    dummy_x_num = torch.randn(1, SEQ_LEN, num_features)
    dummy_x_emb = torch.zeros(1, 3, dtype=torch.long)

    torch.onnx.export(
        model,
        (dummy_x_num, dummy_x_emb),
        str(ONNX_PATH),
        opset_version=17,
        input_names=["x_num", "x_emb"],
        output_names=["price_pred", "dir_pred", "attn_weights"],
        dynamic_axes={
            "x_num": {0: "batch_size"},
            "x_emb": {0: "batch_size"},
            "price_pred": {0: "batch_size"},
            "dir_pred": {0: "batch_size"},
            "attn_weights": {0: "batch_size"},
        },
        do_constant_folding= True,
    )

    log.info(f"Model exported to ONNX -> {ONNX_PATH}")

def verify_onnx(num_features: int):

    import onnxruntime as ort
    import numpy as np

    sess = ort.InferenceSession(str(ONNX_PATH))
    dummy_num = np.random.randn(2, SEQ_LEN, num_features).astype(np.float32)
    dummy_emb = np.zeros((2,3), dtype=np.int64)

    outputs = sess.run(None, {"x_num": dummy_num, "x_emb": dummy_emb})
    log.info(f"ONNX verified - output shapes: {[np.asanyarray(o).shape for o in outputs]}")
    return outputs
