import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import models  # noqa: E402,F401  (configures JAX before any test touches it)

# Tiny dims so the whole suite runs in well under a minute.
TINY = dict(n_features=3, seq_len=32, label_len=16, pred_len=4)
TINY_OVERRIDES = {
    "standard": dict(d_model=16, n_heads=2, d_ff=32, dropout=0.0),
    "pyraformer": dict(d_model=16, d_inner_hid=16, d_bottleneck=8, d_k=8, d_v=8, n_head=2, dropout=0.0,
                       window_size=[2, 2, 2]),
}


def tiny_kwargs(name, **extra):
    """Builder kwargs at TINY dims; ``extra`` may override dims (e.g. label_len) or model settings."""
    dims = {k: extra.pop(k, v) for k, v in TINY.items()}
    group = "pyraformer" if name == "pyraformer" else "standard"
    overrides = {**TINY_OVERRIDES[group], **extra}
    if name == "fedformer":
        overrides.setdefault("modes", 4)
    return models.model_kwargs(name, dims["n_features"], dims["seq_len"], dims["label_len"], dims["pred_len"],
                               **overrides)
