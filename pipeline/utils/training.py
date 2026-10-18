"""
Model-agnostic training, prediction, scaling and checkpointing for the JAX models.

Each epoch runs as one jitted ``lax.scan`` over shuffled fixed-size minibatches,
so a training run compiles exactly once.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from importlib.metadata import version
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from models import build_model

from .windows import Windows


@dataclass
class Scaler:
    """Per-feature standardisation, fit on training rows only."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values):
        mean, std = values.mean(axis=0), values.std(axis=0)
        std = np.where(std == 0, 1.0, std)
        return cls(mean.astype(np.float32), std.astype(np.float32))

    def transform(self, values):
        return ((values - self.mean) / self.std).astype(np.float32)

    def inverse(self, scaled, col=None):
        if col is None:
            return scaled * self.std + self.mean
        return scaled * self.std[col] + self.mean[col]

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(np.asarray(d["mean"], np.float32), np.asarray(d["std"], np.float32))


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_frac: float = 0.05
    clip_norm: float = 1.0
    patience: int = 10
    min_delta: float = 0.0
    # loss = target_weight * MSE(target column) + (1 - target_weight) * MSE(other columns)
    target_weight: float = 0.5
    loss: str = "mse"  # or "huber"


@dataclass
class TrainResult:
    model: eqx.Module  # best-validation weights, in inference mode
    history: dict = field(default_factory=dict)
    best_epoch: int = 0
    best_val_target_mse: float | None = None
    train_time_s: float = 0.0
    n_params: int = 0
    stopped_early: bool = False


def count_params(model):
    return int(sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array))))


def _feature_weights(n_features, target_idx, target_weight):
    if n_features == 1:
        return jnp.ones((1,), jnp.float32)
    w = np.full(n_features, (1.0 - target_weight) / (n_features - 1), np.float32)
    w[target_idx] = target_weight
    return jnp.asarray(w)


def _batched_forward(model, inputs, key=None):
    if key is None:
        return jax.vmap(model)(*inputs)
    keys = jr.split(key, inputs[0].shape[0])
    return jax.vmap(lambda k, *xs: model(*xs, key=k))(keys, *inputs)


def train_model(model, train: Windows, val: Windows | None, cfg: TrainConfig, *, target_idx, key,
                epochs=None, log=print):
    """Minibatch AdamW with warmup-cosine LR, grad clipping and early stopping on validation target MSE.

    Without ``val`` it trains for the full ``epochs`` and returns the final weights.
    """
    epochs = epochs or cfg.epochs
    start = time.perf_counter()
    params, static = eqx.partition(model, eqx.is_inexact_array)
    feature_w = _feature_weights(train.target.shape[-1], target_idx, cfg.target_weight)

    n_train = train.size
    batch = min(cfg.batch_size, n_train)
    steps = n_train // batch
    total = epochs * steps
    schedule = optax.warmup_cosine_decay_schedule(
        0.0, cfg.lr, warmup_steps=max(1, int(cfg.warmup_frac * total)), decay_steps=max(2, total),
        end_value=cfg.lr * 0.01)
    decay_mask = jax.tree_util.tree_map(lambda p: p.ndim >= 2, params)  # no decay on biases / norms
    optimizer = optax.chain(optax.clip_by_global_norm(cfg.clip_norm),
                            optax.adamw(schedule, weight_decay=cfg.weight_decay, mask=decay_mask))
    opt_state = optimizer.init(params)

    def per_element(err):
        return optax.huber_loss(err) if cfg.loss == "huber" else jnp.square(err)

    def loss_fn(p, data, k):
        pred = _batched_forward(eqx.combine(p, static), data[:4], k)
        return jnp.mean(per_element(pred - data[4]) @ feature_w)

    @eqx.filter_jit
    def run_epoch(p, state, data, k):
        perm_key, step_key = jr.split(k)
        perm = jr.permutation(perm_key, n_train)[:steps * batch].reshape(steps, batch)

        def step(carry, xs):
            p, state = carry
            idx, sk = xs
            loss, grads = jax.value_and_grad(loss_fn)(p, tuple(a[idx] for a in data), sk)
            updates, state = optimizer.update(grads, state, p)
            return (eqx.apply_updates(p, updates), state), loss

        (p, state), losses = jax.lax.scan(step, (p, state), (perm, jr.split(step_key, steps)))
        return p, state, losses.mean()

    @eqx.filter_jit
    def evaluate(p, data):
        pred = _batched_forward(eqx.nn.inference_mode(eqx.combine(p, static)), data[:4])
        err = pred - data[4]
        return jnp.mean(per_element(err) @ feature_w), jnp.mean(jnp.square(err[..., target_idx]))

    train_data = tuple(jnp.asarray(a) for a in train)
    val_data = None if val is None else tuple(jnp.asarray(a) for a in val)

    history = {"train_loss": [], "val_loss": [], "val_target_mse": [], "lr": []}
    best_params, best_epoch, best_score, bad_epochs, stopped_early = params, epochs, np.inf, 0, False
    for epoch in range(1, epochs + 1):
        key, epoch_key = jr.split(key)
        params, opt_state, train_loss = run_epoch(params, opt_state, train_data, epoch_key)
        history["train_loss"].append(float(train_loss))
        history["lr"].append(float(schedule(epoch * steps)))
        if not np.isfinite(history["train_loss"][-1]):
            log(f"  epoch {epoch}: non-finite loss, stopping")
            stopped_early = True
            break
        if val_data is None:
            best_params = params
            continue

        val_loss, val_target = (float(x) for x in evaluate(params, val_data))
        history["val_loss"].append(val_loss)
        history["val_target_mse"].append(val_target)
        if val_target < best_score - cfg.min_delta:
            best_params, best_epoch, best_score, bad_epochs = params, epoch, val_target, 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                stopped_early = True
                break

    model = eqx.nn.inference_mode(eqx.combine(best_params, static))
    return TrainResult(
        model=model,
        history=history,
        best_epoch=best_epoch if val_data is not None else len(history["train_loss"]),
        best_val_target_mse=None if val_data is None else float(best_score),
        train_time_s=time.perf_counter() - start,
        n_params=count_params(model),
        stopped_early=stopped_early,
    )


@eqx.filter_jit
def _predict_chunk(model, inputs):
    return _batched_forward(model, inputs)


def predict(model, windows: Windows, batch_size=256):
    """Inference-mode predictions in scaled units, [N, pred_len, C].

    Inputs are processed in fixed-size chunks (the last one padded) so each
    model compiles once per chunk size.
    """
    model = eqx.nn.inference_mode(model)
    n = windows.size
    chunk = min(batch_size, n)
    out = []
    for s in range(0, n, chunk):
        part = [a[s:s + chunk] for a in windows.inputs()]
        real = len(part[0])
        if real < chunk:
            part = [np.concatenate([a, np.repeat(a[-1:], chunk - real, axis=0)]) for a in part]
        out.append(np.asarray(_predict_chunk(model, tuple(jnp.asarray(a) for a in part)))[:real])
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def save_checkpoint(dirpath, name, model, build_kwargs, scaler: Scaler, extra=None):
    """Write ``model.eqx`` (weights) and ``meta.json`` (everything needed to rebuild)."""
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(dirpath / "model.eqx", eqx.nn.inference_mode(model))
    meta = {
        "model": name,
        "build_kwargs": build_kwargs,
        "scaler": scaler.to_dict(),
        "versions": {pkg: version(pkg) for pkg in ("jax", "equinox", "optax")},
        **(extra or {}),
    }
    if name == "fedformer":
        from models.fedformer import FEDFORMER_DEFAULTS, frequency_modes

        # For reference only: mode_seed in build_kwargs rebuilds the same modes.
        modes = frequency_modes({**FEDFORMER_DEFAULTS, **build_kwargs})
        meta["frequency_modes"] = {k: list(v) for k, v in modes.items()}
    (dirpath / "meta.json").write_text(json.dumps(meta, indent=2, default=_json_default))
    return dirpath


def load_checkpoint(dirpath):
    """Returns ``(model, scaler, meta)``; the model is in inference mode."""
    dirpath = Path(dirpath)
    meta = json.loads((dirpath / "meta.json").read_text())
    skeleton = eqx.nn.inference_mode(build_model(meta["model"], key=jr.PRNGKey(0), **meta["build_kwargs"]))
    model = eqx.tree_deserialise_leaves(dirpath / "model.eqx", skeleton)
    return model, Scaler.from_dict(meta["scaler"]), meta


def _json_default(o):
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "__dataclass_fields__"):
        return asdict(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")
