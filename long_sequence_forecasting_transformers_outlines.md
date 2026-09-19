# Four Long-Sequence Forecasting Transformers — Implementation Outlines

**Informer · Autoformer · FEDformer · Pyraformer**

A single reference document containing four self-contained, code-level implementation
outlines. Each outline is written for a downstream coding model that will implement the
architecture in PyTorch **without reading the original paper**: every equation, tensor
shape, default hyperparameter, and module boundary is spelled out explicitly.

Anything that could not be confirmed against a primary source (the paper PDF or the
authors' released code) is tagged **[UNVERIFIED]** inline.

---

## How to use this document

1. Pick one model. Read its `Notation & Tensor Shapes` section first — every later shape
   comment depends on those symbols.
2. Implement bottom-up in the order given by that model's `Implementation Checklist`.
   Each step is independently testable.
3. Before wiring the full model, run the smoke test at the end of the checklist
   (random tensors in, assert output shape).
4. Read that model's `Common Pitfalls` section **before** debugging anything. Most
   failure modes in these architectures are silent (wrong axis, wrong mask polarity,
   re-sampled indices) rather than loud.

All four models share the same experimental scaffolding — ETT-style CSV datasets, a
`StandardScaler` fit on the training split only, calendar/time-feature covariates, an
encoder window of `L_enc` steps, a `label_len` "start token" window, MSE loss on
standardized data, Adam at `1e-4`, and MSE/MAE reported on the standardized scale. If you
implement the data pipeline once, it serves all four.

---

## The four papers

| Model | Venue | Core idea | Complexity | Primary source |
|---|---|---|---|---|
| **Informer** | AAAI 2021 | ProbSparse attention (top-`u` "active" queries only) + self-attention distilling + generative one-pass decoder | `O(L log L)` | [arXiv:2012.07436](https://arxiv.org/abs/2012.07436) · [code](https://github.com/zhouhaoyi/Informer2020) |
| **Autoformer** | NeurIPS 2021 | Progressive seasonal-trend decomposition inside every layer + Auto-Correlation (FFT-based lag discovery, sub-series aggregation) replacing self-attention | `O(L log L)` | [arXiv:2106.13008](https://arxiv.org/abs/2106.13008) · [code](https://github.com/thuml/Autoformer) |
| **FEDformer** | ICML 2022 | Attention performed entirely in the frequency domain on a small **random** subset of modes, plus Autoformer-style decomposition; Fourier and Wavelet variants | `O(L)` | [arXiv:2201.12740](https://arxiv.org/abs/2201.12740) · [code](https://github.com/MAZiqing/FEDformer) |
| **Pyraformer** | ICLR 2022 (oral) | Pyramidal attention over a C-ary multi-resolution tree; each node attends only to neighbours, children and parent | `O(L)`, max signal path `O(1)` | [OpenReview](https://openreview.net/forum?id=0EXmFzUn5I) · [code](https://github.com/ant-research/Pyraformer) |

---

## Shared architectural vocabulary

These symbols mean the same thing in all four outlines:

| Symbol | Meaning |
|---|---|
| `B` | batch size |
| `L_enc` / `seq_len` | encoder input (history) length |
| `L_label` / `label_len` | "start token" window, taken from the **tail of the encoder window** |
| `L_pred` / `pred_len` | forecast horizon |
| `L_dec` | decoder sequence length, `= L_label + L_pred` |
| `d_model` | model width (512 in all four) |
| `H` / `n_heads` | attention heads |
| `E` | per-head width, usually `d_model // n_heads` |
| `d_ff` | FFN inner width (2048 in all four) |
| `enc_in` / `dec_in` / `c_out` | input / decoder-input / output channel counts |
| `d_mark` / `d_time` | number of calendar time-feature columns per step |

---

## What differs between them — a decision table

| Aspect | Informer | Autoformer | FEDformer | Pyraformer |
|---|---|---|---|---|
| Attention replacement | ProbSparse (sampled top-`u` queries) | Auto-Correlation (FFT cross-correlation + time-delay aggregation) | Frequency-domain complex linear map on selected modes | Masked full attention over a pyramidal graph |
| Positional encoding | **Yes** (sinusoidal) | **No** | **No** | **Yes** (sinusoidal) |
| Series decomposition | No | Yes, in every layer | Yes, in every layer (multi-kernel) | No |
| Sequence-length reduction | Distilling (Conv+MaxPool, halves `L`) | None | None | CSCM builds coarser levels (concatenated, not replaced) |
| Decoder | Generative, one forward pass | Non-autoregressive, seasonal + accumulated trend | Non-autoregressive, seasonal + accumulated trend | FC head on gathered ancestors, or cross-attention decoder |
| Causal masking | Yes (decoder self-attention) | No (flags exist but are inert) | No | Only in the optional attention decoder |
| Key gotcha | `.clone()` the expanded mean context; align `ProbMask` rows with selected queries | FFT must run on the time axis after `permute(0,2,3,1)`; train vs eval aggregation differ | Sample the random modes **once** in `__init__`; `irfft` needs `n=` | Mask polarity and the `+1` predict token |

---

## Reading order recommendation

Informer → Autoformer → FEDformer → Pyraformer.

Autoformer's encoder/decoder skeleton, `series_decomp`, `my_Layernorm` and
`DataEmbedding_wo_pos` are reused almost verbatim by FEDformer, so reading those two in
order saves significant effort — FEDformer only swaps the attention block. Informer
contributes the data pipeline, embedding stack and decoder-input convention that all the
others inherit. Pyraformer is the most independent of the four.

---

## Table of contents

- [How to use this document](#how-to-use-this-document)
- [The four papers](#the-four-papers)
- [Shared architectural vocabulary](#shared-architectural-vocabulary)
- [What differs between them — a decision table](#what-differs-between-them-a-decision-table)
- [Reading order recommendation](#reading-order-recommendation)
- [1. Informer](#1-informer)
  - [Notation & Tensor Shapes](#notation-tensor-shapes)
  - [Data Pipeline](#data-pipeline)
  - [Module-by-Module Spec](#module-by-module-spec)
  - [Full Model Assembly](#full-model-assembly)
  - [Default Hyperparameters](#default-hyperparameters)
  - [Training Loop](#training-loop)
  - [Evaluation](#evaluation)
  - [Implementation Checklist](#implementation-checklist)
  - [Common Pitfalls](#common-pitfalls)
- [2. Autoformer](#2-autoformer)
  - [Notation & Tensor Shapes](#notation-tensor-shapes-1)
  - [Data Pipeline](#data-pipeline-1)
  - [Module-by-Module Spec](#module-by-module-spec-1)
  - [Full Model Assembly](#full-model-assembly-1)
  - [Default Hyperparameters](#default-hyperparameters-1)
  - [Training Loop](#training-loop-1)
  - [Evaluation](#evaluation-1)
  - [Implementation Checklist](#implementation-checklist-1)
  - [Common Pitfalls](#common-pitfalls-1)
- [3. FEDformer](#3-fedformer)
  - [Notation & Tensor Shapes](#notation-tensor-shapes-2)
  - [Data Pipeline](#data-pipeline-2)
  - [Module-by-Module Spec](#module-by-module-spec-2)
  - [Full Model Assembly](#full-model-assembly-2)
  - [Default Hyperparameters](#default-hyperparameters-2)
  - [Training Loop](#training-loop-2)
  - [Evaluation](#evaluation-2)
  - [Implementation Checklist](#implementation-checklist-2)
  - [Common Pitfalls](#common-pitfalls-2)
- [4. Pyraformer](#4-pyraformer)
  - [Notation & Tensor Shapes](#notation-tensor-shapes-3)
  - [Data Pipeline](#data-pipeline-3)
  - [The Pyramidal Graph — index construction](#the-pyramidal-graph-index-construction)
  - [Module-by-Module Spec](#module-by-module-spec-3)
  - [Full Model Assembly](#full-model-assembly-3)
  - [Complexity Analysis](#complexity-analysis)
  - [Default Hyperparameters](#default-hyperparameters-3)
  - [Training Loop](#training-loop-3)
  - [Evaluation](#evaluation-3)
  - [Implementation Checklist](#implementation-checklist-3)
  - [Common Pitfalls](#common-pitfalls-3)
  - [[UNVERIFIED] items](#unverified-items)

---

## 1. Informer

Informer is an encoder–decoder Transformer built specifically for **long sequence time-series forecasting (LSTF)**: given a long window of past multivariate observations, predict a long future window (24 to 720+ steps) in a single forward pass. A vanilla Transformer is impractical here because self-attention costs O(L²) time and memory and its decoder must run autoregressively, which is slow and accumulates error. Informer fixes all three problems with (1) **ProbSparse self-attention**, which only computes full attention rows for the `u = c·ln(L_Q)` "most informative" queries (measured by how far their attention distribution is from uniform) and fills the remaining rows with a cheap aggregate of V, giving O(L·ln L) cost; (2) **self-attention distilling**, a Conv1d+MaxPool block between encoder layers that halves the sequence length each layer, cutting stacked-layer memory; and (3) a **generative decoder** that is fed `[known label window ; zeros of length L_pred]` plus the future timestamps and emits the whole prediction in one shot (no autoregression). The model is trained with MSE on z-score-standardized data.

---

### Notation & Tensor Shapes

| Symbol | Meaning | Typical value |
|---|---|---|
| `B` | batch size | 32 |
| `L_enc` (`seq_len`) | encoder input length | 96 / 336 / 720 |
| `L_label` (`label_len`) | decoder "start token" length; must satisfy `L_label <= L_enc` | 48 / 168 |
| `L_pred` (`pred_len`) | prediction horizon | 24 / 48 / 168 / 336 / 720 |
| `L_dec` | decoder input length = `L_label + L_pred` | — |
| `enc_in` | # input channels to encoder | 7 (ETT multivariate), 1 (univariate) |
| `dec_in` | # input channels to decoder | same as `enc_in` |
| `c_out` | # output channels | 7 (M), 1 (S / MS) |
| `d_model` | model width | 512 |
| `n_heads` (`H`) | attention heads | 8 |
| `d_head` (`E`,`D`) | `d_model // n_heads` | 64 |
| `d_ff` | FFN inner dim | 2048 |
| `e_layers` | encoder attention layers | 2 (repo default), 3 (paper) |
| `d_layers` | decoder layers | 1 (repo default), 2 (paper) |
| `c` (`factor`) | ProbSparse sampling factor | 5 |
| `d_time` | # time-feature columns per timestep | 4 for freq='h', 5 for freq='t' |
| `L_Q, L_K, L_V` | query / key / value lengths inside attention | `L_K == L_V` always |
| `U_part` | # keys sampled per query = `min(c·⌈ln L_K⌉, L_K)` | — |
| `u` | # queries kept = `min(c·⌈ln L_Q⌉, L_Q)` | — |

#### Shapes of every model-level tensor

```
x_enc        : (B, L_enc, enc_in)          # standardized past values
x_mark_enc   : (B, L_enc, d_time)          # past timestamps encoded
x_dec        : (B, L_dec, dec_in)          # [label window ; zeros]
x_mark_dec   : (B, L_dec, d_time)          # label+future timestamps
-> output    : (B, L_pred, c_out)          # prediction (standardized space)
target       : (B, L_pred, c_out)
```

Internal:
```
enc_embed_out : (B, L_enc, d_model)
enc_out       : (B, L_enc // 2**(e_layers-1), d_model)   # with distilling
dec_embed_out : (B, L_dec, d_model)
dec_out       : (B, L_dec, d_model) -> project -> (B, L_dec, c_out) -> slice last L_pred
```

Distilling length math (`e_layers=2`, `L_enc=96`): one ConvLayer between the 2 attention layers → `96 -> 48`. With `e_layers=3`, `96 -> 48 -> 24`. Exactly `e_layers - 1` ConvLayers.

---

### Data Pipeline

#### Dataset format (ETT CSV)

A CSV with a `date` column plus 7 numeric columns; the last, `OT` (oil temperature), is the target.

```
date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT
2016-07-01 00:00:00,5.827,2.009,1.599,0.462,4.203,1.340,30.531
...
```
- `ETTh1.csv`, `ETTh2.csv`: hourly, `freq='h'`.
- `ETTm1.csv`, `ETTm2.csv`: 15-minute, `freq='t'`.

#### Feature mode (`features`)
| mode | df columns used | enc_in | dec_in | c_out |
|---|---|---|---|---|
| `M` multivariate→multivariate | all 7 | 7 | 7 | 7 |
| `S` univariate→univariate | `[OT]` only | 1 | 1 | 1 |
| `MS` multivariate→univariate | all 7 | 7 | 7 | 1 |

For `MS`, the target slice from `batch_y` uses `f_dim = -1` (last channel); for `M`/`S`, `f_dim = 0`.

#### Split convention (ETT, fixed index boundaries — 30-day "months")

Hourly (`ETTh*`), with `n = 24` points/day:
```python
border1s = [0,            12*30*24 - seq_len,               12*30*24 + 4*30*24 - seq_len]
border2s = [12*30*24,     12*30*24 + 4*30*24,               12*30*24 + 8*30*24]
# train = rows [0, 8640), val = [8640-seq_len, 11520), test = [11520-seq_len, 14400)
```
Minute-level (`ETTm*`): multiply every term by 4 (`12*30*24*4`, etc.).

Note the `- seq_len` on the *start* of val/test: this gives each split enough left context to form its first window; it does **not** leak, because the loss/metrics only ever use the future part of a window.

Generic custom CSV split: `num_train = int(0.7*N)`, `num_test = int(0.2*N)`, `num_vali = N - num_train - num_test`, with the same `- seq_len` offsets.

#### Standardization (fit on TRAIN ONLY)

```python
class StandardScaler:
    def fit(self, data):      # data: (N, C) numpy from the TRAIN slice only
        self.mean = data.mean(0)   # (C,)
        self.std  = data.std(0)    # (C,)
    def transform(self, data):     return (data - self.mean) / self.std
    def inverse_transform(self, d):return d * self.std + self.mean
```
Fit on `df_data[border1s[0]:border2s[0]]` (the train rows), then transform the **entire** series. MSE/MAE are reported on this standardized scale.

#### Time-feature encoding

Two options; pick one via `embed`.

**A. `embed='timeF'` (repo default).** Continuous features, each normalized to `[-0.5, 0.5]`:
```python
MinuteOfHour = index.minute / 59.0 - 0.5
HourOfDay    = index.hour   / 23.0 - 0.5
DayOfWeek    = index.dayofweek / 6.0 - 0.5
DayOfMonth   = (index.day - 1)      / 30.0 - 0.5
DayOfYear    = (index.dayofyear - 1)/ 365.0 - 0.5
MonthOfYear  = (index.month - 1)    / 11.0 - 0.5
WeekOfYear   = (index.week - 1)     / 52.0 - 0.5
```
Column sets by frequency:
| freq | columns | `d_time` |
|---|---|---|
| `'h'` hourly | HourOfDay, DayOfWeek, DayOfMonth, DayOfYear | 4 |
| `'t'` minutely | MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear | 5 |
| `'d'` daily | DayOfWeek, DayOfMonth, DayOfYear | 3 |
| `'m'` monthly | MonthOfYear | 1 |
| `'w'` weekly | DayOfMonth, WeekOfYear | 2 |
| `'s'` secondly | SecondOfMinute + the 5 above | 6 |

**B. `embed='fixed'` or `'learned'`.** Integer categorical fields, consumed by `TemporalEmbedding`:
```python
dates['month']   = date.month        # 1..12
dates['day']     = date.day          # 1..31
dates['weekday'] = date.weekday()    # 0..6
dates['hour']    = date.hour         # 0..23
dates['minute']  = date.minute // 15 # 0..3   (only for freq='t')
freq_map = {'m':['month'], 'w':['month'],
            'd':['month','day','weekday'], 'b':['month','day','weekday'],
            'h':['month','day','weekday','hour'],
            't':['month','day','weekday','hour','minute']}
```
Column **order matters**: `[month, day, weekday, hour, (minute)]`, indexed as `x_mark[:,:,0..4]`.

#### Window construction (`__getitem__`)

```python
def __len__(self):
    return len(self.data_x) - self.seq_len - self.pred_len + 1

def __getitem__(self, index):
    s_begin = index
    s_end   = s_begin + self.seq_len                     # encoder window end
    r_begin = s_end - self.label_len                     # decoder window start (overlaps encoder tail)
    r_end   = r_begin + self.label_len + self.pred_len

    seq_x      = self.data_x[s_begin:s_end]              # (L_enc, C)
    seq_y      = self.data_y[r_begin:r_end]              # (L_label + L_pred, C)
    seq_x_mark = self.data_stamp[s_begin:s_end]          # (L_enc, d_time)
    seq_y_mark = self.data_stamp[r_begin:r_end]          # (L_label + L_pred, d_time)
    return seq_x, seq_y, seq_x_mark, seq_y_mark
```

#### Batch → model inputs (the only place the zero placeholder is created)

```python
batch_x, batch_y, batch_x_mark, batch_y_mark   # (B,L_enc,C) (B,L_dec,C) (B,L_enc,dt) (B,L_dec,dt)

zeros   = torch.zeros(B, pred_len, batch_y.shape[-1])          # padding=0 (default); padding=1 -> ones
dec_inp = torch.cat([batch_y[:, :label_len, :], zeros], dim=1) # (B, L_dec, C)   <-- future values ZEROED

outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)   # (B, L_pred, c_out)
f_dim   = -1 if features == 'MS' else 0
true    = batch_y[:, -pred_len:, f_dim:]                        # (B, L_pred, c_out)
loss    = MSE(outputs, true)
```
DataLoader: `shuffle=True, drop_last=True` for train; `shuffle=False, drop_last=True` for val/test; `num_workers=0`. All tensors cast to `.float()`.

---

### Module-by-Module Spec

#### 1. `PositionalEmbedding`

```python
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000)
```
Precompute a non-trainable buffer `pe` of shape `(1, max_len, d_model)`:
```python
position = arange(0, max_len).float().unsqueeze(1)                 # (max_len,1)
div_term = (arange(0, d_model, 2).float() * -(log(10000.0)/d_model)).exp()   # (d_model/2,)
pe[:, 0::2] = sin(position * div_term)
pe[:, 1::2] = cos(position * div_term)
pe = pe.unsqueeze(0); self.register_buffer('pe', pe)
```
`forward(x)` where `x: (B, L, *)` returns `self.pe[:, :x.size(1)]` → `(1, L, d_model)` (broadcasts over batch). Register as buffer so it moves with `.to(device)` but gets no gradient.

#### 2. `TokenEmbedding`

```python
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        self.tokenConv = nn.Conv1d(c_in, d_model, kernel_size=3,
                                   padding=1, padding_mode='circular')
        # init every Conv1d: kaiming_normal_(w, mode='fan_in', nonlinearity='leaky_relu')
    def forward(self, x):            # x: (B, L, c_in)
        x = self.tokenConv(x.permute(0, 2, 1))   # (B, d_model, L)
        return x.transpose(1, 2)                 # (B, L, d_model)
```
`padding=1` with `kernel_size=3, stride=1` keeps `L` unchanged. (On PyTorch < 1.5 the repo used `padding=2`; use `padding=1` on modern PyTorch.)

#### 3. `FixedEmbedding` and `TemporalEmbedding`

`FixedEmbedding(c_in, d_model)`: an `nn.Embedding(c_in, d_model)` whose weight is set to the sinusoidal table (same sin/cos formula as `PositionalEmbedding`, with `position = arange(0, c_in)`) and frozen (`requires_grad=False`); `forward` returns `self.emb(x).detach()`.

```python
class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        minute_size=4; hour_size=24; weekday_size=7; day_size=32; month_size=13
        Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
        if freq=='t': self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed    = Embed(hour_size,    d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed     = Embed(day_size,     d_model)
        self.month_embed   = Embed(month_size,   d_model)
    def forward(self, x):                      # x: (B, L, d_time) integer-valued
        x = x.long()
        minute_x  = self.minute_embed(x[:,:,4]) if hasattr(self,'minute_embed') else 0.
        hour_x    = self.hour_embed(x[:,:,3])
        weekday_x = self.weekday_embed(x[:,:,2])
        day_x     = self.day_embed(x[:,:,1])
        month_x   = self.month_embed(x[:,:,0])
        return hour_x + weekday_x + day_x + month_x + minute_x   # (B, L, d_model)
```
Vocab sizes are deliberately `+1` oversized (`day_size=32`, `month_size=13`) because `day`/`month` are 1-based.

#### 4. `TimeFeatureEmbedding`

```python
class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        freq_map = {'h':4,'t':5,'s':6,'m':1,'a':1,'w':2,'d':3,'b':3}
        self.embed = nn.Linear(freq_map[freq], d_model)
    def forward(self, x):    # (B, L, d_time) float
        return self.embed(x) # (B, L, d_model)
```

#### 5. `DataEmbedding` (sum of three, then dropout)

```python
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        self.value_embedding    = TokenEmbedding(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.temporal_embedding = (TimeFeatureEmbedding(d_model, embed_type, freq)
                                   if embed_type == 'timeF'
                                   else TemporalEmbedding(d_model, embed_type, freq))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, x_mark):            # (B,L,c_in), (B,L,d_time)
        x = (self.value_embedding(x)         # (B,L,d_model)
           + self.position_embedding(x)     # (1,L,d_model) broadcast
           + self.temporal_embedding(x_mark))# (B,L,d_model)
        return self.dropout(x)               # (B,L,d_model)
```
(The paper's balancing factor α is fixed to 1 because inputs are standardized.)

#### 6. `ProbAttention` (ProbSparse self-attention) — the core module

```python
class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None,
                 attention_dropout=0.1, output_attention=False)
```

**Sparsity measurement.** For query `q_i` the exact measure is
`M(q_i,K) = ln Σ_j exp(q_i k_jᵀ/√d) − (1/L_K) Σ_j q_i k_jᵀ/√d`,
approximated (Lemma 1 upper bound, dropping `ln L_K`) by the numerically stable **max-mean**
`M̄(q_i,K) = max_j (q_i k_jᵀ/√d) − (1/L_K) Σ_j (q_i k_jᵀ/√d)`.
Large `M̄` ⇒ that query's attention distribution is far from uniform ⇒ it is "active" and worth computing exactly.

**`forward(queries, keys, values, attn_mask)`**

Inputs `(B, L_Q, H, D)`, `(B, L_K, H, D)`, `(B, L_V, H, D)`; output `(B, L_Q, H, D)`.

```python
B, L_Q, H, D = queries.shape
_, L_K, _, _ = keys.shape

queries = queries.transpose(2,1)   # (B,H,L_Q,D)
keys    = keys.transpose(2,1)      # (B,H,L_K,D)
values  = values.transpose(2,1)    # (B,H,L_V,D)

U_part = factor * ceil(log(L_K))          # int; keys sampled per query
u      = factor * ceil(log(L_Q))          # int; queries kept
U_part = min(U_part, L_K)                 # clamp!
u      = min(u,      L_Q)                 # clamp!

scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)
scale = self.scale or 1./sqrt(D)
scores_top = scores_top * scale                      # (B,H,u,L_K)
context = self._get_initial_context(values, L_Q)     # (B,H,L_Q,D)
context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
return context.transpose(2,1).contiguous(), attn     # (B,L_Q,H,D)
```
Note `log` is the **natural log**, and `ceil` is applied before multiplying by `factor`.

**Step A — `_prob_QK(Q, K, sample_k, n_top)`**
```python
B, H, L_K, E = K.shape
_, _, L_Q, _ = Q.shape

# 1) expand keys so each query has its own sampling axis
K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)         # (B,H,L_Q,L_K,E)

# 2) sample_k random key indices per query (with replacement), shared across B and H
index_sample = torch.randint(L_K, (L_Q, sample_k))           # (L_Q, sample_k)
K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
                                                             # (B,H,L_Q,sample_k,E)

# 3) sampled scores: one (1 x E) query times (E x sample_k)
Q_K_sample = torch.matmul(Q.unsqueeze(-2),                   # (B,H,L_Q,1,E)
                          K_sample.transpose(-2,-1)          # (B,H,L_Q,E,sample_k)
                         ).squeeze(-2)                       # (B,H,L_Q,sample_k)

# 4) max-mean measurement per query. NOTE: divides by L_K, not sample_k (as in the repo)
M = Q_K_sample.max(-1)[0] - Q_K_sample.sum(-1) / L_K         # (B,H,L_Q)

# 5) indices of the top-u queries (order does not matter)
M_top = M.topk(n_top, sorted=False)[1]                       # (B,H,u)  int64

# 6) gather those queries and compute FULL attention scores for them only
Q_reduce = Q[torch.arange(B)[:,None,None],
             torch.arange(H)[None,:,None],
             M_top, :]                                       # (B,H,u,E)
Q_K = torch.matmul(Q_reduce, K.transpose(-2,-1))             # (B,H,u,L_K)
return Q_K, M_top
```
The three-index advanced-indexing pattern `[arange(B)[:,None,None], arange(H)[None,:,None], M_top, :]` broadcasts to `(B,H,u)` and is the canonical gather/scatter idiom used throughout this module — reuse it verbatim.

**Step B — `_get_initial_context(V, L_Q)`** (rows *not* selected get this value and are never updated)
```python
B, H, L_V, D = V.shape
if not self.mask_flag:                      # encoder self-attn / cross-attn
    V_sum  = V.mean(dim=-2)                            # (B,H,D)
    context = V_sum.unsqueeze(-2).expand(B,H,L_Q,D).clone()   # (B,H,L_Q,D)  .clone() is REQUIRED
else:                                       # causal decoder self-attn
    assert L_Q == L_V
    context = V.cumsum(dim=-2)                         # (B,H,L_Q,D)
return context
```
Rationale: an inactive query has a near-uniform attention distribution, so its output ≈ mean of the values it may attend to. Under causal masking, query `i` may only see `V[:i+1]`, so the uniform-average is the running mean — the repo stores the **cumulative sum** (an unnormalized stand-in, matching the paper's note "we can use sum(·) as the simpler implement of mean(·)"). Do not normalize it; match the reference.

**Step C — `_update_context(context_in, V, scores, index, L_Q, attn_mask)`**
```python
B, H, L_V, D = V.shape
if self.mask_flag:
    attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
    scores.masked_fill_(attn_mask.mask, -np.inf)        # scores: (B,H,u,L_K)

attn = torch.softmax(scores, dim=-1)                    # (B,H,u,L_K)
context_in[torch.arange(B)[:,None,None],
           torch.arange(H)[None,:,None],
           index, :] = torch.matmul(attn, V).type_as(context_in)   # scatter (B,H,u,D)
return context_in, attn_or_None
```
`output_attention=True` variant: build `attns = ones(B,H,L_V,L_V)/L_V` and scatter `attn` into the selected rows.

**`ProbMask` — the causal mask restricted to the selected query rows**
```python
class ProbMask:
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask   = torch.ones(L, scores.shape[-1], dtype=torch.bool, device=device).triu(1) # (L,L_K)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:,None,None],
                             torch.arange(H)[None,:,None],
                             index, :]                    # (B,H,u,L_K)
        self._mask = indicator.view(scores.shape)
    @property
    def mask(self): return self._mask
```
i.e. take the standard upper-triangular (strictly above diagonal) boolean mask and **select the same `u` rows** that were kept, so mask rows line up with `scores` rows.

**Complexity.** Sampled scores cost `O(L_Q · c·ln L_K)`; full rows cost `O(c·ln L_Q · L_K)`; with `L_Q = L_K = L` both are `O(L ln L)` in time and memory.

#### 7. `FullAttention` (canonical; used for decoder cross-attention, and for the Informer† ablation)

```python
class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None,
                 attention_dropout=0.1, output_attention=False)
    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape          # queries (B,L,H,E)
        _, S, _, D = values.shape           # keys/values (B,S,H,*)
        scale = self.scale or 1./sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)   # (B,H,L,S)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))   # (B,H,L,S)
        V = torch.einsum("bhls,bshd->blhd", A, values)            # (B,L,H,D)
        return V.contiguous(), (A if self.output_attention else None)
```
```python
class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        self._mask = torch.triu(torch.ones([B,1,L,L], dtype=torch.bool), diagonal=1).to(device)
    @property
    def mask(self): return self._mask
```
Note the asymmetry with `ProbAttention`: `FullAttention` applies dropout to the attention weights, `ProbAttention` does not (its `self.dropout` is constructed but unused). Keep this to match the reference.

#### 8. `AttentionLayer` (projections + head reshape)

```python
class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, mix=False):
        d_keys   = d_keys   or d_model // n_heads
        d_values = d_values or d_model // n_heads
        self.inner_attention  = attention
        self.query_projection = nn.Linear(d_model, d_keys   * n_heads)
        self.key_projection   = nn.Linear(d_model, d_keys   * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection   = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads; self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape          # (B,L,d_model)
        _, S, _ = keys.shape             # (B,S,d_model)
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)   # (B,L,H,d_keys)
        keys    = self.key_projection(keys).view(B, S, H, -1)        # (B,S,H,d_keys)
        values  = self.value_projection(values).view(B, S, H, -1)    # (B,S,H,d_values)
        out, attn = self.inner_attention(queries, keys, values, attn_mask)  # (B,L,H,d_values)
        if self.mix:
            out = out.transpose(2,1).contiguous()   # quirk: swaps L and H axes before flatten
        out = out.view(B, L, -1)                    # (B,L,d_model)
        return self.out_projection(out), attn       # (B,L,d_model)
```
`mix=True` is the default **only for the decoder's masked self-attention** (`--mix` default True); encoder layers and cross-attention use `mix=False`.

What `mix=True` actually does: it swaps the time and head axes (`(B,L,H,d_v) → (B,H,L,d_v)`) *before* the `view(B, L, -1)` flatten. The element count is unchanged (`H*L*d_v == L*H*d_v`), so **no shape error is ever raised**, but the `d_model` feature vector handed to `out_projection` is assembled from a different permutation of the underlying elements. This is a genuine quirk of the reference implementation, not a documented design choice. Reproduce it verbatim if you want to match published numbers; if you write the "clean" version (`mix=False` everywhere) expect small numeric differences. Requires `L == H` only if you want the two layouts to coincide — they otherwise just differ.

#### 9. `ConvLayer` (self-attention distilling)

Implements `X_{j+1} = MaxPool(ELU(Conv1d([X_j]_AB)))`:
```python
class ConvLayer(nn.Module):
    def __init__(self, c_in):                      # c_in == d_model
        self.downConv = nn.Conv1d(c_in, c_in, kernel_size=3, padding=1,
                                  padding_mode='circular')
        self.norm       = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool    = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
    def forward(self, x):                          # x: (B, L, d_model)
        x = self.downConv(x.permute(0, 2, 1))      # (B, d_model, L)
        x = self.norm(x)                           # BatchNorm1d over channel dim
        x = self.activation(x)
        x = self.maxPool(x)                        # (B, d_model, floor((L+2*1-3)/2)+1) = ceil(L/2)
        return x.transpose(1, 2)                   # (B, ceil(L/2), d_model)
```
Output length: `L_out = floor((L + 2*1 - 3)/2) + 1`. For even `L` this is exactly `L/2` (96→48→24); for odd `L` it is `(L+1)/2`.
BatchNorm1d must be applied while the tensor is `(B, C, L)`.

#### 10. `EncoderLayer`

```python
class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        d_ff = d_ff or 4*d_model
        self.attention = attention                 # AttentionLayer(ProbAttention(mask_flag=False,...))
        self.conv1 = nn.Conv1d(d_model, d_ff,   kernel_size=1)   # position-wise FFN part 1
        self.conv2 = nn.Conv1d(d_ff,   d_model, kernel_size=1)   # part 2
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):          # x: (B,L,d_model)
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)   # (B,L,d_model)
        x = x + self.dropout(new_x)                # residual
        y = x = self.norm1(x)                      # add & norm  (post-LN)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))  # (B,d_ff,L)
        y = self.dropout(self.conv2(y).transpose(-1,1))                  # (B,L,d_model)
        return self.norm2(x + y), attn
```
Encoder self-attention uses `mask_flag=False` (no causality — the encoder may look at its whole window).

#### 11. `Encoder` (attention layers interleaved with distilling)

```python
class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None)
    def forward(self, x, attn_mask=None):          # x: (B,L,d_model)
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)                  # halves L
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)  # last layer, no conv after
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)
        if self.norm is not None: x = self.norm(x)  # LayerNorm(d_model)
        return x, attns
```
Construct with `len(conv_layers) == len(attn_layers) - 1` so `zip` pairs correctly and the final attention layer has no distilling after it. `distil=False` ⇒ `conv_layers=None`.

#### 12. `EncoderStack` ("Informer-stack" variant)

Several independent `Encoder`s, the *i*-th receiving the last `L // 2**i` timesteps; outputs concatenated on the time axis.
```python
class EncoderStack(nn.Module):
    def __init__(self, encoders, inp_lens)         # e.g. e_layers=[3,2,1], inp_lens=[0,1,2]
    def forward(self, x, attn_mask=None):          # x: (B,L,d_model)
        x_stack, attns = [], []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[1] // (2 ** i_len)
            x_s, attn = encoder(x[:, -inp_len:, :])
            x_stack.append(x_s); attns.append(attn)
        return torch.cat(x_stack, dim=-2), attns   # (B, Σ lengths, d_model)
```
With `e_layers=[3,2,1]` and `L=96`: encoder0 sees 96 and has 2 convs → 24; encoder1 sees 48 with 1 conv → 24; encoder2 sees 24 with 0 convs → 24. Concatenated → `(B, 72, d_model)`. The pyramid (fewer layers for shorter inputs) is what makes the lengths align. The paper reports L + L/4 as the most robust pairing.

#### 13. `DecoderLayer`

```python
class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        d_ff = d_ff or 4*d_model
        self.conv1 = nn.Conv1d(d_model, d_ff, 1); self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        # THREE SEPARATE LayerNorm modules -- do NOT write `norm1 = norm2 = norm3 = ...`,
        # that would tie the learned affine parameters of all three.
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # x: (B,L_dec,d_model)   cross: (B,L_enc_out,d_model)
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x = self.norm1(x)
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))
        return self.norm3(x + y)                  # (B,L_dec,d_model)
```
- self-attention: `AttentionLayer(ProbAttention(mask_flag=True, factor, dropout), d_model, n_heads, mix=True)` → causal, so position `t` cannot see `t+1..`.
- cross-attention: `AttentionLayer(FullAttention(mask_flag=False, factor, dropout), d_model, n_heads, mix=False)` → every decoder position attends to the whole encoder output.

#### 14. `Decoder`

```python
class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None)
    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
        if self.norm is not None: x = self.norm(x)   # LayerNorm(d_model)
        return x                                     # (B,L_dec,d_model)
```

#### 15. Final projection & generative decoding

`self.projection = nn.Linear(d_model, c_out, bias=True)` applied to every decoder timestep, then keep only the tail:
```python
dec_out = self.projection(dec_out)          # (B, L_dec, c_out)
return dec_out[:, -self.pred_len:, :]       # (B, L_pred, c_out)
```
This is the **generative, one-forward-pass** decode: no loop, no feeding predictions back. The first `L_label` outputs are discarded.

---

### Full Model Assembly

```python
class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0')):
        super().__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)

        Attn = ProbAttention if attn == 'prob' else FullAttention

        self.encoder = Encoder(
            attn_layers=[
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout,
                                        output_attention=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model, d_ff, dropout=dropout, activation=activation)
                for _ in range(e_layers)],
            conv_layers=[ConvLayer(d_model) for _ in range(e_layers - 1)] if distil else None,
            norm_layer=nn.LayerNorm(d_model))

        self.decoder = Decoder(
            layers=[
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout,
                                        output_attention=False),
                                   d_model, n_heads, mix=mix),                 # masked self-attn
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout,
                                                 output_attention=False),
                                   d_model, n_heads, mix=False),               # cross-attn
                    d_model, d_ff, dropout=dropout, activation=activation)
                for _ in range(d_layers)],
            norm_layer=nn.LayerNorm(d_model))

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # x_enc      (B, L_enc, enc_in)   x_mark_enc (B, L_enc, d_time)
        # x_dec      (B, L_dec, dec_in)   x_mark_dec (B, L_dec, d_time)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)      # (B, L_enc, d_model)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
                                                             # (B, L_enc//2**(e_layers-1), d_model)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)       # (B, L_dec, d_model)
        dec_out = self.decoder(dec_out, enc_out,
                               x_mask=dec_self_mask, cross_mask=dec_enc_mask)
                                                             # (B, L_dec, d_model)
        dec_out = self.projection(dec_out)                    # (B, L_dec, c_out)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        return dec_out[:, -self.pred_len:, :]                 # (B, L_pred, c_out)
```

`InformerStack` is identical except `e_layers` is a **list** (e.g. `[3,2,1]`) and the encoder is an `EncoderStack` of per-depth `Encoder`s with `inp_lens = list(range(len(e_layers)))`.

---

### Default Hyperparameters

| Name | Default | Notes |
|---|---|---|
| `model` | `informer` | or `informerstack` |
| `features` | `M` | `M` / `S` / `MS` |
| `target` | `OT` | ETT target column |
| `freq` | `h` | `t` for ETTm |
| `seq_len` | 96 | encoder input length |
| `label_len` | 48 | decoder start-token length |
| `pred_len` | 24 | horizon |
| `enc_in` / `dec_in` / `c_out` | 7 / 7 / 7 | ETT `M`; `S`→1/1/1; `MS`→7/7/1 |
| `d_model` | 512 | |
| `n_heads` | 8 | paper also tried 16 (`h=16, d=32` in encoder) |
| `e_layers` | 2 | repo CLI default; paper main config = 3 |
| `d_layers` | 1 | repo CLI default; paper = 2 |
| `s_layers` | `3,2,1` | for `informerstack` |
| `d_ff` | 2048 | CLI default (constructor default is 512 — pass 2048 explicitly) |
| `factor` (`c`) | 5 | ProbSparse sampling factor |
| `padding` | 0 | zero placeholder (1 ⇒ ones) |
| `distil` | True | use ConvLayer distilling |
| `dropout` | 0.05 | CLI default (paper appendix used 0.1) |
| `attn` | `prob` | `full` ⇒ Informer† |
| `embed` | `timeF` | or `fixed` / `learned` |
| `activation` | `gelu` | FFN activation |
| `mix` | True | mix in decoder self-attention |
| `loss` | MSE | `nn.MSELoss()` |
| optimizer | Adam | `torch.optim.Adam` |
| `learning_rate` | 1e-4 | |
| `lradj` | `type1` | `lr = 1e-4 * 0.5**(epoch-1)` — halved every epoch |
| `batch_size` | 32 | `drop_last=True` |
| `train_epochs` | 6 | paper text says 8 |
| `patience` | 3 | early stopping on validation loss |
| `use_amp` | False | optional `torch.cuda.amp` |
| `itr` | 2 | repeat runs |
| `num_workers` | 0 | |

Paper `(seq_len, label_len, pred_len)` combinations used for ETT experiments:

| Dataset | pred_len values | typical `seq_len` | typical `label_len` |
|---|---|---|---|
| ETTh1 / ETTh2 | 24, 48, 168, 336, 720 | 48 / 96 / 168 / 336 / 720 | 24 / 48 / 168 / 336 / 336 |
| ETTm1 / ETTm2 | 24, 48, 96, 288, 672 | 96 / 384 / 672 | 48 / 96 / 288 |

**[UNVERIFIED]** The exact per-horizon `(seq_len, label_len)` pairs above: the paper only states the grid-search ranges `{24,48,96,168,336,480,720}` (ETTh/Weather/ECL) and `{24,48,96,192,288,480,672}` (ETTm) and requires `label_len < seq_len`; it does not publish the selected value per horizon. The repo's README scripts are the practical source; the pairs listed are the commonly used ones.

---

### Training Loop

```python
model     = Informer(...).float().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()
early     = EarlyStopping(patience=3)
scaler    = torch.cuda.amp.GradScaler() if use_amp else None

for epoch in range(train_epochs):                    # 6
    model.train()
    for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
        optimizer.zero_grad()
        pred, true = process_one_batch(...)          # builds dec_inp, calls model
        loss = criterion(pred, true)
        if use_amp:
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()

    vali_loss = evaluate(val_loader)                 # model.eval(), no_grad in spirit
    test_loss = evaluate(test_loader)                # logged only, never used for selection
    early(vali_loss, model, ckpt_dir)                # saves checkpoint.pth on improvement
    if early.early_stop: break
    adjust_learning_rate(optimizer, epoch + 1)       # AFTER early-stopping check

model.load_state_dict(torch.load(ckpt_dir + '/checkpoint.pth'))   # restore best
```

```python
def adjust_learning_rate(optimizer, epoch, args):    # lradj='type1'
    lr = args.learning_rate * (0.5 ** ((epoch - 1) // 1))   # 1e-4, 5e-5, 2.5e-5, ...
    for g in optimizer.param_groups: g['lr'] = lr
# lradj='type2': {2:5e-5, 4:1e-5, 6:5e-6, 8:1e-6, 10:5e-7, 15:1e-7, 20:5e-8}
```

```python
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.counter = 0; self.best_score = None; self.early_stop = False
        self.val_loss_min = np.inf; self.patience = patience; self.delta = delta
    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score; self.save(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_score = score; self.save(val_loss, model, path); self.counter = 0
```

- **Gradient clipping: none.** The reference implementation does not clip.
- **AMP:** optional via `torch.cuda.amp.autocast()` around the forward and `GradScaler` on the backward. Caution: `masked_fill_(..., -np.inf)` plus fp16 softmax can produce NaNs; prefer a large negative finite value (e.g. `-1e4`) or keep attention in fp32 if you enable AMP.
- Validation runs with `model.eval()`; wrap in `torch.no_grad()` (the reference omits it but should have it — saves memory).

---

### Evaluation

Metrics computed on the **standardized** scale (scaler fit on train), over all rolling windows of the test split with stride 1, averaged over channels for multivariate:

```python
MAE  = np.mean(np.abs(pred - true))
MSE  = np.mean((pred - true) ** 2)
RMSE = np.sqrt(MSE)
MAPE = np.mean(np.abs((pred - true) / true))
MSPE = np.mean(np.square((pred - true) / true))
```
`pred`, `true` are stacked to `(N_windows, L_pred, c_out)` before computing. Report **MSE and MAE**.

Prediction lengths evaluated:
- ETTh1 / ETTh2 / Weather / ECL: `{24, 48, 168, 336, 720}` (ECL in the paper uses `{48,168,336,720,960}`)
- ETTm1: `{24, 48, 96, 288, 672}`

Settings: **univariate** (`features='S'`, `enc_in=dec_in=c_out=1`, target `OT` only) and **multivariate** (`features='M'`, 7/7/7). Informer† = same model with `attn='full'`.

#### Reported univariate results (MSE / MAE), Table 1 of the paper

| Dataset | pred_len | Informer | Informer† | LogTrans | Reformer | LSTMa | DeepAR | ARIMA | Prophet |
|---|---|---|---|---|---|---|---|---|---|
| ETTh1 | 24 | 0.098 / 0.247 | 0.092 / 0.246 | 0.103 / 0.259 | 0.222 / 0.389 | 0.114 / 0.272 | 0.107 / 0.280 | 0.108 / 0.284 | 0.115 / 0.275 |
| ETTh1 | 48 | 0.158 / 0.319 | 0.161 / 0.322 | 0.167 / 0.328 | 0.284 / 0.445 | 0.193 / 0.358 | 0.162 / 0.327 | 0.175 / 0.424 | 0.168 / 0.330 |
| ETTh1 | 168 | 0.183 / 0.346 | 0.187 / 0.355 | 0.207 / 0.375 | 1.522 / 1.191 | 0.236 / 0.392 | 0.239 / 0.422 | 0.396 / 0.504 | 1.224 / 0.763 |
| ETTh1 | 336 | 0.222 / 0.387 | 0.215 / 0.369 | 0.230 / 0.398 | 1.860 / 1.124 | 0.590 / 0.698 | 0.445 / 0.552 | 0.468 / 0.593 | 1.549 / 1.820 |
| ETTh1 | 720 | 0.269 / 0.435 | 0.257 / 0.421 | 0.273 / 0.463 | 2.112 / 1.436 | 0.683 / 0.768 | 0.658 / 0.707 | 0.659 / 0.766 | 2.735 / 3.253 |
| ETTh2 | 24 | 0.093 / 0.240 | 0.099 / 0.241 | 0.102 / 0.255 | 0.263 / 0.437 | 0.155 / 0.307 | 0.098 / 0.263 | 3.554 / 0.445 | 0.199 / 0.381 |
| ETTh2 | 48 | 0.155 / 0.314 | 0.159 / 0.317 | 0.169 / 0.348 | 0.458 / 0.545 | 0.190 / 0.348 | 0.163 / 0.341 | 3.190 / 0.474 | 0.304 / 0.462 |
| ETTh2 | 168 | 0.232 / 0.389 | 0.235 / 0.390 | 0.246 / 0.422 | 1.029 / 0.879 | 0.385 / 0.514 | 0.255 / 0.414 | 2.800 / 0.595 | 2.145 / 1.068 |
| ETTh2 | 336 | 0.263 / 0.417 | 0.258 / 0.423 | 0.267 / 0.437 | 1.668 / 1.228 | 0.558 / 0.606 | 0.604 / 0.607 | 2.753 / 0.738 | 2.096 / 2.543 |
| ETTh2 | 720 | 0.277 / 0.431 | 0.285 / 0.442 | 0.303 / 0.493 | 2.030 / 1.721 | 0.640 / 0.681 | 0.429 / 0.580 | 2.878 / 1.044 | 3.355 / 4.664 |
| ETTm1 | 24 | 0.030 / 0.137 | 0.034 / 0.160 | 0.065 / 0.202 | 0.095 / 0.228 | 0.121 / 0.233 | 0.091 / 0.243 | 0.090 / 0.206 | 0.120 / 0.290 |
| ETTm1 | 48 | 0.069 / 0.203 | 0.066 / 0.194 | 0.078 / 0.220 | 0.249 / 0.390 | 0.305 / 0.411 | 0.219 / 0.362 | 0.179 / 0.306 | 0.133 / 0.305 |
| ETTm1 | 96 | 0.194 / 0.372 | 0.187 / 0.384 | 0.199 / 0.386 | 0.920 / 0.767 | 0.287 / 0.420 | 0.364 / 0.496 | 0.272 / 0.399 | 0.194 / 0.396 |
| ETTm1 | 288 | 0.401 / 0.554 | 0.409 / 0.548 | 0.411 / 0.572 | 1.108 / 1.245 | 0.524 / 0.584 | 0.948 / 0.795 | 0.462 / 0.558 | 0.452 / 0.574 |
| ETTm1 | 672 | 0.512 / 0.644 | 0.519 / 0.665 | 0.598 / 0.702 | 1.793 / 1.528 | 1.064 / 0.873 | 2.437 / 1.352 | 0.639 / 0.697 | 2.747 / 1.174 |

#### Reported multivariate results (MSE / MAE), Table 2 of the paper

| Dataset | pred_len | Informer | Informer† | LogTrans | Reformer | LSTMa | LSTnet |
|---|---|---|---|---|---|---|---|
| ETTh1 | 24 | 0.577 / 0.549 | 0.620 / 0.577 | 0.686 / 0.604 | 0.991 / 0.754 | 0.650 / 0.624 | 1.293 / 0.901 |
| ETTh1 | 48 | 0.685 / 0.625 | 0.692 / 0.671 | 0.766 / 0.757 | 1.313 / 0.906 | 0.702 / 0.675 | 1.456 / 0.960 |
| ETTh1 | 168 | 0.931 / 0.752 | 0.947 / 0.797 | 1.002 / 0.846 | 1.824 / 1.138 | 1.212 / 0.867 | 1.997 / 1.214 |
| ETTh1 | 336 | 1.128 / 0.873 | 1.094 / 0.813 | 1.362 / 0.952 | 2.117 / 1.280 | 1.424 / 0.994 | 2.655 / 1.369 |
| ETTh1 | 720 | 1.215 / 0.896 | 1.241 / 0.917 | 1.397 / 1.291 | 2.415 / 1.520 | 1.960 / 1.322 | 2.143 / 1.380 |
| ETTh2 | 24 | 0.720 / 0.665 | 0.753 / 0.727 | 0.828 / 0.750 | 1.531 / 1.613 | 1.143 / 0.813 | 2.742 / 1.457 |
| ETTh2 | 48 | 1.457 / 1.001 | 1.461 / 1.077 | 1.806 / 1.034 | 1.871 / 1.735 | 1.671 / 1.221 | 3.567 / 1.687 |
| ETTh2 | 168 | 3.489 / 1.515 | 3.485 / 1.612 | 4.070 / 1.681 | 4.660 / 1.846 | 4.117 / 1.674 | 3.242 / 2.513 |
| ETTh2 | 336 | 2.723 / 1.340 | 2.626 / 1.285 | 3.875 / 1.763 | 4.028 / 1.688 | 3.434 / 1.549 | 2.544 / 2.591 |
| ETTh2 | 720 | 3.467 / 1.473 | 3.548 / 1.495 | 3.913 / 1.552 | 5.381 / 2.015 | 3.963 / 1.788 | 4.625 / 3.709 |
| ETTm1 | 24 | 0.323 / 0.369 | 0.306 / 0.371 | 0.419 / 0.412 | 0.724 / 0.607 | 0.621 / 0.629 | 1.968 / 1.170 |
| ETTm1 | 48 | 0.494 / 0.503 | 0.465 / 0.470 | 0.507 / 0.583 | 1.098 / 0.777 | 1.392 / 0.939 | 1.999 / 1.215 |
| ETTm1 | 96 | 0.678 / 0.614 | 0.681 / 0.612 | 0.768 / 0.792 | 1.433 / 0.945 | 1.339 / 0.913 | 2.762 / 1.542 |
| ETTm1 | 288 | 1.056 / 0.786 | 1.162 / 0.879 | 1.462 / 1.320 | 1.820 / 1.094 | 1.740 / 1.124 | 1.257 / 2.076 |
| ETTm1 | 672 | 1.192 / 0.926 | 1.231 / 1.103 | 1.669 / 1.461 | 2.187 / 1.232 | 2.736 / 1.555 | 1.917 / 2.941 |

Reported numbers are averages over 5 random train/val shifting selections along time, with 10% of each dataset held out for validation.

---

### Implementation Checklist

1. `utils/masking.py` — `TriangularCausalMask`, `ProbMask`.
2. `utils/timefeatures.py` — `TimeFeature` subclasses, `time_features_from_frequency_str`, `time_features(dates, timeenc, freq)`.
3. `utils/tools.py` — `StandardScaler`, `EarlyStopping`, `adjust_learning_rate`, `dotdict`.
4. `utils/metrics.py` — `MAE, MSE, RMSE, MAPE, MSPE`, and `metric(pred, true) -> (mae, mse, rmse, mape, mspe)`.
5. `models/embed.py` — `PositionalEmbedding`, `TokenEmbedding`, `FixedEmbedding`, `TemporalEmbedding`, `TimeFeatureEmbedding`, `DataEmbedding`.
6. `models/attn.py` — `FullAttention`, `ProbAttention` (`_prob_QK`, `_get_initial_context`, `_update_context`), `AttentionLayer`.
7. `models/encoder.py` — `ConvLayer`, `EncoderLayer`, `Encoder`, `EncoderStack`.
8. `models/decoder.py` — `DecoderLayer`, `Decoder`.
9. `models/model.py` — `Informer`, `InformerStack`.
10. `data/data_loader.py` — `Dataset_ETT_hour`, `Dataset_ETT_minute`, `Dataset_Custom`, `Dataset_Pred`.
11. `exp/exp_basic.py`, `exp/exp_informer.py` — `_build_model`, `_get_data`, `_select_optimizer`, `_select_criterion`, `vali`, `train`, `test`, `predict`, `_process_one_batch`.
12. `main_informer.py` — argparse with the defaults in the table above, `data_parser` dict for per-dataset `enc_in/dec_in/c_out`, loop over `itr`.
13. Smoke tests: assert output shape `(B, L_pred, c_out)`; assert `ConvLayer` halves length; assert `ProbAttention` with `factor` large enough to force `u == L_Q` and `U_part == L_K` gives output close to `FullAttention` (unmasked); assert causal decoder self-attention output for position `t` is unchanged when inputs at `> t` are perturbed.

---

### Common Pitfalls

1. **Off-by-one / clamping in top-u.** `u = factor * ceil(ln(L_Q))` can exceed `L_Q` for short sequences (e.g. `L_Q=24`, `c=5` → `5*4 = 20`, fine; but `L_Q=7` → `5*2 = 10 > 7`). You **must** clamp `u = min(u, L_Q)` and `U_part = min(U_part, L_K)`, otherwise `topk` and `randint` blow up. Use natural log and `ceil` *before* multiplying by `factor`.
2. **`L_Q == L_V` requirement under masking.** `_get_initial_context` asserts `L_Q == L_V` when `mask_flag=True`, because `cumsum` over V is only meaningful for causal *self*-attention. Never use `ProbAttention(mask_flag=True)` for cross-attention. Conversely the unmasked branch works for any `L_Q`, `L_V`.
3. **Cumulative-sum context initialization.** Under masking, unselected rows must be initialized with `V.cumsum(dim=-2)` (running aggregate over visible values), **not** the global mean — using the mean leaks future values across the causal boundary. And do not normalize the cumsum: match the reference (`sum` as a simpler stand-in for `mean`).
4. **`.clone()` on the expanded mean context.** `V_sum.unsqueeze(-2).expand(...)` returns a non-contiguous view with stride 0 on the time axis; scattering into it without `.clone()` silently corrupts all rows. Always `.clone()`.
5. **`ProbMask` row alignment.** The causal mask must be *gathered with the same `index`* used to select queries — a plain `triu` mask of shape `(u, L_K)` is wrong, because row `r` of `scores` corresponds to original query position `index[b,h,r]`, not to `r`.
6. **Circular padding.** `nn.Conv1d(..., kernel_size=3, padding=1, padding_mode='circular')` must be applied on `(B, C, L)` (channels-first), so `permute(0,2,1)` first and `transpose(1,2)` back. On PyTorch < 1.5 `padding` had to be 2 for the same output length; on ≥ 1.5 use 1. If you get `L+1` or `L-1` out of `TokenEmbedding`, your padding is wrong.
7. **Never leak future values into the decoder placeholder.** `dec_inp = cat([batch_y[:, :label_len, :], zeros(B, pred_len, C)])`. Only the *timestamps* (`x_mark_dec`, which covers `label_len + pred_len`) may come from the future; the *values* for the prediction region must be exactly zero (or ones with `padding=1`). Slicing `batch_y[:, :label_len+pred_len, :]` directly into the decoder is the single most common fatal bug.
8. **Scaler fit on train only.** `scaler.fit(df_data[border1s[0]:border2s[0]])` then `transform` everything. Fitting on the full series leaks test statistics. Also note val/test start indices are offset by `-seq_len` to provide left context — that is intentional and not leakage.
9. **Distilling count and lengths.** `len(conv_layers) == e_layers - 1`. With distilling, `enc_out` has length `L_enc // 2**(e_layers-1)`, not `L_enc`. Don't assume the cross-attention key length equals `seq_len`. Also `seq_len` should be divisible by `2**(e_layers-1)` to avoid odd-length ceil behavior.
10. **BatchNorm1d placement.** In `ConvLayer`, normalize while shaped `(B, d_model, L)`. Applying it after the transpose normalizes over time instead of channels.
11. **Post-LN, and three distinct LayerNorms in the decoder.** `norm1/norm2/norm3` must be separate `nn.LayerNorm` instances (sharing one module ties the learned affine parameters). Both encoder and decoder use post-norm residuals (`norm(x + sublayer(x))`), not pre-norm.
12. **`mix=True` only on decoder self-attention.** Encoder layers and cross-attention use `mix=False`. Getting this wrong changes results silently (no shape error, since `view` preserves element count).
13. **LR schedule ordering.** `adjust_learning_rate` is called *after* the early-stopping check, with `epoch+1`, so epoch 1 trains at 1e-4 and epoch 2 at 5e-5.
14. **`drop_last=True` on the test loader** in the reference means the last partial batch is discarded; metrics differ slightly if you change this. Also `output_attention` changes the return type to a tuple — index `[0]`.
15. **`-np.inf` + softmax.** If an entire masked row is `-inf` (can happen at position 0 combined with dropout/fp16), softmax yields NaN. Use `-1e9`/`-1e4` if you hit NaNs, especially with AMP.


---

## 2. Autoformer

Autoformer is an encoder–decoder Transformer for **long-term multivariate time-series forecasting** (predict the next `L_pred` steps from the past `L_enc` steps). It makes two changes to the vanilla Transformer. First, **progressive series decomposition**: a cheap moving-average block (`series_decomp`) is inserted *inside* every encoder/decoder layer; it splits the hidden representation into a smooth **trend-cyclical** part and a **seasonal** (residual) part. The encoder throws its trend parts away and models only seasonality; the decoder *accumulates* its trend parts into a running trend prediction while refining the seasonal part. Second, **Auto-Correlation replaces self-attention**: instead of point-wise dot-product attention, it uses the FFT to compute the cross-correlation between queries and keys over all time lags τ, keeps the top-`k = ⌊c·log L⌋` lags, softmax-normalizes their correlation scores, and aggregates `Roll(V, τ_i)` (circularly shifted value series) weighted by those scores — a *series-level* (sub-series) aggregation with `O(L log L)` cost. The final forecast is `seasonal_part + trend_part`, sliced to the last `L_pred` steps.

All content below is taken from the paper (arXiv:2106.13008v5) and the official implementation (github.com/thuml/Autoformer). Where the two differ, the code is authoritative and the difference is called out.

---

### Notation & Tensor Shapes

| Symbol | Code name | Meaning | Default |
|---|---|---|---|
| `B` | — | batch size | 32 |
| `L_enc` | `seq_len` (paper: `I`) | encoder input length | 96 (36 for ILI) |
| `L_label` | `label_len` | "start-token" / label window length, taken from the **end of the encoder window** | 48 (18 for ILI) |
| `L_pred` | `pred_len` (paper: `O`) | forecast horizon | 96/192/336/720 (24/36/48/60 for ILI) |
| `L_dec` | — | decoder sequence length `= L_label + L_pred` | 144 for 96+48 |
| `d_model` | `d_model` | model width | 512 |
| `H` | `n_heads` | attention heads | 8 |
| `E` | `d_head = d_model // n_heads` | per-head channel width | 64 |
| `d_ff` | `d_ff` | FFN inner width | 2048 |
| `c` | `factor` | top-k factor, `k = int(c * log(L))` (natural log) | 1 (ETT) / 3 (others) |
| `enc_in` | `enc_in` | # input channels to encoder | 7 (ETT/ILI), 321 (Electricity), 862 (Traffic), 21 (Weather), 8 (Exchange) |
| `dec_in` | `dec_in` | # input channels to decoder | = `enc_in` |
| `c_out` | `c_out` | # output channels | = `enc_in` (multivariate) or 1 (univariate) |
| `kernel` | `moving_avg` | moving-average kernel of `series_decomp` | 25 |
| `d_mark` | — | # time (calendar) features per step | 4 for freq `h`, 5 for `t`, 3 for `d`, 2 for `w` |

Canonical tensor shapes:

```
x_enc        : [B, L_enc, enc_in]              # standardized past values
x_mark_enc   : [B, L_enc, d_mark]              # past time features
x_dec        : [B, L_label + L_pred, dec_in]   # label window then zeros (see Data Pipeline)
x_mark_dec   : [B, L_label + L_pred, d_mark]   # time features for label + future window
output       : [B, L_pred, c_out]
```

Inside Auto-Correlation the time axis is moved to the **last** position: `[B, H, E, L]`.

---

### Data Pipeline

#### Dataset format

All 6 benchmarks are single CSV files with a first column named `date` (parseable timestamps) and then one column per variate. The last column is the univariate `target` (`OT` for ETT / Electricity / Traffic / Weather / Exchange, `OT` also used for ILI in the official scripts).

| Dataset | file | #variates | sampling freq | `--freq` |
|---|---|---|---|---|
| ETTh1 / ETTh2 | `ETTh1.csv` | 7 | hourly | `h` |
| ETTm1 / ETTm2 | `ETTm1.csv` | 7 | 15 min | `t` |
| Electricity (ECL) | `electricity.csv` | 321 | hourly | `h` |
| Exchange | `exchange_rate.csv` | 8 | daily | `d` |
| Traffic | `traffic.csv` | 862 | hourly | `h` |
| Weather | `weather.csv` | 21 | 10 min | `h` in official scripts |
| ILI | `national_illness.csv` | 7 | weekly | `h` in official scripts (default) |

Column reordering (for the generic `Dataset_Custom` loader): `df = df[['date'] + [all columns except date and target] + [target]]`, so the target is always last.

#### Splits

* **ETT (hour)**: `border1s = [0, 12*30*24 - L_enc, 12*30*24 + 4*30*24 - L_enc]`, `border2s = [12*30*24, 12*30*24 + 4*30*24, 12*30*24 + 8*30*24]` → 12 months train / 4 months val / 4 months test.
* **ETT (minute, 15 min)**: same formulas multiplied by 4: `border1s = [0, 12*30*24*4 - L_enc, (12*30*24 + 4*30*24)*4 - L_enc]`, `border2s = [12*30*24*4, (12*30*24+4*30*24)*4, (12*30*24+8*30*24)*4]`.
* **All other datasets** (`Dataset_Custom`): `num_train = int(0.7*N)`, `num_test = int(0.2*N)`, `num_vali = N - num_train - num_test`; `border1s = [0, num_train - L_enc, N - num_test - L_enc]`, `border2s = [num_train, num_train + num_vali, N]`.
  `set_type`: 0=train, 1=val, 2=test. Subtracting `L_enc` from val/test starts lets the first val/test window have full history.

#### Standardization

Fit `StandardScaler` (`mean`, `std` per column) **only on the train slice** `df_data[border1s[0]:border2s[0]]`, then transform the *whole* series. Never re-fit on val/test. All reported metrics are computed on this standardized scale.

#### Time-feature encoding

Two modes; use `timeenc=1` (`--embed timeF`, the default for Autoformer):

* `timeenc = 0` (`--embed fixed`/`learned`): integer calendar features `[month, day, weekday, hour]` (and `minute//15` when `freq='t'`), consumed by `TemporalEmbedding` (`nn.Embedding` lookups).
* `timeenc = 1` (`--embed timeF`): float features scaled to `[-0.5, 0.5]`, consumed by `TimeFeatureEmbedding` (a single `nn.Linear(d_inp, d_model, bias=False)`):
  * `MinuteOfHour = minute/59 - 0.5`, `HourOfDay = hour/23 - 0.5`, `DayOfWeek = dayofweek/6 - 0.5`, `DayOfMonth = (day-1)/30 - 0.5`, `DayOfYear = (dayofyear-1)/365 - 0.5`, `MonthOfYear = (month-1)/11 - 0.5`, `WeekOfYear = (week-1)/52 - 0.5`, `SecondOfMinute = second/59 - 0.5`.
  * `d_inp` per freq: `{'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}`. For `h`: `[MonthOfYear, DayOfMonth, DayOfWeek, HourOfDay]`; for `t`: same + `MinuteOfHour`; for `d`: `[MonthOfYear, DayOfMonth, DayOfWeek]`.

#### Windowing (`__getitem__`)

```python
s_begin = index
s_end   = s_begin + L_enc
r_begin = s_end - L_label          # label window overlaps the tail of the encoder window
r_end   = r_begin + L_label + L_pred

seq_x      = data[s_begin:s_end]         # [L_enc, C]
seq_y      = data[r_begin:r_end]         # [L_label + L_pred, C]
seq_x_mark = stamp[s_begin:s_end]        # [L_enc, d_mark]
seq_y_mark = stamp[r_begin:r_end]        # [L_label + L_pred, d_mark]
__len__ = len(data) - L_enc - L_pred + 1
```

#### DECODER INPUT CONSTRUCTION (critical)

Two separate things happen: (a) the *dataloader-side* `x_dec` tensor built in the training loop, and (b) the *model-side* seasonal/trend initialization built inside `Autoformer.forward`.

(a) In the experiment loop:

```python
# batch_y: [B, L_label + L_pred, C]
dec_inp = torch.zeros_like(batch_y[:, -L_pred:, :])            # [B, L_pred, C]
x_dec   = torch.cat([batch_y[:, :L_label, :], dec_inp], dim=1) # [B, L_label + L_pred, C]
```

So `x_dec` = real label-window values followed by zeros. **Note:** the Autoformer model only uses `x_dec.shape` (for `B` and `C`) — it does *not* read `x_dec`'s values. Its seasonal/trend init are derived from `x_enc`. (Other models in the same repo do use `x_dec`, which is why it is still passed.)

(b) Inside `Autoformer.forward` (paper Eq. 2):

```python
mean  = x_enc.mean(dim=1).unsqueeze(1).repeat(1, L_pred, 1)            # [B, L_pred, enc_in]
zeros = torch.zeros(x_dec.shape[0], L_pred, x_dec.shape[2],
                    device=x_enc.device)                                # [B, L_pred, dec_in]

seasonal_init, trend_init = self.decomp(x_enc)   # each [B, L_enc, enc_in]

# seasonal init = last L_label seasonal steps  ++  zeros
seasonal_init = torch.cat([seasonal_init[:, -L_label:, :], zeros], dim=1)  # [B, L_label+L_pred, enc_in]
# trend init   = last L_label trend steps      ++  encoder mean repeated
trend_init    = torch.cat([trend_init[:,    -L_label:, :], mean],  dim=1)  # [B, L_label+L_pred, enc_in]
```

`seasonal_init` goes through `dec_embedding` into `d_model`; `trend_init` stays in data space (`enc_in == c_out` channels) and is the initial value `T_de^0` that the decoder adds its projected trends to.

---

### Module-by-Module Spec

#### 1. `moving_avg`

```python
class moving_avg(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1):
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):                      # x: [B, L, D]
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)   # [B, (k-1)//2, D]
        end   = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)   # [B, (k-1)//2, D]
        x = torch.cat([front, x, end], dim=1)  # [B, L + 2*((k-1)//2), D]
        x = self.avg(x.permute(0, 2, 1))       # [B, D, L]  (exact only for odd k)
        return x.permute(0, 2, 1)              # [B, L, D]
```

Replicate ("edge") padding of the first/last time step, **not** zero padding, so the smoothed series does not dip at the boundaries. With odd `kernel_size` (default 25) the output length is exactly `L`.

#### 2. `series_decomp`

```python
class series_decomp(nn.Module):
    def __init__(self, kernel_size: int = 25):
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):            # x: [B, L, D]
        moving_mean = self.moving_avg(x)   # trend-cyclical, [B, L, D]
        res = x - moving_mean              # seasonal,        [B, L, D]
        return res, moving_mean            # (seasonal, trend)  <-- THIS ORDER
```

Math (paper Eq. 1): `X_t = AvgPool(Padding(X))`, `X_s = X - X_t`. Return order is `(seasonal, trend)`.

#### 3. `series_decomp_multi` — **variant, not used by Autoformer** **[UNVERIFIED for Autoformer]**

The official Autoformer repo contains **only** the single-kernel `series_decomp`. A multi-kernel variant (average of moving averages with several kernel sizes, e.g. `[24, 12]`, optionally with learned softmax weights over kernels) appears in the follow-up FEDformer codebase from the same group, not in Autoformer. If you implement it, do it as:

```python
class series_decomp_multi(nn.Module):           # OPTIONAL variant
    def __init__(self, kernel_sizes=(25,)):
        self.movings = nn.ModuleList([moving_avg(k, 1) for k in kernel_sizes])
    def forward(self, x):
        trend = torch.stack([m(x) for m in self.movings], dim=-1).mean(-1)  # [B, L, D]
        return x - trend, trend
```

Default configuration must use the single-kernel `series_decomp(25)`.

#### 4. `AutoCorrelation`

Constructor: `AutoCorrelation(mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False)`.
`mask_flag`, `scale`, `attention_dropout` are stored but **never used** in the forward pass — Auto-Correlation applies no causal mask and no dropout. Keep the arguments for API compatibility.

Forward signature: `forward(queries, keys, values, attn_mask) -> (out, attn_or_None)` with

```
queries: [B, L, H, E]      # L = query length
keys   : [B, S, H, E]
values : [B, S, H, D]      # D == E in practice
out    : [B, L, H, E]
```

**Step 0 — length matching (`L_q` vs `L_v`).** Exactly as in the official code:

```python
B, L, H, E = queries.shape
_, S, _, D = values.shape
if L > S:
    zeros  = torch.zeros_like(queries[:, :(L - S), :]).float()   # [B, L-S, H, E]
    values = torch.cat([values, zeros], dim=1)                   # [B, L, H, D]
    keys   = torch.cat([keys,   zeros], dim=1)                   # [B, L, H, E]
else:
    values = values[:, :L, :, :]                                 # truncate
    keys   = keys[:,   :L, :, :]
```
i.e. **if the query is longer, zero-pad keys and values at the end; if shorter, truncate keys and values to the first `L` steps.** In the decoder's cross Auto-Correlation `L = L_label + L_pred = 144 > S = L_enc = 96`, so 48 zero steps are appended.

**Step 1 — rFFT along time.** Permute to put time last, then `rfft` on `dim=-1`:

```python
q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)  # [B, H, E, L//2+1] complex
k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(),    dim=-1)  # [B, H, E, L//2+1] complex
```
`permute(0,2,3,1)` maps `[B, L, H, E] -> [B, H, E, L]`.

**Step 2 — cross-correlation in frequency domain.**
```python
res = q_fft * torch.conj(k_fft)    # [B, H, E, L//2+1]
```
This is the Wiener–Khinchin identity: `R_{Q,K}(τ) = IFFT( FFT(Q) · conj(FFT(K)) )(τ)`, the circular cross-correlation `Σ_t Q_t K_{t-τ}` (unnormalized, so magnitudes are ~`L`·variance; the subsequent softmax handles the scale).

**Step 3 — irFFT back to the lag domain.**
```python
corr = torch.fft.irfft(res, n=L, dim=-1)   # [B, H, E, L]  real; index τ = 0..L-1 is the lag
```
Dim ordering is exactly `[B, H, E, L]` where the last axis indexes the time lag τ.

**Step 4/5/6 — top-k lag selection, softmax, time-delay aggregation.** Three variants; all take `values` permuted to `[B, H, D, L]` and `corr` of shape `[B, H, E, L]`, and return `[B, H, D, L]`.

`top_k = int(self.factor * math.log(length))` with `length = L` and natural log. E.g. `c=1, L=96 → int(4.564) = 4`; `c=3, L=96 → int(13.69) = 13`; `c=3, L=144 → int(14.9) = 14`.

**(a) `time_delay_agg_training` (used when `self.training` is True)** — one shared set of lags for the whole batch:

```python
def time_delay_agg_training(self, values, corr):
    # values: [B, H, D, L]   corr: [B, H, E, L]
    head, channel, length = values.shape[1], values.shape[2], values.shape[3]
    top_k = int(self.factor * math.log(length))
    mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)          # [B, L]  (mean over H then over E)
    index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]   # [top_k] lags, shared across batch
    weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)  # [B, top_k]
    tmp_corr = torch.softmax(weights, dim=-1)                        # [B, top_k]
    delays_agg = torch.zeros_like(values).float()                    # [B, H, D, L]
    for i in range(top_k):
        pattern = torch.roll(values, -int(index[i]), -1)             # circular shift LEFT by lag
        delays_agg = delays_agg + pattern * tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1) \
                                              .repeat(1, head, channel, length)
    return delays_agg                                                # [B, H, D, L]
```
Note: lag **indices** come from the batch-mean correlation (a "batch-normalization style" design, paper App. G "Speedup version"), but the softmax **weights** are still per-sample.

**(b) `time_delay_agg_inference` (used when `self.training` is False)** — per-sample lags via `gather` on a tiled index:

```python
def time_delay_agg_inference(self, values, corr):
    batch, head, channel, length = values.shape
    init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0) \
                      .repeat(batch, head, channel, 1).to(values.device)   # [B, H, D, L]
    top_k = int(self.factor * math.log(length))
    mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)                # [B, L]
    weights, delay = torch.topk(mean_value, top_k, dim=-1)                 # [B, top_k], [B, top_k]
    tmp_corr = torch.softmax(weights, dim=-1)                              # [B, top_k]
    tmp_values = values.repeat(1, 1, 1, 2)                                 # [B, H, D, 2L] emulate wrap-around
    delays_agg = torch.zeros_like(values).float()
    for i in range(top_k):
        tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1) \
                                            .repeat(1, head, channel, length)   # [B, H, D, L], values in [0, 2L)
        pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)             # [B, H, D, L]
        delays_agg = delays_agg + pattern * tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1) \
                                                  .repeat(1, head, channel, length)
    return delays_agg
```
`gather` on the doubled tensor with index `t + τ` is exactly `roll(values, -τ, -1)` but with a *different τ per batch element*. `delay` must be `long` (topk indices already are).

**(c) `time_delay_agg_full` (standard / non-speedup version, paper Eq. 6 literally)** — a separate lag set per `(B, H, E)` position; not used by default but implement it for completeness:

```python
def time_delay_agg_full(self, values, corr):
    batch, head, channel, length = values.shape
    init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0) \
                      .repeat(batch, head, channel, 1).to(values.device)   # [B, H, D, L]
    top_k = int(self.factor * math.log(length))
    weights, delay = torch.topk(corr, top_k, dim=-1)        # [B, H, E, top_k] each
    tmp_corr = torch.softmax(weights, dim=-1)               # [B, H, E, top_k]
    tmp_values = values.repeat(1, 1, 1, 2)                  # [B, H, D, 2L]
    delays_agg = torch.zeros_like(values).float()
    for i in range(top_k):
        tmp_delay = init_index + delay[..., i].unsqueeze(-1) # [B, H, D, L]
        pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
        delays_agg = delays_agg + pattern * tmp_corr[..., i].unsqueeze(-1)
    return delays_agg
```

**Forward glue:**

```python
if self.training:
    V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
else:
    V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
# V: [B, L, H, D]
if self.output_attention:
    return V.contiguous(), corr.permute(0, 3, 1, 2)   # attn: [B, L, H, E]
return V.contiguous(), None
```

#### 5. `AutoCorrelationLayer`

```python
class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads, d_keys=None, d_values=None):
        d_keys   = d_keys   or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys   * n_heads)
        self.key_projection   = nn.Linear(d_model, d_keys   * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection   = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape            # queries: [B, L, d_model]
        _, S, _ = keys.shape               # keys/values: [B, S, d_model]
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)   # [B, L, H, E]
        keys    = self.key_projection(keys).view(B, S, H, -1)        # [B, S, H, E]
        values  = self.value_projection(values).view(B, S, H, -1)    # [B, S, H, E]
        out, attn = self.inner_correlation(queries, keys, values, attn_mask)  # [B, L, H, E]
        out = out.view(B, L, -1)                                     # [B, L, d_model]
        return self.out_projection(out), attn                        # [B, L, d_model]
```

All four projections have `bias=True` (PyTorch default).

#### 6. `DataEmbedding_wo_pos`

```python
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        self.tokenConv = nn.Conv1d(c_in, d_model, kernel_size=3, padding=1,
                                   padding_mode='circular', bias=False)
        # init: kaiming_normal_(w, mode='fan_in', nonlinearity='leaky_relu') for every Conv1d
    def forward(self, x):                      # x: [B, L, c_in]
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)   # [B, L, d_model]
```
(`padding=1` for torch >= 1.5; older versions needed `padding=2`. Use 1.)

```python
class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='timeF', freq='h', dropout=0.05):
        self.value_embedding    = TokenEmbedding(c_in, d_model)
        self.temporal_embedding = (TimeFeatureEmbedding(d_model, embed_type, freq)
                                   if embed_type == 'timeF'
                                   else TemporalEmbedding(d_model, embed_type, freq))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, x_mark):              # x: [B, L, c_in], x_mark: [B, L, d_mark]
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)   # [B, L, d_model]
        return self.dropout(x)
```

Sub-embeddings:
* `TimeFeatureEmbedding`: `nn.Linear(d_inp, d_model, bias=False)` applied to the float time features → `[B, L, d_model]`.
* `TemporalEmbedding` (only if `embed_type != 'timeF'`): sums `month_embed(x[:,:,0]) + day_embed(x[:,:,1]) + weekday_embed(x[:,:,2]) + hour_embed(x[:,:,3]) (+ minute_embed(x[:,:,4]) if freq=='t')`, with vocab sizes `month=13, day=32, weekday=7, hour=24, minute=4`. `Embed = FixedEmbedding` (frozen sinusoidal table) when `embed_type=='fixed'`, else `nn.Embedding`.

**No positional encoding.** Reason (stated in the model source): "The series-wise connection inherently contains the sequential information. Thus, we can discard the position embedding of transformers." Auto-Correlation aggregates whole *shifted sub-series* rather than permutation-invariant point sets, and `TokenEmbedding`'s kernel-3 convolution already injects local ordering, so absolute positions are unnecessary (and would fight the roll-invariance the mechanism relies on).

#### 7. `my_Layernorm`

```python
class my_Layernorm(nn.Module):
    """LayerNorm specialized for the seasonal part: removes the per-series temporal mean."""
    def __init__(self, channels):
        self.layernorm = nn.LayerNorm(channels)
    def forward(self, x):                                   # [B, L, d_model]
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)   # [B, L, d_model]
        return x_hat - bias
```
Standard LayerNorm over the channel dim, then subtract the mean over the time dim — keeps the seasonal representation zero-mean in time.

#### 8. `EncoderLayer`

Constructor: `EncoderLayer(attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu")`; `d_ff = d_ff or 4*d_model`. Sub-modules: `attention` (an `AutoCorrelationLayer`), `conv1 = nn.Conv1d(d_model, d_ff, 1, bias=False)`, `conv2 = nn.Conv1d(d_ff, d_model, 1, bias=False)`, `decomp1 = series_decomp(moving_avg)`, `decomp2 = series_decomp(moving_avg)`, `dropout`, `activation = F.relu if activation=="relu" else F.gelu`.

```python
def forward(self, x, attn_mask=None):          # x: [B, L_enc, d_model]
    new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)   # [B, L_enc, d_model]
    x = x + self.dropout(new_x)                                  # residual
    x, _ = self.decomp1(x)                                       # keep seasonal, DISCARD trend
    y = x
    y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))   # [B, d_ff, L_enc]
    y = self.dropout(self.conv2(y).transpose(-1, 1))                    # [B, L_enc, d_model]
    res, _ = self.decomp2(x + y)                                 # keep seasonal, DISCARD trend
    return res, attn                                             # [B, L_enc, d_model]
```
Matches paper Eq. 3; the trend outputs (`_`) are the "eliminated trend part". `transpose(-1,1)` on a 3-D tensor swaps dims 1 and 2 → `[B, d_model, L]`, which is what `Conv1d` wants.

#### 9. `Encoder`

```python
class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
    def forward(self, x, attn_mask=None):      # x: [B, L_enc, d_model]
        attns = []
        for attn_layer in self.attn_layers:            # conv_layers is None for Autoformer
            x, attn = attn_layer(x, attn_mask=attn_mask)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)                            # my_Layernorm(d_model)
        return x, attns                                 # [B, L_enc, d_model]
```
(`conv_layers` are Informer-style distilling layers; Autoformer passes `None`. If provided, the pattern is: run `attn_layer` then `conv_layer` for the first `len(conv_layers)` layers, then the last `attn_layer`.)

#### 10. `DecoderLayer`

Constructor: `DecoderLayer(self_attention, cross_attention, d_model, c_out, d_ff=None, moving_avg=25, dropout=0.1, activation="relu")`. Sub-modules add `decomp3` and

```python
self.projection = nn.Conv1d(d_model, c_out, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=False)
```

```python
def forward(self, x, cross, x_mask=None, cross_mask=None):
    # x:     [B, L_dec, d_model]      (L_dec = L_label + L_pred)
    # cross: [B, L_enc, d_model]      (encoder output)
    x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
    x, trend1 = self.decomp1(x)                                   # [B, L_dec, d_model] each
    x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
    x, trend2 = self.decomp2(x)
    y = x
    y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))   # [B, d_ff, L_dec]
    y = self.dropout(self.conv2(y).transpose(-1, 1))                    # [B, L_dec, d_model]
    x, trend3 = self.decomp3(x + y)

    residual_trend = trend1 + trend2 + trend3                           # [B, L_dec, d_model]
    residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
    return x, residual_trend      # x: [B, L_dec, d_model], residual_trend: [B, L_dec, c_out]
```
This is paper Eq. 4: the `W_{l,i}` projectors are merged into one shared kernel-3 circular `Conv1d(d_model → c_out, bias=False)` applied to the *sum* of the three trends. The self Auto-Correlation is built with `mask_flag=True` and the cross one with `mask_flag=False`, but `AutoCorrelation` ignores the flag, so **no causal masking actually happens** — pass `x_mask=None, cross_mask=None`.

#### 11. `Decoder`

```python
class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer               # my_Layernorm(d_model)
        self.projection = projection         # nn.Linear(d_model, c_out, bias=True)
    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        # x: [B, L_dec, d_model], cross: [B, L_enc, d_model], trend: [B, L_dec, c_out]
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend        # trend accumulation, [B, L_dec, c_out]
        if self.norm is not None:
            x = self.norm(x)                      # [B, L_dec, d_model]
        if self.projection is not None:
            x = self.projection(x)                # [B, L_dec, c_out]
        return x, trend                           # (seasonal_part, trend_part)
```
The "seasonal + trend output sum" happens in the top-level model: `dec_out = trend_part + seasonal_part`.

---

### Full Model Assembly

```python
class Autoformer(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len   = configs.seq_len      # L_enc
        self.label_len = configs.label_len    # L_label
        self.pred_len  = configs.pred_len     # L_pred
        self.output_attention = configs.output_attention

        self.decomp = series_decomp(configs.moving_avg)          # kernel 25

        # NO positional embedding
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model,
                                                  configs.embed, configs.freq, configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model,
                                                  configs.embed, configs.freq, configs.dropout)

        self.encoder = Encoder(
            [EncoderLayer(
                AutoCorrelationLayer(
                    AutoCorrelation(False, configs.factor,
                                    attention_dropout=configs.dropout,
                                    output_attention=configs.output_attention),
                    configs.d_model, configs.n_heads),
                configs.d_model, configs.d_ff,
                moving_avg=configs.moving_avg, dropout=configs.dropout,
                activation=configs.activation)
             for _ in range(configs.e_layers)],
            norm_layer=my_Layernorm(configs.d_model))

        self.decoder = Decoder(
            [DecoderLayer(
                AutoCorrelationLayer(                       # self
                    AutoCorrelation(True, configs.factor,
                                    attention_dropout=configs.dropout, output_attention=False),
                    configs.d_model, configs.n_heads),
                AutoCorrelationLayer(                       # cross
                    AutoCorrelation(False, configs.factor,
                                    attention_dropout=configs.dropout, output_attention=False),
                    configs.d_model, configs.n_heads),
                configs.d_model, configs.c_out, configs.d_ff,
                moving_avg=configs.moving_avg, dropout=configs.dropout,
                activation=configs.activation)
             for _ in range(configs.d_layers)],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # x_enc:      [B, L_enc, enc_in]
        # x_mark_enc: [B, L_enc, d_mark]
        # x_dec:      [B, L_label + L_pred, dec_in]   (values unused; shape used)
        # x_mark_dec: [B, L_label + L_pred, d_mark]

        # ---- decomposition init ----
        mean  = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)   # [B, L_pred, enc_in]
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]],
                            device=x_enc.device)                                    # [B, L_pred, dec_in]
        seasonal_init, trend_init = self.decomp(x_enc)                              # [B, L_enc, enc_in] x2
        trend_init    = torch.cat([trend_init[:, -self.label_len:, :], mean],  dim=1)  # [B, L_dec, enc_in]
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)  # [B, L_dec, enc_in]

        # ---- encoder ----
        enc_out = self.enc_embedding(x_enc, x_mark_enc)              # [B, L_enc, d_model]
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)  # [B, L_enc, d_model]

        # ---- decoder ----
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)      # [B, L_dec, d_model]
        seasonal_part, trend_part = self.decoder(
            dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
            trend=trend_init)                                        # [B, L_dec, c_out] x2

        # ---- final ----
        dec_out = trend_part + seasonal_part                         # [B, L_dec, c_out]
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        return dec_out[:, -self.pred_len:, :]                        # [B, L_pred, c_out]
```

Note `dec_embedding` receives `seasonal_init` (which has `enc_in == dec_in` channels) together with the **full** `x_mark_dec` of length `L_dec`, so the conv-embedding length matches.

---

### Default Hyperparameters

| Arg | Default | Notes |
|---|---|---|
| `d_model` | 512 | |
| `n_heads` | 8 | `d_head = 64` |
| `e_layers` | 2 | encoder layers `N` |
| `d_layers` | 1 | decoder layers `M` |
| `d_ff` | 2048 | |
| `moving_avg` | 25 | odd kernel |
| `factor` (`c`) | 1 | official scripts use **1 for ETT**, **3 for Electricity / Exchange / Traffic / Weather / ILI** |
| `dropout` | 0.05 | |
| `activation` | `gelu` | `relu` is the layer-level default but the CLI default is `gelu` |
| `embed` | `timeF` | float time features |
| `optimizer` | Adam, `lr = 1e-4` | default betas (0.9, 0.999) |
| `loss` | MSE (`nn.MSELoss`) | |
| `batch_size` | 32 | |
| `train_epochs` | 10 | |
| `patience` | 3 | early stopping on validation loss |
| `lradj` | `type1` | `lr = 1e-4 * 0.5**(epoch-1)`, applied each epoch |
| `use_amp` | False | |
| `itr` | 2 (scripts use 1) | number of repeated runs |
| `num_workers` | 10 | dataloader |
| `L_enc` (`seq_len`) | 96 | **36 for ILI** |
| `L_label` (`label_len`) | 48 | **18 for ILI** |
| `L_pred` (`pred_len`) | 96 | benchmark set {96, 192, 336, 720}; **{24, 36, 48, 60} for ILI** |
| `enc_in` / `dec_in` / `c_out` | 7 / 7 / 7 | per dataset: ETT & ILI 7, Exchange 8, Weather 21, Electricity 321, Traffic 862; `c_out = 1` for univariate (`--features S`) |
| `features` | `M` | `M` multivariate→multivariate, `S` univariate, `MS` multivariate→univariate |

ETT scripts additionally run `pred_len ∈ {24, 48, 168, 336, 720}` for the appendix "full ETT benchmark".

---

### Training Loop

```python
model     = Autoformer(configs).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)   # 1e-4
criterion = nn.MSELoss()
early_stopping = EarlyStopping(patience=args.patience)                    # 3
scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

for epoch in range(args.train_epochs):                 # 10
    model.train()
    for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
        optimizer.zero_grad()
        batch_x, batch_y = batch_x.float().to(device), batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)

        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).to(device)

        if args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        f_dim   = -1 if args.features == 'MS' else 0
        outputs = outputs[:, -args.pred_len:, f_dim:]        # [B, L_pred, c]
        target  = batch_y[:, -args.pred_len:, f_dim:]        # [B, L_pred, c]
        loss = criterion(outputs, target)

        if args.use_amp:
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()

    vali_loss = evaluate(model, vali_loader, criterion)      # model.eval(), no_grad
    early_stopping(vali_loss, model, ckpt_dir)               # saves checkpoint.pth on improvement
    if early_stopping.early_stop:
        break
    adjust_learning_rate(optimizer, epoch + 1, args)         # lradj 'type1'

model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'checkpoint.pth')))
```

* **LR schedule (`type1`)**: at the start of epoch index `e` (1-based, called *after* the epoch), set `lr = learning_rate * 0.5 ** (e - 1)` — i.e. halve the LR every epoch. (`type2` is an explicit dict `{2:5e-5, 4:1e-5, 6:5e-6, 8:1e-6, 10:5e-7, 15:1e-7, 20:5e-8}`.)
* **Early stopping**: patience 3, `delta = 0`; counter increments when `-vali_loss` fails to exceed the best score; on improvement, save `state_dict` to `checkpoint.pth`. Restore the best checkpoint before testing.
* **No gradient clipping** and **no weight decay** in the official code.
* **AMP**: `torch.cuda.amp.autocast` + `GradScaler`, off by default. Caveat: `torch.fft` ops are not fp16-safe on all versions; if AMP produces NaNs, wrap the `AutoCorrelation.forward` FFT section in `with torch.cuda.amp.autocast(enabled=False):` and cast q/k to `float32`.
* Model selection uses **validation** loss only; the test loss printed during training is diagnostic.

---

### Evaluation

* Predictions and targets are compared **on the standardized scale** (the scaler is fit on train and never inverted for metrics).
* Metrics: `MAE = mean(|pred - true|)`, `MSE = mean((pred - true)^2)`; the repo also computes `RMSE`, `MAPE`, `MSPE`. Reported: **MSE and MAE**.
* Aggregate over the whole test set: collect `preds`/`trues` of shape `[n_batches, B, L_pred, c]`, reshape to `[-1, L_pred, c]`, then average elementwise.
* **Multivariate setting** (`--features M`): all `c_out = enc_in` channels predicted and scored.
* **Univariate setting** (`--features S`): only the `target` column (`OT`); `enc_in = dec_in = c_out = 1`. Univariate results are reported for the ETT datasets.
* Datasets: ETT (ETTh1, ETTh2, ETTm1, ETTm2), Electricity, Exchange-Rate, Traffic, Weather, ILI (national illness). Horizons {96, 192, 336, 720}, and {24, 36, 48, 60} for ILI.
* Each configuration is run and averaged over repeats (`--itr`); the paper reports mean (and, in Appendix E.4, standard deviations).

**Reported numbers.** The paper's headline claim is a **38% average relative MSE improvement** over the previous best baselines across the six benchmarks (baselines include Informer, LogTrans, Reformer, LSTNet, LSTM, TCN, N-BEATS, DeepAR, Prophet, ARIMA). **[UNVERIFIED]** The full Table 2 could not be extracted from the fetched source; the following commonly-cited Autoformer multivariate MSE/MAE values are given only as rough targets and must not be treated as exact:

| Dataset (multivariate) | horizon | MSE | MAE |
|---|---|---|---|
| ETTm2 | 96 | ~0.255 | ~0.339 |
| ETTm2 | 720 | ~0.422 | ~0.419 |
| Electricity | 96 | ~0.201 | ~0.317 |
| Exchange | 96 | ~0.197 | ~0.323 |
| Traffic | 96 | ~0.613 | ~0.388 |
| Weather | 96 | ~0.266 | ~0.336 |
| ILI | 24 | ~3.48 | ~1.29 |

All of the above are **[UNVERIFIED]**; consult Table 2 of the paper for authoritative values.

---

### Implementation Checklist

Write these files/classes in this order:

1. `layers/Autoformer_EncDec.py`
   1. `moving_avg(kernel_size, stride)`
   2. `series_decomp(kernel_size)` → returns `(seasonal, trend)`
   3. *(optional)* `series_decomp_multi(kernel_sizes)`
   4. `my_Layernorm(channels)`
2. `layers/AutoCorrelation.py`
   5. `AutoCorrelation(mask_flag, factor, scale, attention_dropout, output_attention)` with `time_delay_agg_training`, `time_delay_agg_inference`, `time_delay_agg_full`, `forward`
   6. `AutoCorrelationLayer(correlation, d_model, n_heads, d_keys, d_values)`
3. `layers/Embed.py`
   7. `TokenEmbedding(c_in, d_model)`
   8. `FixedEmbedding(c_in, d_model)`, `TemporalEmbedding(d_model, embed_type, freq)`
   9. `TimeFeatureEmbedding(d_model, embed_type, freq)`
   10. `DataEmbedding_wo_pos(c_in, d_model, embed_type, freq, dropout)`
4. Back in `layers/Autoformer_EncDec.py`
   11. `EncoderLayer`, `Encoder`
   12. `DecoderLayer`, `Decoder`
5. `models/Autoformer.py`
   13. `class Autoformer(nn.Module)` (a.k.a. `Model`) — assembly + `forward`
6. `utils/timefeatures.py`
   14. `TimeFeature` classes and `time_features(dates, freq)` returning `[d_inp, T]` (transpose to `[T, d_inp]`)
7. `data_provider/data_loader.py`
   15. `Dataset_ETT_hour`, `Dataset_ETT_minute`, `Dataset_Custom`, `Dataset_Pred` (+ `StandardScaler`)
8. `data_provider/data_factory.py`
   16. `data_provider(args, flag)` → `(dataset, DataLoader)`; `shuffle=True, drop_last=True` for train, `shuffle=False, drop_last=True` for val/test, `batch_size=1, drop_last=False` for pred
9. `utils/tools.py`
   17. `adjust_learning_rate`, `EarlyStopping`
10. `utils/metrics.py`
    18. `MAE, MSE, RMSE, MAPE, MSPE, metric`
11. `exp/exp_main.py`
    19. `train`, `vali`, `test`, `predict`, `_predict` (decoder-input construction)
12. `run.py`
    20. argparse with all defaults from the table above

Sanity test after step 5: feed random tensors `x_enc[2,96,7]`, `x_mark_enc[2,96,4]`, `x_dec[2,144,7]`, `x_mark_dec[2,144,4]` and assert the output is `[2, 96, 7]`; run once in `train()` mode and once in `eval()` mode (different aggregation paths).

---

### Common Pitfalls

1. **FFT must be over the time dimension after permuting to `[B, H, E, L]`.** `queries` arrive as `[B, L, H, E]`; you must `permute(0, 2, 3, 1)` *before* `torch.fft.rfft(..., dim=-1)`. Running `rfft` on the raw layout transforms the channel axis and silently produces garbage. Likewise `irfft(res, n=L, dim=-1)` — passing `n=L` is required, otherwise the output length is `2*(L//2+1)-2`, which is wrong for odd `L`.
2. **Top-k lag selection differs between train and eval.** Training uses lags from the correlation averaged over batch *and* heads *and* channels (one shared lag set, `torch.roll`); inference uses per-sample lags (`torch.topk` over `[B, L]` then `torch.gather`). Do not use one path for both — gate on `self.training` (the `nn.Module` flag, set by `model.train()` / `model.eval()`), and remember `model.eval()` before validation/test, otherwise your test numbers are computed with the training-mode aggregation.
3. **Padding V (and K) when `L_q > L_v`.** In the decoder's cross Auto-Correlation `L_q = L_label + L_pred` (144) but `L_v = L_enc` (96). Append `L_q - L_v` zero steps to **both** keys and values; if `L_q < L_v`, truncate to the first `L_q` steps. Forgetting this crashes (shape mismatch) or, worse, broadcasts wrongly.
4. **`gather` needs the value tensor doubled.** `time_delay_agg_inference` / `_full` build `tmp_values = values.repeat(1,1,1,2)` of length `2L` so that index `t + τ` (up to `2L-2`) stays in range and emulates circular wrap-around. Using the un-doubled tensor gives an index-out-of-bounds error. The index tensor must be `int64`.
5. **Replicate (edge) padding in `series_decomp`, with an odd kernel.** Pad `(kernel-1)//2` copies of the first and last step. With an **even** `kernel_size` the padded length is `L + kernel - 2`, so `AvgPool1d` outputs `L - 1` steps and every downstream residual `x - trend` breaks. Keep `moving_avg` odd (25), or pad asymmetrically if you must support even kernels. Never zero-pad — it biases the trend toward 0 at the boundaries.
6. **Trend init uses the encoder *mean*, not zeros.** `trend_init = concat(trend(x_enc)[:, -L_label:, :], mean(x_enc).repeat(1, L_pred, 1))`. Only the *seasonal* placeholder is zeros. Initializing the trend with zeros removes the level of the series and the model will not recover it (the decoder only adds *residual* trends).
7. **No positional embedding** — use `DataEmbedding_wo_pos` (token conv + temporal), not `DataEmbedding`. Adding positional encoding degrades results because it breaks the shift-equivariance Auto-Correlation exploits.
8. **`series_decomp` returns `(seasonal, trend)` in that order.** Swapping them silently trains a nonsense model. In the encoder the *second* return value is discarded; in the decoder the second is accumulated.
9. **Decoder trend projection is a kernel-3 circular `Conv1d(d_model → c_out, bias=False)`** applied to `trend1 + trend2 + trend3`, permuted to `[B, d_model, L]` first and transposed back. The seasonal branch uses a separate `nn.Linear(d_model, c_out, bias=True)` at the `Decoder` level. Don't reuse one for the other.
10. **Masks are inert.** `mask_flag` / `attn_mask` are accepted but unused by `AutoCorrelation`; there is no causal masking in Autoformer. Don't add one — the decoder is non-autoregressive and predicts all `L_pred` steps in one pass.
11. **`int(c * math.log(L))` uses natural log and truncates.** With `c=1, L=96` you get `k=4`. If `k` ends up `0` (very small `L` with `c<1`) `topk` returns nothing and the output is all zeros — guard with `max(1, ...)` if you support tiny windows.
12. **Slice the output last.** The decoder produces `L_label + L_pred` steps; the loss and metrics use only `[:, -L_pred:, :]`, and the target is `batch_y[:, -L_pred:, :]`. For `--features MS`, slice channels with `f_dim = -1`.
13. **Scaler leakage.** Fit `StandardScaler` on the train slice only, and note that val/test windows start `L_enc` steps *before* their border so the first window has real history.
14. **FFN convolutions are `bias=False`** kernel-1 `Conv1d`s (equivalent to bias-free linear layers), and dropout is applied *twice* (after activation and after the second conv).


---

## 3. FEDformer

FEDformer is a Transformer for long-term multivariate time-series forecasting that does all of its "attention" in the frequency domain. Instead of computing an L×L attention matrix, it takes an `rfft` along the time axis, keeps only a small **random subset of `modes` frequency components**, applies a learnable **complex-valued** linear map to those components, and inverse-transforms back to the time domain — which gives O(L) time and memory in the sequence length. On top of this it inherits Autoformer's **progressive seasonal-trend decomposition**: every sub-layer splits its activation into a moving-average trend and a seasonal residual; the seasonal part flows through the network and the trend parts are accumulated and added back at the end. Two interchangeable variants exist: **`version='Fourier'`** (Fourier basis, `FourierBlock` + `FourierCrossAttention`) and **`version='Wavelets'`** (multiwavelet basis with Legendre/Chebyshev polynomials, `MultiWaveletTransform` + `MultiWaveletCross`). Everything below is taken from the official reference implementation (github.com/MAZiqing/FEDformer) and the ICML 2022 paper; anything I could not confirm from those sources is tagged **[UNVERIFIED]**.

---

### Notation & Tensor Shapes

| Symbol | Repo name | Meaning | Default |
|---|---|---|---|
| `B` | — | batch size | 32 |
| `L_enc` | `seq_len` | encoder input length | 96 (ILI: 36) |
| `L_label` | `label_len` | decoder "start token" length, taken from the end of the encoder window | 48 (ILI: 18) |
| `L_pred` | `pred_len` | forecast horizon | 96 / 192 / 336 / 720 (ILI: 24/36/48/60) |
| `L_dec` | — | decoder sequence length = `L_label + L_pred`. **Note:** the model code computes decoder-side seq lens as `seq_len//2 + pred_len`, which equals `L_label + L_pred` only because `label_len == seq_len//2` in all published configs | 48+`L_pred` |
| `d_model` | `d_model` | model width | 512 |
| `n_heads` | `n_heads` | heads, `H` | 8 |
| `E` | — | per-head width = `d_model // n_heads` | 64 |
| `d_ff` | `d_ff` | FFN width | 2048 |
| `modes` | `modes` | number of frequency components kept | 64 |
| — | `mode_select` | `'random'` or `'low'` | `'random'` |
| `enc_in` | `enc_in` | encoder input channels (= #variates) | 7 for ETT |
| `dec_in` | `dec_in` | decoder input channels | = `enc_in` |
| `c_out` | `c_out` | output channels | = `enc_in` (M task), 1 (S task) |
| — | `moving_avg` | kernel size(s) for decomposition. `int` → `series_decomp`; `list` → `series_decomp_multi` | `[24]` in `run.py`; `25` is the Autoformer-style scalar default; `[12,24]` used in the model's `__main__` demo |
| `d_mark` | — | number of time features, from `freq`: `{'h':4,'t':5,'s':6,'m':1,'a':1,'w':2,'d':3,'b':3}` | 4 for hourly |

Canonical tensor layouts:

```text
x_enc        : [B, L_enc, enc_in]
x_mark_enc   : [B, L_enc, d_mark]
x_dec        : [B, L_label + L_pred, dec_in]
x_mark_dec   : [B, L_label + L_pred, d_mark]
enc_out      : [B, L_enc, d_model]
dec_out      : [B, L_label + L_pred, d_model]  -> projected to [B, ., c_out]
model output : [B, L_pred, c_out]
inside attention blocks q/k/v : [B, L, H, E]  (then permuted to [B, H, E, L] for FFT)
```

---

### Data Pipeline

#### CSV format

All benchmark CSVs share one schema: first column `date` (parseable timestamp), then one column per variate. The univariate target column is named `OT` in the ETT files (`--target OT`).

| Dataset | file | #variates (`enc_in`) | `freq` | split rule |
|---|---|---|---|---|
| ETTh1 / ETTh2 | `ETTh1.csv` | 7 | `h` | fixed month borders (below) |
| ETTm1 / ETTm2 | `ETTm1.csv` | 7 | `t` (15 min) | fixed month borders (below) |
| Electricity | `electricity.csv` | 321 | `h` | 70/10/20 |
| Traffic | `traffic.csv` | 862 | `h` | 70/10/20 |
| Weather | `weather.csv` | 21 | `t` (10 min) | 70/10/20 |
| Exchange | `exchange_rate.csv` | 8 | `d` | 70/10/20 |
| ILI (illness) | `national_illness.csv` | 7 | `w` | 70/10/20 |

**ETT-hour borders** (rows, 1 row = 1 hour):
```python
border1s = [0, 12*30*24 - seq_len, 12*30*24 + 4*30*24 - seq_len]
border2s = [12*30*24, 12*30*24 + 4*30*24, 12*30*24 + 8*30*24]
# = 12 months train / 4 months val / 4 months test
```
**ETT-minute borders** (same but ×4 because 15-min sampling):
```python
border1s = [0, 12*30*24*4 - seq_len, 12*30*24*4 + 4*30*24*4 - seq_len]
border2s = [12*30*24*4, 12*30*24*4 + 4*30*24*4, 12*30*24*4 + 8*30*24*4]
```
**Custom datasets (all others):**
```python
num_train = int(len(df) * 0.7); num_test = int(len(df) * 0.2)
num_vali  = len(df) - num_train - num_test
border1s = [0, num_train - seq_len, len(df) - num_test - seq_len]
border2s = [num_train, num_train + num_vali, len(df)]
```
The `- seq_len` offset makes val/test windows start early enough that their first sample has a full encoder context.

#### Standardization
`sklearn.preprocessing.StandardScaler` is **fit on the train slice only** (`df_data[border1s[0]:border2s[0]]`) and then applied to the whole series. Metrics are reported in standardized space (no inverse transform before MSE/MAE in `exp_main.test`).

#### Time features (`timeenc=1`, `embed='timeF'`)
Float features in `[-0.5, 0.5]`, produced by `utils/timefeatures.time_features(dates, freq)`:
- `h`: [month, day, weekday, hour] → 4 dims
- `t`: + minute-of-hour bucket → 5 dims
- `d`: 3 dims, `w`: 2 dims, `m`: 1 dim (see `freq_map` above)

Each raw field is scaled as e.g. `hour/23 - 0.5`, `weekday/6 - 0.5`, `(day-1)/30 - 0.5`, `(month-1)/11 - 0.5`, `minute//15 / 3 - 0.5`. **[UNVERIFIED]** exact per-field normalizers (read `utils/timefeatures.py` if exactness matters); any consistent scaling into ±0.5 works.

#### Windowing (`__getitem__`)
```python
s_begin = index;            s_end = s_begin + seq_len
r_begin = s_end - label_len; r_end = r_begin + label_len + pred_len
seq_x      = data_x[s_begin:s_end]        # [L_enc, C]
seq_y      = data_y[r_begin:r_end]        # [L_label + L_pred, C]
seq_x_mark = stamp[s_begin:s_end]         # [L_enc, d_mark]
seq_y_mark = stamp[r_begin:r_end]         # [L_label + L_pred, d_mark]
__len__    = len(data_x) - seq_len - pred_len + 1
```

#### Decoder input built in the training loop
```python
dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :])                  # [B, L_pred, C]
dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1)       # [B, L_label+L_pred, C]
```
So `x_dec` = (true label window) ‖ (zeros for the horizon). The targets are `batch_y[:, -pred_len:, f_dim:]` where `f_dim = -1` for the `MS` task else `0`.

#### Decoder input built **inside** the model (decomposition init)
The model ignores the numeric content of `x_dec` except its shape; it re-derives the seasonal/trend initialization from `x_enc`:
```python
mean  = x_enc.mean(dim=1, keepdim=True).repeat(1, pred_len, 1)          # [B, L_pred, C]
zeros = torch.zeros(B, pred_len, x_dec.shape[2], device=x_enc.device)   # [B, L_pred, C]
seasonal_init, trend_init = self.decomp(x_enc)   # series_decomp_multi(moving_avg) on x_enc; each [B, L_enc, C]

trend_init    = cat([trend_init[:, -label_len:, :], mean], dim=1)       # [B, L_label+L_pred, C]
seasonal_init = F.pad(seasonal_init[:, -label_len:, :], (0,0,0,pred_len))  # [B, L_label+L_pred, C]
```
`F.pad(..., (0,0,0,pred_len))` zero-pads the **time** dim on the right — identical to `cat([seasonal, zeros], 1)`.
Only `seasonal_init` is embedded and fed to the decoder stack; `trend_init` is passed separately as the running trend accumulator. `x_dec` itself is unused apart from `x_dec.shape[2]`; `x_mark_dec` **is** used (temporal embedding of the decoder positions).

---

### Module-by-Module Spec

#### 1. `moving_avg(kernel_size, stride=1)`
Trend extractor by 1-D average pooling with replicate padding so output length == input length.
```python
def forward(self, x):                    # x: [B, L, C]
    front = x[:, 0:1, :].repeat(1, kernel_size - 1 - ((kernel_size - 1) // 2), 1)
    end   = x[:, -1:, :].repeat(1, (kernel_size - 1) // 2, 1)
    x = torch.cat([front, x, end], dim=1)        # [B, L + kernel_size - 1, C]
    x = self.avg(x.permute(0, 2, 1))             # AvgPool1d(kernel_size, stride=1, padding=0) -> [B, C, L]
    return x.permute(0, 2, 1)                    # [B, L, C]
```
Note the asymmetric padding: `front` gets `k-1-floor((k-1)/2)` copies, `end` gets `floor((k-1)/2)`. For even `k` (e.g. 24) that is 12 front / 11 end; for odd `k=25` it is 12/12. Output length is exactly `L`.

#### 2. `series_decomp(kernel_size)`
```python
def forward(self, x):                   # [B, L, C]
    moving_mean = self.moving_avg(x)    # trend    [B, L, C]
    res = x - moving_mean               # seasonal [B, L, C]
    return res, moving_mean             # (seasonal, trend)
```

#### 3. `series_decomp_multi(kernel_size: list)`
Several moving averages combined by a **learnable, input-dependent softmax gate**.
```python
def __init__(self, kernel_size):        # e.g. [24] or [12, 24]
    self.moving_avg = [moving_avg(k, stride=1) for k in kernel_size]   # (repo uses a plain list)
    self.layer = nn.Linear(1, len(kernel_size))

def forward(self, x):                                       # x: [B, L, C]
    means = [f(x).unsqueeze(-1) for f in self.moving_avg]   # each [B, L, C, 1]
    moving_mean = torch.cat(means, dim=-1)                  # [B, L, C, K]
    w = nn.Softmax(-1)(self.layer(x.unsqueeze(-1)))         # x.unsqueeze(-1): [B,L,C,1] -> [B,L,C,K]
    moving_mean = torch.sum(moving_mean * w, dim=-1)        # [B, L, C]
    res = x - moving_mean
    return res, moving_mean
```
So the K trends are combined with per-(batch, time, channel) weights that sum to 1, produced by a 1→K linear layer applied to the scalar value at that position. With `K = 1` the softmax weight is identically 1 and this reduces to plain `series_decomp` (but still creates the useless `Linear(1,1)`).

**Implementation warning:** the repo stores `self.moving_avg` as a Python list, so those submodules are not registered. They contain no parameters (`AvgPool1d`), so training still works, but `.to(device)` is a no-op for them — fine. Prefer `nn.ModuleList` in a clean implementation.

#### 4. `get_frequency_modes(seq_len, modes=64, mode_select_method='random')`
```python
def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    modes = min(modes, seq_len // 2)
    if mode_select_method == 'random':
        index = list(range(0, seq_len // 2))
        np.random.shuffle(index)
        index = index[:modes]
    else:                                  # 'low'
        index = list(range(0, modes))
    index.sort()
    return index                           # sorted list of ints, len == modes, values in [0, seq_len//2)
```
Returned value is a **sorted list of rfft bin indices**. `rfft` of a length-`L` signal has `L//2 + 1` bins; the candidate pool is only `range(L//2)` (bin `L//2` itself is never selected). Call this **once in `__init__`** and store `self.index` — the selection must be frozen for the lifetime of the model (train and test must use the same modes).

Examples: `seq_len=96, modes=64` → `modes = min(64,48) = 48`, so with `'random'` **all** 48 bins are selected (`index == [0..47]`). For `seq_len=96+ pred_len=720` decoder side, `seq_len//2+pred_len = 768`, `768//2 = 384 > 64`, so 64 of 384 bins are randomly chosen.

#### 5. `FourierBlock` — Frequency Enhanced Block (FEB-f), used as encoder & decoder **self**-attention
```python
class FourierBlock(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, modes=0, mode_select_method='random'):
        self.index = get_frequency_modes(seq_len, modes, mode_select_method)
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(
            8, in_channels // 8, out_channels // 8, len(self.index), dtype=torch.cfloat))
        # shape [H=8, E_in=d_model//8, E_out=d_model//8, M]
```
`in_channels = out_channels = d_model`. The literal `8` is the hard-coded head count; it must equal `n_heads`. Initialization is `U[0,1)` (independently for real and imaginary parts of the cfloat tensor) times `1/(d_model*d_model)`.

```python
def forward(self, q, k, v, mask):
    B, L, H, E = q.shape                 # [B, L, 8, d_model//8]
    x = q.permute(0, 2, 3, 1)            # [B, H, E, L]   (k and v are IGNORED)
    x_ft = torch.fft.rfft(x, dim=-1)     # [B, H, E, L//2+1], complex
    out_ft = torch.zeros(B, H, E, L//2+1, device=x.device, dtype=torch.cfloat)
    for wi, i in enumerate(self.index):
        # per selected bin: complex matmul over the channel dim, per head
        out_ft[:, :, :, wi] = torch.einsum("bhi,hio->bho",
                                           x_ft[:, :, :, i],          # [B,H,E]
                                           self.weights1[:, :, :, wi])# [H,E,E]
    x = torch.fft.irfft(out_ft, n=x.size(-1))   # [B, H, E, L]
    return (x, None)
```
Math: for each head `h` and each selected frequency `f_m`, `Y[:,h,:,m] = W_h[:,:,m]^T X[:,h,:,f_m]` with complex `W`. Note the **repo quirk**: the result of bin `index[wi]` is written to output bin `wi`, not back to `index[wi]`. I.e. the selected modes are *compacted* to the lowest `M` bins of the output spectrum. This is what the released code does; reproduce it verbatim for fidelity. (Scattering back to `index[wi]` would be the "expected" behaviour — **[UNVERIFIED]** whether this was intentional. `FourierCrossAttention` *does* scatter back to `index_q[i]`.)

`AutoCorrelationLayer` returns `out.view(B, L, -1)` from a `[B, L, H, E]` tensor, so `FourierBlock` must return `[B, H, E, L]`-shaped data ... and it does **not** permute back. This is another repo quirk: the returned tensor `[B, H, E, L]` is reshaped by `out.view(B, L, -1)` in `AutoCorrelationLayer`, which requires `H*E*L == L*d_model` (true) but scrambles the axis semantics. Keep it as-is to match released weights/results; if you want the "clean" version, `return x.permute(0, 3, 1, 2)`. **[UNVERIFIED]** which one the reported numbers used — the released code path is the one shown above.

##### Complex-tensor handling
`torch.cfloat` parameters + `torch.einsum` on complex tensors require PyTorch ≥ 1.8 (repo `requirements.txt` targets `torch==1.9.x`). If you must support older versions, store two real parameters and do the multiply manually:
```python
# w_r, w_i : real nn.Parameters of shape [H, E, E, M]
out_r = einsum("bhi,hio->bho", x_r, w_r) - einsum("bhi,hio->bho", x_i, w_i)
out_i = einsum("bhi,hio->bho", x_r, w_i) + einsum("bhi,hio->bho", x_i, w_r)
```
Also note: complex parameters are *not* supported by some optimizers/AMP paths; `Adam` handles them in ≥1.9, and `--use_amp` should be left off.

#### 6. `FourierCrossAttention` — Frequency Enhanced Attention (FEA-f), decoder **cross**-attention
```python
class FourierCrossAttention(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv,
                 modes=64, mode_select_method='random', activation='tanh', policy=0):
        self.index_q  = get_frequency_modes(seq_len_q,  modes, mode_select_method)   # M_q
        self.index_kv = get_frequency_modes(seq_len_kv, modes, mode_select_method)   # M_kv
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(
            8, in_channels // 8, out_channels // 8, len(self.index_q), dtype=torch.cfloat))
```
`seq_len_q = seq_len//2 + pred_len` (decoder length), `seq_len_kv = seq_len` (encoder length). `M_q` and `M_kv` can differ (`min(modes, len//2)` each).

```python
def forward(self, q, k, v, mask):
    B, L, H, E = q.shape
    xq = q.permute(0, 2, 3, 1); xk = k.permute(0, 2, 3, 1); xv = v.permute(0, 2, 3, 1)  # [B,H,E,L*]

    xq_ft = torch.fft.rfft(xq, dim=-1)                 # [B,H,E,L_q//2+1]
    xq_ft_ = zeros(B, H, E, M_q, dtype=cfloat)
    for i, j in enumerate(self.index_q):  xq_ft_[..., i] = xq_ft[..., j]

    xk_ft = torch.fft.rfft(xk, dim=-1)                 # [B,H,E,L_kv//2+1]
    xk_ft_ = zeros(B, H, E, M_kv, dtype=cfloat)
    for i, j in enumerate(self.index_kv): xk_ft_[..., i] = xk_ft[..., j]

    xqk_ft = torch.einsum("bhex,bhey->bhxy", xq_ft_, xk_ft_)   # [B,H,M_q,M_kv]  (contract channel dim)
    if activation == 'tanh':
        xqk_ft = xqk_ft.tanh()                                  # complex tanh, elementwise  <-- REPO DEFAULT
    elif activation == 'softmax':
        xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)             # softmax over M_kv on magnitudes
        xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))

    xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xk_ft_)   # [B,H,E,M_q]   NOTE: xk_ft_, not xv!
    xqkvw   = torch.einsum("bhex,heox->bhox", xqkv_ft, self.weights1)  # [B,H,E,M_q]
    out_ft  = zeros(B, H, E, L // 2 + 1, dtype=cfloat)          # L == L_q here
    for i, j in enumerate(self.index_q): out_ft[..., j] = xqkvw[..., i]   # scatter BACK to original bins
    out = torch.fft.irfft(out_ft / self.in_channels / self.out_channels, n=xq.size(-1))  # [B,H,E,L_q]
    return (out, None)
```
Key facts to reproduce exactly:
- **Default activation is `'tanh'`** (`--cross_activation tanh`); `softmax` is the documented alternative.
- The "value" tensor `xv` is computed but **never used** — the second einsum uses `xk_ft_` again. This is in the released code; keep it (or note the deviation if you "fix" it). **[UNVERIFIED]** as intentional.
- Weight einsum subscript order is `"bhex,heox->bhox"` (weights indexed `[h, e_in, e_out, m]`), different from `FourierBlock`'s `"bhi,hio->bho"` per-mode loop.
- **Normalization factor** = `in_channels * out_channels` = `d_model * d_model` = `512*512 = 262144`, applied to the spectrum *before* `irfft`.
- `irfft(..., n=xq.size(-1))` is mandatory to get length `L_q` back (otherwise you get `2*(L_q//2+1)-2`, which differs for odd `L_q`).
- Same `[B,H,E,L]` vs `[B,L,H,E]` return-layout quirk as `FourierBlock`.

#### 7. `AutoCorrelationLayer` (the multi-head wrapper both blocks live inside)
```python
class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads, d_keys=None, d_values=None):
        d_keys = d_keys or d_model // n_heads; d_values = d_values or d_model // n_heads
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection   = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection   = nn.Linear(d_values * n_heads, d_model)

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape; _, S, _ = keys.shape; H = self.n_heads
        q = self.query_projection(queries).view(B, L, H, -1)   # [B,L,H,E]
        k = self.key_projection(keys).view(B, S, H, -1)
        v = self.value_projection(values).view(B, S, H, -1)
        out, attn = self.inner_correlation(q, k, v, attn_mask)
        out = out.view(B, L, -1)                               # [B,L,d_model]
        return self.out_projection(out), attn
```
`attn_mask` is always `None` for FEDformer (no causal masking; the decoder is non-autoregressive).

#### 8. `MultiWaveletTransform` / `MultiWaveletCross` — OPTIONAL ADVANCED SECTION (`version='Wavelets'`)

Implement the Fourier version first; this section is self-contained and can be skipped. Extra dependencies: `sympy` (for the polynomial filters) and `einops`.

**Filter construction — `layers/utils.get_filter(base, k)`** returns six `k×k` numpy matrices `H0, H1, G0, G1, PHI0, PHI1` for `base ∈ {'legendre','chebyshev'}`. For Legendre with the orthonormal shifted-Legendre basis `φ_i` on [0,1] and multiwavelets `ψ_i`, using the `k`-point Gauss–Legendre nodes `x_m` (roots of `P_k(2x-1)`) and weights `w_m = 1/(k · P_k'(2x_m-1) · P_{k-1}(2x_m-1))`:
```
H0[i,j] = 1/sqrt(2) * Σ_m w_m φ_i(x_m/2)     φ_j(x_m)
G0[i,j] = 1/sqrt(2) * Σ_m w_m ψ_i(x_m/2)     φ_j(x_m)
H1[i,j] = 1/sqrt(2) * Σ_m w_m φ_i((x_m+1)/2) φ_j(x_m)
G1[i,j] = 1/sqrt(2) * Σ_m w_m ψ_i((x_m+1)/2) φ_j(x_m)
PHI0 = PHI1 = I_k          (legendre)
```
where `ψ_i(x) = ψ1_i(x)·1[x≤0.5] + ψ2_i(x)·1[x>0.5]`. For `chebyshev`, `2k` Chebyshev nodes with uniform weights `π/(2k)` are used and `PHI0/PHI1` are Gram matrices of `φ` at the half-interval nodes (not identity). Then
```
H0r = H0 @ PHI0 ; G0r = G0 @ PHI0 ; H1r = H1 @ PHI1 ; G1r = G1 @ PHI1   (entries |·|<1e-8 zeroed)
ec_s = concat([H0.T, H1.T], axis=0)   # [2k, k]  decomposition, smooth
ec_d = concat([G0.T, G1.T], axis=0)   # [2k, k]  decomposition, detail
rc_e = concat([H0r, G0r], axis=0)     # [2k, k]  reconstruction, even positions
rc_o = concat([H1r, G1r], axis=0)     # [2k, k]  reconstruction, odd positions
```
All four are `register_buffer`s (non-trainable).

**One decomposition step / reconstruction step** (tensors `[B, N, c, k]`):
```python
def wavelet_transform(self, x):
    xa = torch.cat([x[:, ::2], x[:, 1::2]], dim=-1)   # [B, N/2, c, 2k]
    d = xa @ self.ec_d                                # detail  [B, N/2, c, k]
    s = xa @ self.ec_s                                # smooth  [B, N/2, c, k]
    return d, s

def evenOdd(self, x):                                 # x: [B, N, c, 2k]
    x_e = x @ self.rc_e; x_o = x @ self.rc_o          # each [B, N, c, k]
    out = zeros(B, 2N, c, k); out[:, ::2] = x_e; out[:, 1::2] = x_o
    return out                                        # [B, 2N, c, k]
```

**`MultiWaveletTransform(ich=d_model, k=8, alpha=16, c=128, nCZ=1, L=0, base='legendre')`** (self-attention replacement):
```python
forward(q, k, v, mask):            # q: [B,L,H,E], v: [B,S,H,D]
  pad/crop k,v along time to length L   # zeros if L>S else v[:, :L]
  values = v.view(B, L, -1)             # [B, L, d_model]
  V = self.Lk0(values).view(B, L, c, k) # Linear(ich, c*k)
  for i in range(nCZ): V = MWT_CZ1d(...)(V); if i < nCZ-1: V = relu(V)
  V = self.Lk1(V.view(B, L, -1))        # Linear(c*k, ich) -> [B,L,d_model]
  return V.view(B, L, -1, D), None      # [B,L,H,E]
```
`MWT_CZ1d(k, alpha, L, c, base)`: with `ns = floor(log2(N))`, pad `N` up to `nl = 2**ceil(log2(N))` by wrapping the first `nl-N` steps; then for `i in range(ns - L)`: `d, x = wavelet_transform(x)`, `Ud[i] = A(d) + B(x)`, `Us[i] = C(d)`; then `x = T0(x)` (`nn.Linear(k,k)`, coarsest scale); then reconstruct for `i` from `ns-1-L` down to `0`: `x = x + Us[i]; x = cat([x, Ud[i]], -1); x = evenOdd(x)`; finally crop `x[:, :N]`. `A`, `B`, `C` are `sparseKernelFT1d(k, alpha, c)`: flatten `[B,N,c,k]→[B,c*k,N]`, `rfft`, multiply the lowest `l = min(alpha, N//2+1)` bins by a complex weight `[c*k, c*k, alpha]` via `einsum("bix,iox->box")` with scale `1/(c*k*c*k)`, zero the rest, `irfft(n=N)`.

**`MultiWaveletCross(in_channels, out_channels, seq_len_q, seq_len_kv, modes, c=64, k=8, ich=512, L=0, base='legendre', activation='tanh')`** (cross-attention replacement): projects q/k/v with `Lq/Lk/Lv: Linear(ich, c*k)` → `[B,N,c,k]`, pads to power of two, runs `ns - L` wavelet decomposition steps on each of q/k/v, then at each scale applies `FourierCrossAttentionW` — a weight-free version of `FourierCrossAttention` that always takes the **lowest** `min(L//2, modes)` bins and divides by `in_channels*out_channels` — `attn1` on the detail-detail pair, `attn2` on the smooth part of the detail tuple, `attn3` on the smooth-smooth pair, and `attn4` on the coarsest signals; then reconstructs with `evenOdd` and projects back with `out: Linear(c*k, ich)`.

Wavelet hyperparameters in `run.py`: `--L 3` (CLI default; the class defaults are `L=0`), `--base legendre`, `--cross_activation tanh`, `k=8`, `c=64` for cross / `c=128` for the self transform. The paper's ablations report `L=0..3`. **[UNVERIFIED]** which `L` produced the published Wavelet numbers (CLI says 3, class default and the model's demo config use 0/1).

#### 9. `DataEmbedding_wo_pos(c_in, d_model, embed_type='timeF', freq='h', dropout=0.05)`
```python
value_embedding    = TokenEmbedding(c_in, d_model)      # Conv1d(c_in, d_model, k=3, padding=1,
                                                        #        padding_mode='circular', bias=False)
                                                        # kaiming_normal_(fan_in, leaky_relu) init
temporal_embedding = TimeFeatureEmbedding(d_model, freq) if embed_type == 'timeF'
                     else TemporalEmbedding(d_model, embed_type, freq)
dropout            = nn.Dropout(0.05)

def forward(self, x, x_mark):                 # x: [B,L,c_in], x_mark: [B,L,d_mark]
    x = self.value_embedding(x) + self.temporal_embedding(x_mark)   # [B,L,d_model]
    return self.dropout(x)
```
- `TokenEmbedding`: `x.permute(0,2,1)` → Conv1d → `.transpose(1,2)`.
- `TimeFeatureEmbedding` = `nn.Linear(d_mark, d_model, bias=False)`.
- `TemporalEmbedding` (for `embed='fixed'/'learned'`) = sum of embeddings of month(13)/day(32)/weekday(7)/hour(24)[/minute(4) if `freq=='t'`], each `→ d_model`.
- **No positional encoding is added** (the class still constructs a `PositionalEmbedding`, but never uses it — harmless dead weight; you may omit it).

#### 10. `my_Layernorm(channels)` — final norm for the seasonal branch
```python
def forward(self, x):                           # [B,L,d_model]
    x_hat = self.layernorm(x)                   # nn.LayerNorm(d_model)
    bias  = x_hat.mean(dim=1, keepdim=True).repeat(1, x.shape[1], 1)
    return x_hat - bias                         # removes the per-channel temporal mean
```

#### 11. `EncoderLayer(attention, d_model, d_ff, moving_avg, dropout, activation='gelu')`
```python
conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1, bias=False)
conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1, bias=False)
# NOTE: decomp1 and decomp2 must be TWO SEPARATE instances -- series_decomp_multi owns a
# learnable Linear(1, K) gate, so `decomp1 = decomp2 = ...` would tie their parameters.
_mk = (lambda: series_decomp_multi(moving_avg)) if isinstance(moving_avg, list) \
      else (lambda: series_decomp(moving_avg))
decomp1 = _mk(); decomp2 = _mk()
act = F.gelu if activation != 'relu' else F.relu

def forward(self, x, attn_mask=None):           # x: [B, L_enc, d_model]
    new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)  # FourierBlock via AutoCorrelationLayer
    x = x + self.dropout(new_x)                                 # residual
    x, _ = self.decomp1(x)                                      # keep seasonal, DISCARD trend
    y = x
    y = self.dropout(self.act(self.conv1(y.transpose(-1, 1))))  # [B, d_ff, L]
    y = self.dropout(self.conv2(y).transpose(-1, 1))            # [B, L, d_model]
    res, _ = self.decomp2(x + y)                                # keep seasonal, DISCARD trend
    return res, attn
```
The encoder throws both trends away — only the decoder accumulates trend.

#### 12. `DecoderLayer(self_attention, cross_attention, d_model, c_out, d_ff, moving_avg, dropout, activation='gelu')`
```python
conv1 = Conv1d(d_model, d_ff, 1, bias=False); conv2 = Conv1d(d_ff, d_model, 1, bias=False)
decomp1, decomp2, decomp3 = _mk(), _mk(), _mk()   # three SEPARATE instances (see EncoderLayer note)
projection = nn.Conv1d(d_model, c_out, kernel_size=3, stride=1, padding=1,
                       padding_mode='circular', bias=False)

def forward(self, x, cross, x_mask=None, cross_mask=None):
    # x: [B, L_dec, d_model]; cross (encoder output): [B, L_enc, d_model]
    x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
    x, trend1 = self.decomp1(x)
    x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
    x, trend2 = self.decomp2(x)
    y = self.dropout(self.act(self.conv1(x.transpose(-1, 1))))
    y = self.dropout(self.conv2(y).transpose(-1, 1))
    x, trend3 = self.decomp3(x + y)
    residual_trend = trend1 + trend2 + trend3                       # [B, L_dec, d_model]
    residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)  # [B, L_dec, c_out]
    return x, residual_trend                                        # seasonal, trend increment
```

#### 13. `Encoder(attn_layers, conv_layers=None, norm_layer=my_Layernorm(d_model))`
```python
def forward(self, x, attn_mask=None):
    attns = []
    for layer in self.attn_layers:            # e_layers = 2
        x, attn = layer(x, attn_mask=attn_mask); attns.append(attn)
    if self.norm is not None: x = self.norm(x)
    return x, attns                           # x: [B, L_enc, d_model]
```

#### 14. `Decoder(layers, norm_layer=my_Layernorm(d_model), projection=nn.Linear(d_model, c_out, bias=True))`
```python
def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
    # x: [B, L_dec, d_model]; trend: [B, L_dec, c_out] == trend_init
    for layer in self.layers:                 # d_layers = 1
        x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
        trend = trend + residual_trend        # trend accumulation, [B, L_dec, c_out]
    if self.norm is not None:       x = self.norm(x)          # my_Layernorm
    if self.projection is not None: x = self.projection(x)    # [B, L_dec, c_out]
    return x, trend
```
There are therefore **two** output projections: a per-layer `Conv1d(d_model→c_out, k=3, circular, bias=False)` for the trend increments, and a final `Linear(d_model→c_out, bias=True)` for the seasonal part.

---

### Full Model Assembly

```python
class FEDformer(nn.Module):                       # named `Model` in the repo
    def __init__(self, configs):
        super().__init__()
        self.version   = configs.version          # 'Fourier' | 'Wavelets'
        self.mode_select, self.modes = configs.mode_select, configs.modes
        self.seq_len, self.label_len, self.pred_len = configs.seq_len, configs.label_len, configs.pred_len
        self.output_attention = configs.output_attention

        # --- decomposition used for decoder init
        ks = configs.moving_avg
        self.decomp = series_decomp_multi(ks) if isinstance(ks, list) else series_decomp(ks)

        # --- embeddings (no positional encoding)
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model,
                                                  configs.embed, configs.freq, configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model,
                                                  configs.embed, configs.freq, configs.dropout)

        # --- the three frequency blocks
        if configs.version == 'Wavelets':
            encoder_self_att  = MultiWaveletTransform(ich=configs.d_model, L=configs.L, base=configs.base)
            decoder_self_att  = MultiWaveletTransform(ich=configs.d_model, L=configs.L, base=configs.base)
            decoder_cross_att = MultiWaveletCross(in_channels=configs.d_model, out_channels=configs.d_model,
                                                  seq_len_q=self.seq_len // 2 + self.pred_len,
                                                  seq_len_kv=self.seq_len, modes=configs.modes,
                                                  ich=configs.d_model, base=configs.base,
                                                  activation=configs.cross_activation)
        else:  # 'Fourier'
            encoder_self_att  = FourierBlock(in_channels=configs.d_model, out_channels=configs.d_model,
                                             seq_len=self.seq_len, modes=configs.modes,
                                             mode_select_method=configs.mode_select)
            decoder_self_att  = FourierBlock(in_channels=configs.d_model, out_channels=configs.d_model,
                                             seq_len=self.seq_len // 2 + self.pred_len, modes=configs.modes,
                                             mode_select_method=configs.mode_select)
            decoder_cross_att = FourierCrossAttention(in_channels=configs.d_model, out_channels=configs.d_model,
                                                      seq_len_q=self.seq_len // 2 + self.pred_len,
                                                      seq_len_kv=self.seq_len, modes=configs.modes,
                                                      mode_select_method=configs.mode_select)

        self.encoder = Encoder(
            [EncoderLayer(AutoCorrelationLayer(encoder_self_att, configs.d_model, configs.n_heads),
                          configs.d_model, configs.d_ff, moving_avg=configs.moving_avg,
                          dropout=configs.dropout, activation=configs.activation)
             for _ in range(configs.e_layers)],
            norm_layer=my_Layernorm(configs.d_model))

        self.decoder = Decoder(
            [DecoderLayer(AutoCorrelationLayer(decoder_self_att, configs.d_model, configs.n_heads),
                          AutoCorrelationLayer(decoder_cross_att, configs.d_model, configs.n_heads),
                          configs.d_model, configs.c_out, configs.d_ff, moving_avg=configs.moving_avg,
                          dropout=configs.dropout, activation=configs.activation)
             for _ in range(configs.d_layers)],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # x_enc      [B, L_enc, enc_in]      x_mark_enc [B, L_enc, d_mark]
        # x_dec      [B, L_dec, dec_in]      x_mark_dec [B, L_dec, d_mark]   (L_dec = L_label + L_pred)
        mean  = x_enc.mean(1).unsqueeze(1).repeat(1, self.pred_len, 1)            # [B, L_pred, enc_in]
        zeros = torch.zeros(x_dec.shape[0], self.pred_len, x_dec.shape[2],
                            device=x_enc.device)                                  # [B, L_pred, dec_in] (unused)
        seasonal_init, trend_init = self.decomp(x_enc)                            # [B, L_enc, enc_in] each

        trend_init    = torch.cat([trend_init[:, -self.label_len:, :], mean], 1)  # [B, L_dec, enc_in]
        seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :],
                              (0, 0, 0, self.pred_len))                           # [B, L_dec, enc_in]

        enc_out = self.enc_embedding(x_enc, x_mark_enc)                           # [B, L_enc, d_model]
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)           # [B, L_enc, d_model]

        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)                   # [B, L_dec, d_model]
        seasonal_part, trend_part = self.decoder(dec_out, enc_out,
                                                 x_mask=dec_self_mask,
                                                 cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # seasonal_part [B, L_dec, c_out]   trend_part [B, L_dec, c_out]
        dec_out = trend_part + seasonal_part                                      # [B, L_dec, c_out]
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        return dec_out[:, -self.pred_len:, :]                                     # [B, L_pred, c_out]
```
Note `trend_init` has `enc_in` channels while `residual_trend` has `c_out`; they are added together, so `enc_in == c_out` is required (true for `M` and `S` tasks; the `MS` task with `enc_in≠c_out` would break here).

---

### Default Hyperparameters

| Arg | Default | Notes |
|---|---|---|
| `model` | `FEDformer` | |
| `version` | `Fourier` | or `Wavelets` |
| `mode_select` | `random` | or `low` |
| `modes` | `64` | effective = `min(modes, len//2)` per block |
| `L` | `3` (CLI) / `0` (class) | wavelet decomposition level, ignored for Fourier |
| `base` | `legendre` | or `chebyshev` |
| `cross_activation` | `tanh` | or `softmax` |
| `seq_len` | `96` | ILI: `36` |
| `label_len` | `48` | ILI: `18` |
| `pred_len` | `96` | sweep `{96,192,336,720}`; ILI `{24,36,48,60}` |
| `enc_in`/`dec_in`/`c_out` | `7` | 321 ECL, 862 Traffic, 21 Weather, 8 Exchange, 7 ILI; `1/1/1` for univariate `S` |
| `d_model` | `512` | |
| `n_heads` | `8` | must stay 8: hard-coded in the complex weight shape |
| `e_layers` | `2` | |
| `d_layers` | `1` | |
| `d_ff` | `2048` | |
| `moving_avg` | `[24]` | list → `series_decomp_multi`; `[12,24]` and scalar `25` also appear |
| `factor` | `1` (CLI), `3` in scripts | unused by FEDformer |
| `dropout` | `0.05` | |
| `embed` | `timeF` | |
| `activation` | `gelu` | |
| `features` | `M` | `M` multivariate→multivariate, `S` uni→uni, `MS` multi→uni |
| `freq` | `h` | |
| optimizer | Adam, `lr=1e-4` | |
| `lradj` | `type1` | `lr = 1e-4 * 0.5**(epoch-1)`, applied at the end of each epoch |
| `batch_size` | `32` | |
| `train_epochs` | `10` | |
| `patience` | `3` | early stopping on val loss |
| `loss` | `mse` | `nn.MSELoss` |
| `itr` | `3` | 3 runs, results averaged |
| seed | `2021` | `random`, `numpy`, `torch` all seeded in `run.py` |
| `use_amp` | `False` | keep off — complex ops + AMP don't mix |

---

### Training Loop

```python
train_loader, vali_loader, test_loader = get_data('train'), get_data('val'), get_data('test')
# DataLoader: train shuffle=True drop_last=True; val/test shuffle=False drop_last=True
optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
criterion = nn.MSELoss()
early_stopping = EarlyStopping(patience=args.patience)

for epoch in range(args.train_epochs):
    model.train()
    for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
        optim.zero_grad()
        batch_x, batch_y = batch_x.float().to(dev), batch_y.float().to(dev)
        batch_x_mark, batch_y_mark = batch_x_mark.float().to(dev), batch_y_mark.float().to(dev)

        dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :])
        dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(dev)

        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)   # [B, L_pred, c_out]
        f_dim = -1 if args.features == 'MS' else 0
        target = batch_y[:, -pred_len:, f_dim:]
        loss = criterion(outputs, target)
        loss.backward()
        optim.step()

    vali_loss = evaluate(vali_loader); test_loss = evaluate(test_loader)
    early_stopping(vali_loss, model, checkpoint_path)   # saves 'checkpoint.pth' on improvement
    if early_stopping.early_stop: break
    adjust_learning_rate(optim, epoch + 1, args)        # type1: lr *= 0.5 each epoch

model.load_state_dict(torch.load(checkpoint_path + '/checkpoint.pth'))
```
`EarlyStopping`: tracks `best_score = -val_loss`; if `score < best_score + delta(=0)` increment counter, stop at `patience`; otherwise save checkpoint. `evaluate` runs the identical forward under `torch.no_grad()` with `model.eval()` and averages `criterion(pred.cpu(), true.cpu())`.

No gradient clipping, no warmup, no weight decay.

---

### Evaluation

Metrics over the standardized test set (`utils/metrics.py`):
```python
MAE  = np.mean(np.abs(pred - true))
MSE  = np.mean((pred - true) ** 2)
RMSE = sqrt(MSE)
MAPE = np.mean(np.abs((pred - true) / true))
MSPE = np.mean(((pred - true) / true) ** 2)
```
Predictions are collected as arrays of shape `[N_windows, L_pred, c_out]` and reshaped to `[-1, L_pred, c_out]` before metric computation. Reported values are **MSE and MAE** only, averaged over `itr=3` runs.

Two protocols:
- **Multivariate** (`--features M`): `enc_in = dec_in = c_out = #variates`, metrics over all channels.
- **Univariate** (`--features S`, `--target OT`): `enc_in = dec_in = c_out = 1`.

Datasets: ETTm1, ETTm2, ETTh1, ETTh2, Electricity, Traffic, Weather, Exchange-rate, ILI. Horizons `{96,192,336,720}` (ILI `{24,36,48,60}`), all with `L_enc = 96` (ILI 36).

Headline paper claims: FEDformer reduces error vs. the previous SOTA (Autoformer) by **14.8%** (multivariate) and **22.6%** (univariate) averaged over the benchmark.

Representative multivariate MSE/MAE, `L_enc=96` (FEDformer-f) — **[UNVERIFIED]** exact digits, I could not re-read the results tables from the PDF; treat as a sanity band, not as ground truth:

| Dataset | H=96 | H=192 | H=336 | H=720 |
|---|---|---|---|---|
| ETTm2 | ≈0.20 | ≈0.27 | ≈0.33 | ≈0.43 |
| Electricity | ≈0.19 | ≈0.20 | ≈0.21 | ≈0.24 |
| Weather | ≈0.22 | ≈0.28 | ≈0.34 | ≈0.42 |
| Exchange | ≈0.15 | ≈0.27 | ≈0.46 | ≈1.2 |

If your ETTm2/H=96 multivariate MSE is in the 0.19–0.22 range, the implementation is almost certainly correct; if it is >0.4 something is wrong.

---

### Implementation Checklist

Implement in this order; each step is testable in isolation.

1. `utils/timefeatures.py` — `time_features(dates, freq)` → `[d_mark, T]`.
2. `utils/metrics.py` — `metric(pred, true)` → `(mae, mse, rmse, mape, mspe)`.
3. `utils/tools.py` — `EarlyStopping`, `adjust_learning_rate`, `StandardScaler` helpers.
4. `data_provider/data_loader.py` — `Dataset_ETT_hour`, `Dataset_ETT_minute`, `Dataset_Custom`; `data_provider/data_factory.py` — `data_provider(args, flag)` returning `(dataset, DataLoader)`.
5. `layers/Embed.py` — `TokenEmbedding`, `TimeFeatureEmbedding`, `TemporalEmbedding`, `PositionalEmbedding`, `DataEmbedding_wo_pos`.
6. `layers/Autoformer_EncDec.py` — `moving_avg`, `series_decomp`, `series_decomp_multi`, `my_Layernorm`, `EncoderLayer`, `Encoder`, `DecoderLayer`, `Decoder`.
7. `layers/AutoCorrelation.py` — `AutoCorrelationLayer` (the multi-head wrapper) only; the `AutoCorrelation` block itself is not needed for FEDformer.
8. `layers/FourierCorrelation.py` — `get_frequency_modes`, `FourierBlock`, `FourierCrossAttention`.
9. `models/FEDformer.py` — `Model` (the assembly above). **Smoke-test now** with the `__main__` config: `B=3, seq_len=96, label_len=48, pred_len=96, enc_in=7, d_model=16, d_ff=16` → output `[3, 96, 7]`.
10. *(optional)* `layers/utils.py` — `get_phi_psi`, `get_filter`, Legendre/Chebyshev helpers; `layers/MultiWaveletCorrelation.py` — `sparseKernelFT1d`, `MWT_CZ1d`, `MultiWaveletTransform`, `FourierCrossAttentionW`, `MultiWaveletCross`.
11. `exp/exp_basic.py`, `exp/exp_main.py` — `Exp_Main` with `train/vali/test/predict`.
12. `run.py` — argparse per the hyperparameter table + seeding.
13. `scripts/run_M.sh`, `scripts/run_S.sh` — the dataset × horizon sweeps.

Suggested unit tests:
- `moving_avg` preserves length for kernels 12, 24, 25 and arbitrary `L`.
- `series_decomp`: `seasonal + trend == x` (within fp tolerance).
- `series_decomp_multi`: softmax weights sum to 1 along the last dim.
- `get_frequency_modes(96, 64, 'random')` returns 48 sorted unique indices in `[0,48)`.
- `FourierBlock` output shape and `irfft` length exactness for odd/even `L`.
- Two forward passes of the same model give identical mode indices (`model.encoder.attn_layers[0].attention.inner_correlation.index`).
- Full model forward with `pred_len ∈ {96,192,336,720}` and `L_dec` odd/even.

---

### Common Pitfalls

1. **Re-sampling the random modes.** `get_frequency_modes` must be called exactly **once, in `__init__`**, and the result stored (`self.index`). If you call it inside `forward`, the learned complex weights get paired with different frequency bins every step and the model will not learn (and test-time behaviour becomes non-deterministic). Also seed before construction if you want reproducibility.
2. **Complex tensors.** `dtype=torch.cfloat` parameters, complex `einsum`, and `.tanh()` on complex tensors need PyTorch ≥ 1.8/1.9. On older versions use the real/imaginary parameter-pair formulation shown above. Don't enable AMP; don't use `weight_decay` blindly with complex params; `nn.init.*` mostly does not accept complex tensors (use `scale * torch.rand(..., dtype=torch.cfloat)`).
3. **`modes` is capped by `seq_len // 2`.** With `seq_len=96, modes=64` you get 48 modes, not 64, and the complex weight tensor's last dim must be `len(self.index)`, never `modes`. Getting this wrong gives a silent shape mismatch or a size-1 broadcast.
4. **`irfft` needs `n=`.** `torch.fft.irfft(out_ft)` returns length `2*(n_freq-1)`, which is wrong whenever the original length was odd (e.g. `L_dec = 48 + 35`). Always pass `n=x.size(-1)`.
5. **Non-contiguous scatter indices.** `self.index` is sorted but has gaps. In `FourierCrossAttention` you must write `out_ft[..., j] = xqkvw[..., i]` (gather-index `j`, compact-index `i`) — mixing the two up scrambles frequencies. In `FourierBlock` the released code deliberately writes to the *compact* position `wi`; pick one convention and be consistent with the weight indexing.
6. **einsum subscript order.** `FourierBlock` uses `"bhi,hio->bho"` per mode with weights `[H,E_in,E_out,M]`; `FourierCrossAttention` uses `"bhex,heox->bhox"` over all modes at once. The `x`/`y` in `"bhex,bhey->bhxy"` are the **mode** axes and `e` is the channel axis being contracted — swapping them produces a plausible-looking but wrong `[B,H,E,E]` tensor.
7. **No positional encoding.** `DataEmbedding_wo_pos` adds token + temporal embeddings only. Accidentally using `DataEmbedding` (which adds `PositionalEmbedding`) changes results.
8. **The hard-coded `8`** in the complex weight shape `torch.rand(8, in//8, out//8, M)` is `n_heads`. If you set `--n_heads 4` the weight tensor still says 8 and the einsum will fail (or silently broadcast). Parameterize it by `n_heads` if you need flexibility.
9. **Head/time axis layout on return.** `FourierBlock`/`FourierCrossAttention` return `[B, H, E, L]` while `AutoCorrelationLayer` does `out.view(B, L, -1)`. The element count matches, so no error is raised — the axes are just reinterpreted. Match the reference code exactly rather than "fixing" it silently, or you'll be unable to compare numbers.
10. **`trend_init` vs `residual_trend` channel counts.** `trend_init` has `enc_in` channels, `residual_trend` has `c_out`. They're summed, so `enc_in` must equal `c_out` (the `MS` task will crash/broadcast incorrectly).
11. **Decoder lengths.** The blocks are built with `seq_len//2 + pred_len`, but the actual decoder tensor is `label_len + pred_len` long. These agree only when `label_len == seq_len//2`. If you ever set `label_len != seq_len//2`, the pre-selected mode indices and the `irfft` lengths will be inconsistent.
12. **Only fit the scaler on train.** Fitting `StandardScaler` on the whole file leaks test statistics and makes results look better than they are; also remember the `- seq_len` offset in `border1s` for val/test.
13. **`series_decomp_multi` submodules in a plain Python list** are invisible to `.parameters()`/`.to()`. Parameter-free here, but wrap in `nn.ModuleList` to be safe.
14. **`v` is unused in `FourierCrossAttention`** and `k, v` are unused in `FourierBlock`. This is genuine in the released code — don't "fix" it and then wonder why your numbers differ.


---

## 4. Pyraformer

Pyraformer is a Transformer for long-range time-series forecasting that replaces full self-attention with **pyramidal attention (PAM)** over a multi-resolution **C-ary tree** built on top of the input sequence. A *coarser-scale construction module* (CSCM) creates, level by level, shorter "summary" sequences (length L, L/C, L/C², L/C³) by strided convolution; all levels are concatenated into one long sequence of N nodes and a single fixed `[N, N]` attention mask restricts each node to attend to (a) its `A` nearest neighbours **at the same scale**, (b) its `C` **children** at the finer scale, and (c) its single **parent** at the coarser scale. Because each query attends to at most `A + C + 1` keys, time and space complexity is **O(L)**, while the tree gives a **maximum signal-traversing path of O(1)** (a constant `2(S-1) + 2(L/C^{S-1}-1)/(A-1)`) between any two time points. Prediction is done either by gathering, for every finest-scale position, its ancestor features at all scales and pushing the **last** such vector through one `Linear(S·d_model → L_pred·c_out)` ("FC" head, best-performing), or by a 2-layer cross-attention decoder fed with zero-valued "prediction tokens".

Everything below matches the official implementation at `github.com/ant-research/Pyraformer` (files `pyraformer/Layers.py`, `SubLayers.py`, `Modules.py`, `embed.py`, `Pyraformer_LR.py`, `long_range_main.py`) and the ICLR-2022 paper.

---

### Notation & Tensor Shapes

| Symbol | Code name | Meaning | Typical value |
|---|---|---|---|
| `B` | `batch_size` | batch size | 32 |
| `L` | `input_size` | historical input length (per paper `L`) | 168 / 336 / 384 / 672 |
| `L_pred` / `M` | `predict_step` | forecast horizon | 96…720 |
| `d_model` / `D` | `d_model` | node feature dimension | 512 |
| `d_inner` / `D_F` | `d_inner_hid` | hidden dim of position-wise FFN | 512 |
| `d_bottleneck` | `d_bottleneck` | bottleneck dim inside CSCM | 128 |
| `n_head` / `H` | `n_head` | attention heads | 4 (default) / 6 (long-range experiments) |
| `d_k`, `d_v` / `D_K` | `d_k`,`d_v` | per-head key/value dim | 128, 128 |
| `n_layer` / `N` | `n_layer` | number of **encoder (attention) layers** | 4 |
| `S` | `len(window_size)+1` | number of **scales / pyramid levels** | 4 |
| `C` | `window_size` | list of per-level branching factors | `[4,4,4]` (also `[5,5,5]`, `[6,6,6]`, `[12,7,4]`) |
| `A` | `inner_size` | intra-scale neighbourhood size (incl. self) | 3 or 5 (odd) |
| `enc_in` / `c_out` | `enc_in` | number of variables/channels | 7 (ETT), 1 (elect/flow/synthetic) |
| `all_size` | `all_size` | `[L, L//C0, L//C0//C1, ...]`, length `S` | `[169,42,10,2]` |
| `N_nodes` | `sum(all_size)` | total nodes in the pyramid | e.g. 223 |

**Careful:** the paper's `N` = number of attention layers; `S` = number of scales. Do **not** confuse them. In this document `N_nodes` is the concatenated pyramid length.

Core tensors:

```
x_enc      : [B, L, enc_in]          # historical observations (standardized)
x_mark_enc : [B, L, n_cov]           # time-feature covariates for the history
x_dec      : [B, L_pred, enc_in]     # zeros ("prediction tokens")
x_mark_dec : [B, L_pred, n_cov]      # future covariates (known ahead of time)
y          : [B, L_pred, c_out]      # target
seq_enc    : [B, N_nodes, d_model]   # concatenated pyramid after CSCM / encoder
mask       : [N_nodes, N_nodes] bool # True == BLOCKED (see convention below)
```

---

### Data Pipeline

#### Datasets

| Dataset | File | Channels `enc_in` | Covariates `n_cov` | Embedding | Task |
|---|---|---|---|---|---|
| ETTh1 / ETTh2 | `data/ETT/ETTh1.csv` (hourly, 2 yrs) | 7 (`HUFL,HULL,MUFL,MULL,LUFL,LULL,OT`) | 4 | `DataEmbedding` | long-range |
| ETTm1 / ETTm2 | `data/ETT/ETTm1.csv` (15-min) | 7 | 4 | `DataEmbedding` | long-range |
| Electricity | `LD2011_2014.txt` → hourly, 321/370 users | 1 | 3 + series-id | `CustomEmbedding` | single-step & long-range |
| Wind | Kaggle 30-yrs European wind, hourly, 28 countries | 1 | 3 + id | `SingleStepEmbedding` | single-step |
| App Flow | `app_zone_rpc_hour_encrypted.csv`, 1083 series | 1 | 3 + id | `SingleStepEmbedding` | single-step |
| Synthetic | `simulate_sin.py` → `synthetic.npy`, 60 series | 1 | 3 + id | `CustomEmbedding` | long-range |

#### Splits

* **ETT (follow Informer):** 12 months train / 4 months val / 4 months test. Concretely for ETTh1 with hourly rows: `border1s = [0, 12*30*24 - L, 12*30*24 + 4*30*24 - L]`, `border2s = [12*30*24, 12*30*24 + 4*30*24, 12*30*24 + 8*30*24]`. For ETTm1 multiply every `24` by `4` (15-min granularity). The `-L` offset makes each split able to produce windows whose history reaches back into the previous split — keep it, it is the standard convention.
* **Electricity (long-range):** 2011-04-01 … 2014-04-01. **Electricity (single-step):** 2011-01-01 … 2014-09-01.
* **Wind / App Flow:** train:test ≈ 32:1, split chronologically per series.
* `Dataset_Custom` (elect/flow) uses ratios 0.7 / 0.1 / 0.2 over rows. **[UNVERIFIED]** exact ratios in the repo's `data_loader.py`; use 0.7/0.1/0.2 unless you replicate exact numbers.

#### Standardization

Fit a `StandardScaler` (per-channel mean/std) **on the training split only**, then apply to train/val/test:

```python
mean = train_values.mean(0); std = train_values.std(0) + 1e-8   # [n_channels]
values = (values - mean) / std
```
Keep `mean`/`std` around so `inverse_transform` can de-normalize predictions when `-inverse` is set. Metrics in the paper are computed on **normalized** values (`-inverse` off) for ETT.

#### Time-feature covariates

For `DataEmbedding` (ETT) use **4 continuous features** scaled to roughly `[-0.5, 0.5]`:

```python
month   = df.date.dt.month   / 12.0 - 0.5
day     = df.date.dt.day     / 31.0 - 0.5
weekday = df.date.dt.weekday /  6.0 - 0.5
hour    = df.date.dt.hour    / 23.0 - 0.5
# ETTm additionally may use minute//15; keep d_inp=4 by dropping one or extending TimeFeatureEmbedding.
```
For `CustomEmbedding` (elect/flow) use `n_cov = covariate_size + 1`, where the **last** column is an integer series id passed to `nn.Embedding(seq_num, d_model)` and the first `covariate_size` columns are continuous.

#### Window construction

Sliding windows with stride 1 over the split:

```python
# index i in [0, len(split) - L - L_pred]
x_enc      = data[i        : i+L]                 # [L, enc_in]
y          = data[i+L      : i+L+L_pred]          # [L_pred, enc_in]
x_mark_enc = stamp[i       : i+L]                 # [L, n_cov]
x_mark_dec = stamp[i+L     : i+L+L_pred]          # [L_pred, n_cov]
```
Pyraformer does **not** use Informer's overlapping `label_len`; the decoder input is pure zeros of length `L_pred`.

Batch shapes out of the DataLoader: `([B,L,enc_in], [B,L_pred,enc_in], [B,L,n_cov], [B,L_pred,n_cov], mean, std)`.

#### Length / divisibility requirement

The CSCM applies `Conv1d(kernel=C[i], stride=C[i])`, so level sizes are `floor` divisions:
`all_size[i+1] = floor(all_size[i] / C[i])`. Floor is **allowed** (the paper states that when `L` is not divisible by `C`, only `floor(L/C)` nodes are created and the trailing finer nodes are all attached to the last coarse node — `get_mask` implements exactly this). Still, **prefer** `L_eff % prod(C) == 0` for clean trees; otherwise either pad the front of the sequence with zeros/edge-repeat until divisible, or accept the floor behaviour (matches the reference code).

> **Important off-by-one:** with the FC prediction head the code appends **one extra "predict token"** to the history, so the encoder is built with `input_size = L + 1` (e.g. `L=168 → 169`, `all_size = [169,42,10,2]`). With the attention decoder no token is appended and `input_size = L`.

---

### The Pyramidal Graph — index construction

#### Level sizes

```
all_size[0] = input_size                       # finest scale, s = 0 (paper's s = 1)
all_size[s] = floor(all_size[s-1] / C[s-1])    # s = 1 .. S-1
N_nodes     = sum(all_size)
```
Nodes are laid out in **one flat sequence**, finest scale first:
`offset(s) = sum(all_size[:s])`; flat index of node `(s, l)` is `offset(s) + l`.

#### Neighbour sets for node `(s, l)` (paper Eq. 2, 1-indexed there, 0-indexed here)

* **Intra-scale** `A(s,l)`: `{ (s, j) : |j - l| <= (A-1)//2 , 0 <= j < all_size[s] }` — the `A` adjacent nodes at the same scale *including itself*. Clip at the level boundaries.
* **Children** `C(s,l)` (only if `s >= 1`): `{ (s-1, j) : l*C[s-1] <= j < (l+1)*C[s-1] }`; for the **last** node of level `s` the range is extended to the end of level `s-1`, i.e. `j` in `[l*C[s-1], all_size[s-1])`.
* **Parent** `P(s,l)` (only if `s <= S-2`): `{ (s+1, min(l // C[s], all_size[s+1]-1)) }` — the parent relation is the transpose of the children relation, so it is created automatically by symmetrizing.

Attention (paper Eq. 3): the node attends only to `N(s,l) = A ∪ C ∪ P`; everything else is masked out.

#### Mask convention (PICK ONE AND KEEP IT)

**`mask[i, j] == True` means "query i may NOT attend to key j"** (i.e. `True = blocked`), which is what `ScaledDotProductAttention` consumes via `scores.masked_fill(mask, -1e9)`. The builder below creates a `1 = allowed` float matrix and inverts it at the very end.

#### `get_mask`

```python
import math, torch

def get_mask(input_size, window_size, inner_size, device='cpu'):
    """
    input_size : int   -- length of the finest scale (== L, or L+1 for the FC head)
    window_size: list[int] of length S-1, e.g. [4,4,4]
    inner_size : int   -- A, odd (3 or 5)
    returns:
      mask     : BoolTensor [N_nodes, N_nodes], True == masked out
      all_size : list[int] of length S
    """
    # ---- 1. level sizes -------------------------------------------------
    all_size = [input_size]
    for i in range(len(window_size)):
        all_size.append(math.floor(all_size[i] / window_size[i]))
    seq_length = sum(all_size)                       # N_nodes
    mask = torch.zeros(seq_length, seq_length, device=device)   # 1 == allowed

    # ---- 2. intra-scale (A adjacent nodes, incl. self) ------------------
    inner_window = inner_size // 2                   # A=3 -> 1 ; A=5 -> 2
    for layer_idx in range(len(all_size)):
        start = sum(all_size[:layer_idx])
        for i in range(start, start + all_size[layer_idx]):
            left  = max(i - inner_window, start)
            right = min(i + inner_window + 1, start + all_size[layer_idx])
            mask[i, left:right] = 1                  # includes the diagonal

    # ---- 3. inter-scale (children; parents come from symmetrization) ----
    for layer_idx in range(1, len(all_size)):
        start = sum(all_size[:layer_idx])            # offset of level layer_idx
        for i in range(start, start + all_size[layer_idx]):
            child_base = start - all_size[layer_idx - 1]        # offset of finer level
            left  = child_base + (i - start) * window_size[layer_idx - 1]
            if i == start + all_size[layer_idx] - 1:            # LAST node of this level
                right = start                                   # -> absorb the tail
            else:
                right = child_base + (i - start + 1) * window_size[layer_idx - 1]
            mask[i, left:right] = 1      # parent attends to children
            mask[left:right, i] = 1      # children attend to parent  (symmetric)

    mask = (1 - mask).bool()             # invert: True == MASKED OUT
    return mask, all_size
```

Notes:
* `inner_size` should be **odd**. For even `A`, `A//2` makes the window `A+1` wide and symmetric — an off-by-one; just require odd.
* The graph is **undirected/bidirectional**: the mask is symmetric by construction, so information flows both up and down the tree.
* Build the mask **once** in `Encoder.__init__` and reuse (`self.mask`); never rebuild per batch. At forward time do `mask.repeat(B,1,1)` (or better: `mask.unsqueeze(0)` and let broadcasting handle it).

#### `refer_points` — gathering ancestors

For every finest-scale position `i`, this returns its flat index at every scale (itself, its parent, grandparent, …), used by the FC prediction head.

```python
def refer_points(all_sizes, window_size, device='cpu'):
    """
    returns LongTensor of shape [1, input_size, S, 1]
    indexes[0, i, j, 0] = flat index of the level-j ancestor of finest-scale node i
    """
    input_size = all_sizes[0]
    indexes = torch.zeros(input_size, len(all_sizes), device=device)
    for i in range(input_size):
        indexes[i][0] = i
        former_index = i
        for j in range(1, len(all_sizes)):
            start = sum(all_sizes[:j])                       # offset of level j
            inner_layer_idx = former_index - (start - all_sizes[j - 1])   # local idx at level j-1
            former_index = start + min(inner_layer_idx // window_size[j - 1],
                                       all_sizes[j] - 1)     # clamp to last node
            indexes[i][j] = former_index
    return indexes.unsqueeze(0).unsqueeze(3).long()          # [1, input_size, S, 1]
```

Usage in the encoder (S = `len(all_size)`):

```python
idx = self.indexes.repeat(B, 1, 1, d_model).view(B, -1, d_model)  # [B, L*S, d_model]
all_enc = torch.gather(seq_enc, 1, idx)                           # [B, L*S, d_model]
seq_enc = all_enc.view(B, all_size[0], -1)                        # [B, L, S*d_model]
```

#### `get_subsequent_mask` — for the attention decoder

```python
def get_subsequent_mask(input_size, window_size, predict_step, truncate):
    """
    Causal mask for the SECOND decoder layer, whose keys are [encoder_out ; decoder_out].
    returns BoolTensor [1, predict_step, K]  (True == masked out)
    """
    if truncate:                      # only the finest scale is kept as context
        K_hist = input_size
    else:                             # the whole pyramid is context
        all_size = [input_size]
        for w in window_size:
            all_size.append(math.floor(all_size[-1] / w))
        K_hist = sum(all_size)
    mask = torch.zeros(predict_step, K_hist + predict_step)
    for i in range(predict_step):
        mask[i][: K_hist + i + 1] = 1        # can see all history + own past + itself
    return (1 - mask).bool().unsqueeze(0)
```
(The first decoder layer uses **no** mask — the prediction tokens cross-attend freely to the whole encoder output.)

#### Worked example: `L = 16`, `C = [2, 2]`, `A = 3`

```
all_size = [16, 8, 4]         S = 3
offsets  = [ 0, 16, 24]       N_nodes = 28
flat idx : 0..15   -> level 0 (finest)
           16..23  -> level 1
           24..27  -> level 2 (coarsest)
```

Allowed (`mask == False`) entries:

* **Intra-scale, level 0** (`inner_window = 1`): `{(i, i-1), (i, i), (i, i+1)}` for `i = 0..15`, clipped at `0` and `15`. E.g. row 0 allows columns `{0,1}`; row 7 allows `{6,7,8}`; row 15 allows `{14,15}`.
* **Intra-scale, level 1:** rows `16..23` allow `{i-1,i,i+1}` clipped to `[16,23]`. Row 16 → `{16,17}`; row 23 → `{22,23}`.
* **Intra-scale, level 2:** rows `24..27` allow `{i-1,i,i+1}` clipped to `[24,27]`.
* **Children, level 1 → level 0:** node `16+k` (k=0..6) → columns `{2k, 2k+1}`; node `23` (last) → columns `{14, 15}` (the tail rule gives `left=14, right=16` since `start=16`). So node 16→{0,1}, 17→{2,3}, …, 22→{12,13}, 23→{14,15}.
* **Children, level 2 → level 1:** node `24+k` (k=0..2) → `{16+2k, 17+2k}`; node `27` (last) → `left = 16 + 3*2 = 22, right = start = 24` → `{22, 23}`. So 24→{16,17}, 25→{18,19}, 26→{20,21}, 27→{22,23}.
* **Parents:** every child pair above is mirrored, e.g. `mask[0,16] = mask[1,16] = allowed`, `mask[16,24] = allowed`.

Everything else (e.g. `mask[0, 15]`, `mask[0, 24]`) is `True` = masked. Total allowed entries ≈ `3·16 + 3·8 + 3·4 - boundary_corrections + 2·(16 + 8)` — each query sees at most `A + C + 1 = 3 + 2 + 1 = 6` keys.

---

### Module-by-Module Spec

#### 1. `PositionalEmbedding`

```python
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        # pe: [1, max_len, d_model], registered as a buffer, NOT trainable
        # pe[p, 2i]   = sin(p * exp(2i * -log(10000)/d_model))
        # pe[p, 2i+1] = cos(p * exp(2i * -log(10000)/d_model))
    def forward(self, x):            # x: [B, L, *]
        return self.pe[:, :x.size(1)]   # [1, L, d_model]
```

#### 2. `TokenEmbedding` (value embedding)

```python
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        self.tokenConv = nn.Conv1d(c_in, d_model, kernel_size=3,
                                   padding=1, padding_mode='circular')
        # kaiming_normal_(w, mode='fan_in', nonlinearity='leaky_relu')
    def forward(self, x):            # [B, L, c_in]
        return self.tokenConv(x.permute(0,2,1)).transpose(1,2)   # [B, L, d_model]
```
A plain `nn.Linear(c_in, d_model)` also works but the circular 1-D conv is what the paper uses (inherited from Informer).

#### 3. `DataEmbedding` (ETT)

```python
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        self.value_embedding    = TokenEmbedding(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.temporal_embedding = nn.Linear(4, d_model)     # TimeFeatureEmbedding, d_inp = 4
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, x_mark):    # x:[B,L,c_in]  x_mark:[B,L,4]
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)       # [B, L, d_model]
```

#### 4. `CustomEmbedding` (electricity / app-flow / synthetic)

```python
class CustomEmbedding(nn.Module):
    def __init__(self, c_in, d_model, temporal_size, seq_num, dropout=0.1):
        self.value_embedding    = TokenEmbedding(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.temporal_embedding = nn.Linear(temporal_size, d_model)
        self.seqid_embedding    = nn.Embedding(seq_num, d_model)
    def forward(self, x, x_mark):    # x_mark: [B, L, temporal_size + 1]; last col = series id
        x = (self.value_embedding(x) + self.position_embedding(x)
             + self.temporal_embedding(x_mark[:, :, :-1])
             + self.seqid_embedding(x_mark[:, :, -1].long()))
        return self.dropout(x)
```

#### 5. CSCM — `Bottleneck_Construct` (default)

Constructor: `Bottleneck_Construct(d_model, window_size, d_inner)` where `d_inner = d_bottleneck = 128`.

```python
class ConvLayer(nn.Module):
    def __init__(self, c_in, window_size):
        self.downConv   = nn.Conv1d(c_in, c_in, kernel_size=window_size, stride=window_size)
        self.norm       = nn.BatchNorm1d(c_in)        # over channels; input must be [B, C, T]
        self.activation = nn.ELU()
    def forward(self, x):            # [B, c_in, T]  ->  [B, c_in, floor(T/window_size)]
        return self.activation(self.norm(self.downConv(x)))

class Bottleneck_Construct(nn.Module):
    def __init__(self, d_model, window_size, d_inner):
        self.conv_layers = nn.ModuleList([ConvLayer(d_inner, w) for w in window_size])
        self.down = nn.Linear(d_model, d_inner)
        self.up   = nn.Linear(d_inner, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):                      # [B, L, d_model]
        temp = self.down(enc_input).permute(0, 2, 1)   # [B, d_inner, L]
        all_inputs = []
        for conv in self.conv_layers:
            temp = conv(temp)                          # [B, d_inner, L/C0], then /C0C1, then /C0C1C2
            all_inputs.append(temp)
        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)   # [B, L/C0+L/C0C1+L/C0C1C2, d_inner]
        all_inputs = self.up(all_inputs)                            # [B, ..., d_model]
        all_inputs = torch.cat([enc_input, all_inputs], dim=1)      # [B, N_nodes, d_model]
        return self.norm(all_inputs)                                # LayerNorm over d_model
```

Shapes with `L=169, C=[4,4,4], d_model=512, d_inner=128`:
`[B,169,512] -down-> [B,169,128] -permute-> [B,128,169] -conv0-> [B,128,42] -conv1-> [B,128,10] -conv2-> [B,128,2]`;
concat over time → `[B,128,54]` → transpose `[B,54,128]` → `up` `[B,54,512]` → concat with input → `[B,223,512]` → LayerNorm.
Note the level order in the flat sequence is exactly `[level0 | level1 | level2 | level3]`, matching `all_size` and `get_mask`.

**Variants** (same output contract, replace the conv stack):

* `Conv_Construct(d_model, window_size, d_inner)`: no bottleneck — `ConvLayer(d_model, C[i])` applied directly on the permuted `[B,d_model,L]`; level 0 is appended *first* then coarser levels; final `LayerNorm`. ~10× more parameters, MSE only ~1.5% better (paper Table 8).
* `MaxPooling_Construct` / `AvgPooling_Construct`: `nn.MaxPool1d(kernel_size=C[i])` / `nn.AvgPool1d(kernel_size=C[i])`, zero parameters, worse MSE (0.842 / 0.833 vs 0.808 for bottleneck conv on ETTh1-168).

#### 6. `ScaledDotProductAttention`

```python
class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.2):
        self.temperature = temperature          # d_k ** 0.5
        self.dropout = nn.Dropout(attn_dropout)
    def forward(self, q, k, v, mask=None):
        # q,k: [B, n_head, Lq, d_k]; v: [B, n_head, Lv, d_v]
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))   # [B,H,Lq,Lk]
        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)        # mask: [B,1,Lq,Lk] bool, True == blocked
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)                 # [B,H,Lq,d_v]
        return output, attn
```

#### 7. `MultiHeadAttention`

Constructor `MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=0.1, normalize_before=True)`.

```python
w_qs = nn.Linear(d_model, n_head*d_k, bias=False)   # xavier_uniform_
w_ks = nn.Linear(d_model, n_head*d_k, bias=False)
w_vs = nn.Linear(d_model, n_head*d_v, bias=False)
fc   = nn.Linear(n_head*d_v, d_model)               # xavier_uniform_
attention  = ScaledDotProductAttention(temperature=d_k**0.5, attn_dropout=dropout)
layer_norm = nn.LayerNorm(d_model, eps=1e-6)
dropout    = nn.Dropout(dropout)

def forward(q, k, v, mask=None):        # q:[B,Lq,d_model], k,v:[B,Lk,d_model]
    residual = q
    if normalize_before: q = layer_norm(q)
    q = w_qs(q).view(B, Lq, H, d_k).transpose(1,2)   # [B,H,Lq,d_k]
    k = w_ks(k).view(B, Lk, H, d_k).transpose(1,2)
    v = w_vs(v).view(B, Lv, H, d_v).transpose(1,2)
    if mask is not None and mask.dim() == 3: mask = mask.unsqueeze(1)   # [B,1,Lq,Lk]
    out, attn = attention(q, k, v, mask=mask)        # [B,H,Lq,d_v]
    out = out.transpose(1,2).contiguous().view(B, Lq, H*d_v)
    out = dropout(fc(out)) + residual
    if not normalize_before: out = layer_norm(out)
    return out, attn
```
Note: **only the query is layer-normed** in pre-norm mode (this is the reference behaviour). The long-range model uses `normalize_before=False` (post-norm).

#### 8. `PositionwiseFeedForward`

```python
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1, normalize_before=True):
        self.w_1 = nn.Linear(d_in, d_hid); self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6); self.dropout = nn.Dropout(dropout)
    def forward(self, x):                  # [B, T, d_in]
        residual = x
        if self.normalize_before: x = self.layer_norm(x)
        x = self.dropout(F.gelu(self.w_1(x)))     # GELU, not ReLU
        x = self.dropout(self.w_2(x)) + residual
        if not self.normalize_before: x = self.layer_norm(x)
        return x
```

#### 9. `EncoderLayer`

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, normalize_before=True):
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout, normalize_before)
        self.pos_ffn  = PositionwiseFeedForward(d_model, d_inner, dropout, normalize_before)
    def forward(self, enc_input, slf_attn_mask=None):   # [B, N_nodes, d_model]
        x, attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        return self.pos_ffn(x), attn                    # [B, N_nodes, d_model]
```
(The paper additionally ships a TVM CUDA kernel `PAM_TVM.PyramidalAttention` that materializes only `A+C+1` keys per query via the `get_q_k` / `get_k_q` index tables, giving true O(L) memory. The naive masked version above is mathematically identical but O(N_nodes²) in memory — fine for `L ≤ ~1000`.)

#### 10. `Encoder`

```python
class Encoder(nn.Module):
    def __init__(self, opt):
        # NOTE the +1 for the FC head's predict token
        size = opt.input_size if opt.decoder == 'attention' else opt.input_size + 1
        self.mask, self.all_size = get_mask(size, opt.window_size, opt.inner_size, opt.device)
        if opt.decoder == 'FC':
            self.indexes = refer_points(self.all_size, opt.window_size, opt.device)  # [1,L+1,S,1]
        self.layers = nn.ModuleList([
            EncoderLayer(opt.d_model, opt.d_inner_hid, opt.n_head, opt.d_k, opt.d_v,
                         dropout=opt.dropout, normalize_before=False)
            for _ in range(opt.n_layer)])                                    # n_layer = 4
        self.enc_embedding = DataEmbedding(...) or CustomEmbedding(...)
        self.conv_layers   = Bottleneck_Construct(opt.d_model, opt.window_size, opt.d_bottleneck)

    def forward(self, x_enc, x_mark_enc):
        seq_enc = self.enc_embedding(x_enc, x_mark_enc)      # [B, L(+1), d_model]
        mask    = self.mask.repeat(B, 1, 1).to(device)       # [B, N, N] bool
        seq_enc = self.conv_layers(seq_enc)                  # [B, N, d_model]
        for layer in self.layers:
            seq_enc, _ = layer(seq_enc, mask)                # [B, N, d_model]  (same mask every layer)

        if self.decoder_type == 'FC':
            idx = self.indexes.repeat(B, 1, 1, d_model).view(B, -1, d_model)  # [B, (L+1)*S, d_model]
            all_enc = torch.gather(seq_enc, 1, idx)                           # [B, (L+1)*S, d_model]
            seq_enc = all_enc.view(B, self.all_size[0], -1)                   # [B, L+1, S*d_model]
        elif self.decoder_type == 'attention' and self.truncate:
            seq_enc = seq_enc[:, :self.all_size[0]]                           # drop coarse nodes
        return seq_enc
```

#### 11. Prediction heads

##### (a) `FC` head — "prediction module 1" (default; best results in the paper)

* Append one zero **predict token** to the history (`batch_x` → length `L+1`; its covariate row is `x_mark_dec[:, 0:1, :]`).
* Encode; gather ancestors with `refer_points` → `[B, L+1, S*d_model]`.
* Take the **last** position only (the predict token, which after PAM has absorbed the whole pyramid) and map it to all horizons at once:

```python
self.predictor = Predictor(S * d_model, predict_step * enc_in)   # S = 4 -> Linear(2048, M*7), bias=False
enc_output = self.encoder(x_enc, x_mark_enc)[:, -1, :]           # [B, S*d_model]
pred = self.predictor(enc_output).view(B, predict_step, -1)      # [B, L_pred, c_out]
```

```python
class Predictor(nn.Module):
    def __init__(self, dim, num_types):
        self.linear = nn.Linear(dim, num_types, bias=False)   # xavier_normal_ init
    def forward(self, data): return self.linear(data)
```

Used for: **all single-step experiments** (there `num_types` = 1 mean + 1 variance, or just 1), and it is the head that produced the reported long-range numbers for ETTh1/ETTm1/Electricity.

##### (b) `attention` head — "prediction module 2"

```python
class Decoder(nn.Module):
    def __init__(self, opt, mask):        # mask from get_subsequent_mask
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout, normalize_before=False),
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout, normalize_before=False)])
        self.dec_embedding = DataEmbedding/CustomEmbedding(...)
    def forward(self, x_dec, x_mark_dec, refer):    # x_dec = ZEROS [B, M, enc_in]; refer = encoder out [B, N, d_model]
        F_p  = self.dec_embedding(x_dec, x_mark_dec)                 # [B, M, d_model]  "prediction tokens"
        F_d1, _ = self.layers[0](F_p, refer, refer)                  # Q=F_p, K=V=F_e ; no mask
        refer_enc = torch.cat([refer, F_d1], dim=1)                  # [B, N+M, d_model]
        m = self.mask.repeat(B, 1, 1)                                # [B, M, N+M]
        F_d2, _ = self.layers[1](F_d1, refer_enc, refer_enc, slf_attn_mask=m)
        return F_d2                                                  # [B, M, d_model]
# then: pred = Predictor(d_model, enc_in)(F_d2)   -> [B, M, c_out]
```
`DecoderLayer` = `MultiHeadAttention(Q,K,V,mask)` → `PositionwiseFeedForward`, identical internals to `EncoderLayer` but with separate Q vs K/V.

The paper reports that head (a) beats head (b) — the FC head can exploit features of different resolutions, whereas the full-attention decoder cannot distinguish them. Try both, keep the better.

---

### Full Model Assembly

```python
class Pyraformer(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.predict_step = opt.predict_step
        self.decoder_type = opt.decoder            # 'FC' | 'attention'
        self.input_size   = opt.input_size
        self.encoder = Encoder(opt)
        if opt.decoder == 'attention':
            mask = get_subsequent_mask(opt.input_size, opt.window_size,
                                       opt.predict_step, opt.truncate)
            self.decoder   = Decoder(opt, mask)
            self.predictor = Predictor(opt.d_model, opt.enc_in)
        else:  # 'FC'
            S = len(opt.window_size) + 1           # == 4
            self.predictor = Predictor(S * opt.d_model, opt.predict_step * opt.enc_in)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, pretrain=False):
        # x_enc:[B,L(+1),enc_in]  x_mark_enc:[B,L(+1),n_cov]
        # x_dec:[B,M,enc_in] (zeros)  x_mark_dec:[B,M,n_cov]
        if self.decoder_type == 'attention':
            enc_out = self.encoder(x_enc, x_mark_enc)             # [B, N_nodes, d_model]
            dec_out = self.decoder(x_dec, x_mark_dec, enc_out)    # [B, M, d_model]
            if pretrain:                                          # auto-encoding warm-up
                dec_out = torch.cat([enc_out[:, :self.input_size], dec_out], dim=1)  # [B, L+M, d_model]
            pred = self.predictor(dec_out)                        # [B, (L+)M, enc_in]
        else:  # FC
            enc_out = self.encoder(x_enc, x_mark_enc)[:, -1, :]   # [B, S*d_model]
            pred = self.predictor(enc_out).view(enc_out.size(0),
                                                self.predict_step, -1)  # [B, M, enc_in]
        return pred
```

Caller-side for the FC head (done in the training loop, **not** inside the model):

```python
predict_token = torch.zeros(B, 1, x_enc.size(-1), device=x_enc.device)
x_enc      = torch.cat([x_enc, predict_token], dim=1)            # [B, L+1, enc_in]
x_mark_enc = torch.cat([x_mark_enc, x_mark_dec[:, 0:1, :]], 1)   # [B, L+1, n_cov]
```

---

### Complexity Analysis

Let `L(s) = L / C^{s-1}` be the number of nodes at scale `s` (1-indexed, `s = 1..S`).

**Number of Q-K dot products.** A query at `(s, l)` attends to
`P(s,l) = P_intra + P_inter ≤ A + (C + 1)`
(`A` intra-scale incl. self, `C` children, `1` parent; the finest scale has no children so `P ≤ A + 1`, the coarsest has no parent).

```
P = Σ_{s=1..S} Σ_l P(s,l)
  ≤ L(A+1) + Σ_{s=2..S} L(s)·(A + C + 1)
  =  L · ( Σ_{s=1..S} C^{-(s-1)}·A + Σ_{s=2..S} C^{-(s-1)} + Σ_{s=1..S-1} C^{-(s-1)} + 1 )
  <  L · ( (A+2) · Σ_{s=1..S} C^{-(s-1)} + 1 )
```
The geometric sum `Σ C^{-(s-1)} < C/(C-1) ≤ 2`, so `P = O(A·L)`; with `A` a constant (3 or 5) independent of `L`, **complexity = O(L)** in both time and memory. Choosing `C ∝ L^{1/(S-1)}` makes `P = O( A·(L^{S/(S-1)} - 1)/(L^{1/(S-1)} - 1) ) → O(A·L)`.

**Maximum signal traversing path.** The worst-case pair is `n(1,1)` ↔ `n(1,L)`; the shortest route climbs the tree, walks across the coarsest scale, and descends:

```
n(1,1) → n(2,1) → … → n(S,1) → … → n(S, L(S)) → n(S-1, L(S-1)) → … → n(1,L)
L_max = 2(S-1) + 2(L(S) - 1)/(A - 1)
```
With `N` stacked attention layers, the coarsest scale has a **global receptive field** iff (Lemma 1)

```
L / C^{S-1} - 1  ≤  (A - 1) · N / 2
```
Equivalently pick `C` with `L^{1/(S-1)} ≥ C ≥ ( L / ((A-1)N/2 + 1) )^{1/(S-1)}`. Then `L_max = O(2(S-1) + N) = O(S + N)`, and since `A, S, N` are fixed constants, **max path = O(1)**.

Comparison: Full attention `O(L²)` / `O(1)`; LogTrans `O(L log L)` / `O(log L)`; Longformer `O(L)` / `O(L)`; ETC `O(GL)` / `O(1)`; **Pyraformer `O(L)` / `O(1)`**.

---

### Default Hyperparameters

| Arg | Default | Notes |
|---|---|---|
| `d_model` | 512 | |
| `d_inner_hid` (`d_ff`) | 512 | FFN hidden width |
| `d_bottleneck` | 128 | CSCM bottleneck |
| `d_k` = `d_v` | 128 | per head |
| `n_head` | 4 (CLI default), **6** for the long-range ETT/Elect runs | |
| `n_layer` (`N`) | 4 | encoder layers |
| `S` (scales) | 4 | = `len(window_size)+1` |
| `window_size` (`C`) | `[4, 4, 4]` | also 5,5,5 / 6,6,6 / 12,7,4 |
| `inner_size` (`A`) | 3 | or 5; must be odd |
| `CSCM` | `Bottleneck_Construct` | alternatives: `Conv_Construct`, `MaxPooling_Construct`, `AvgPooling_Construct` |
| `decoder` | `FC` | or `attention` |
| `dropout` | 0.05 | |
| optimizer | Adam, `lr = 1e-4` | long-range |
| lr schedule | `StepLR(step_size=1, gamma=0.1)` | ÷10 every epoch |
| `batch_size` | 32 | |
| `epoch` | 5 (long-range), 10 (single-step) | |
| loss | MSE (long-range); MSE + 100× log-likelihood (single-step) | |
| `iter_num` | 5 | repeat runs, average metrics |
| single-step lr | `1e-5`, halved each epoch | |
| single-step `C` | 4, `A` = 3, `H` = 4 | history 169 (Electricity), 192 (Wind, App Flow), incl. end token |
| `truncate` | False | drop coarse nodes from decoder context if True |
| `use_tvm` | False | requires constant `window_size` |

#### Per-experiment settings (paper Table 5)

| Dataset | `L_pred` | `N` | `S` | `H` | `A` | `C` | history `L` |
|---|---|---|---|---|---|---|---|
| ETTh1 | 168 | 4 | 4 | 6 | 3 | 4 | 168 |
| ETTh1 | 336 | 4 | 4 | 6 | 3 | 4 | 168 |
| ETTh1 | 720 | 4 | 4 | 6 | 5 | 4 | 336 |
| ETTm1 | 96 | 4 | 4 | 6 | 3 | 5 | 384 |
| ETTm1 | 288 | 4 | 4 | 6 | 5 | 5 | 672 |
| ETTm1 | 672 | 4 | 4 | 6 | 3 | 6 | 672 |
| Electricity | 168 | 4 | 4 | 6 | 3 | 4 | 168 |
| Electricity | 336 | 4 | 4 | 6 | 3 | 4 | 168 |
| Electricity | 720 | 4 | 4 | 6 | 3 | 5 | 336 |

Hyper-parameter selection recipe (Appendix K): fix `N` by compute budget → pick `S` from the natural granularities (hourly data ⇒ hour/day/week/month ⇒ `S = 4`) → keep `A` small (3 or 5) → choose `C` satisfying the Lemma-1 inequality, validated on a held-out set; per-scale `C` matched to known periods (e.g. `[12, 7, 4]` for half-day/half-week/half-month) can help.

---

### Training Loop

```python
opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Pyraformer(opt).to(opt.device)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=opt.lr_step)  # gamma=0.1

for epoch in range(opt.epoch):                      # 5
    model.train()
    for batch_x, batch_y, batch_x_mark, batch_y_mark, mean, std in train_loader:
        # -> float().to(device)
        dec_inp = torch.zeros_like(batch_y)                       # [B, M, enc_in]
        if opt.decoder == 'FC':
            tok = torch.zeros(batch_x.size(0), 1, batch_x.size(-1), device=device)
            batch_x      = torch.cat([batch_x, tok], dim=1)
            batch_x_mark = torch.cat([batch_x_mark, batch_y_mark[:, 0:1, :]], dim=1)
        optimizer.zero_grad()
        out = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, False)   # [B, M, enc_in]
        if opt.inverse:
            out, batch_y = dataset.inverse_transform(out, batch_y, mean, std)
        losses = nn.MSELoss(reduction='none')(out, batch_y)
        losses.mean().backward()
        optimizer.step()
    evaluate(...)
    scheduler.step()
```

Optional tricks (all off by default, toggled per experiment):
* **`-pretrain`** (attention decoder only, epoch 0): run as an auto-encoder — concatenate `enc_out[:, :L]` to the decoder output and supervise against `cat([batch_x, batch_y])`, forcing the PAM to also *reconstruct* the input and thereby learn long-range structure instead of copying the last value.
* **`-hard_sample_mining`**: `TopkMSELoss(topk)` keeps only the `topk` worst samples in the batch; schedule `topk = B` for `epoch<2`, `int(B*(5-epoch)/(6-epoch))` for `epoch<4`, else `int(0.5*B)`.
* **Weighted sampler** by each window's mean value (single-step experiments).
* Run `iter_num = 5` independent trainings and average the metrics; keep the checkpoint with the best test MSE.

---

### Evaluation

**Long-range (multi-step):** MSE and MAE on the normalized series.

```python
mae  = np.mean(np.abs(pred - true))
mse  = np.mean((pred - true) ** 2)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((pred - true) / true))
mspe = np.mean(((pred - true) / true) ** 2)
```

**Single-step:** NRMSE and ND
`NRMSE = sqrt( mean((z - ẑ)²) ) / mean(|z|)`,  `ND = Σ|z - ẑ| / Σ|z|`.

#### Reported Pyraformer numbers (paper Tables 2 & 3)

Long-range (MSE / MAE):

| Dataset | 168 | 336 | 720 |
|---|---|---|---|
| ETTh1 | 0.808 / 0.683 | 0.945 / 0.766 | 1.022 / 0.806 |
| Electricity | 0.719 / 0.256 | 1.533 / 0.291 | 4.312 / 0.346 |

| ETTm1 | 96 | 288 | 672 |
|---|---|---|---|
| MSE / MAE | 0.480 / 0.486 | 0.754 / 0.659 | 0.857 / 0.707 |

Single-step (NRMSE / ND): Electricity 0.324 / 0.041; Wind 0.161 / 0.072; App Flow 0.366 / 0.067.
Q-K pairs (proxy for cost) e.g. ETTh1-168: Pyraformer 26 472 vs Informer 188 040 vs Reformer 1 016 064.

Sanity ablations to reproduce: bottleneck-conv CSCM ≈ conv CSCM but 90% fewer params; removing the PAM (CSCM only) degrades ETTm1-96 MSE from 0.480 → 0.576; longer history helps (ETTm1, `L_pred=1344`: MSE 1.234 @ L=84 → 1.057 @ L=672).

---

### Implementation Checklist

1. `pyraformer/embed.py` — `PositionalEmbedding`, `TokenEmbedding`, `TimeFeatureEmbedding`, `DataEmbedding`, `CustomEmbedding`, `SingleStepEmbedding`.
2. `pyraformer/Modules.py` — `ScaledDotProductAttention`.
3. `pyraformer/SubLayers.py` — `MultiHeadAttention`, `PositionwiseFeedForward`.
4. `pyraformer/Layers.py` —
   a. `get_mask(input_size, window_size, inner_size, device)` → `(mask [N,N] bool, all_size)`
   b. `refer_points(all_sizes, window_size, device)` → `[1, L, S, 1]` long
   c. `get_subsequent_mask(input_size, window_size, predict_step, truncate)`
   d. `ConvLayer`, `Bottleneck_Construct`, `Conv_Construct`, `MaxPooling_Construct`, `AvgPooling_Construct`
   e. `EncoderLayer`, `DecoderLayer`, `Decoder`, `Predictor`
5. `pyraformer/Pyraformer_LR.py` — `Encoder`, `Model` (the full `Pyraformer`).
6. `data_loader.py` — `Dataset_ETT_hour`, `Dataset_ETT_minute`, `Dataset_Custom`, `Dataset_Synthetic`, scaler + time features.
7. `utils/tools.py` — `metric(pred, true)` → `(mae, mse, rmse, mape, mspe)`, `TopkMSELoss`.
8. `long_range_main.py` — argparse (see hyper-parameter table), `dataset_parameters()`, `train_epoch`, `eval_epoch`, `train`, `evaluate`, 5-iteration averaging.
9. `single_step_main.py` — single-step variant (Gaussian head, weighted sampler, pretraining).
10. Unit tests: (i) `get_mask` on the `L=16, C=[2,2], A=3` example matches the listed entries; (ii) `refer_points` ancestors are monotone and in-range; (iii) forward pass shapes; (iv) one overfit step on 8 samples drives the loss toward 0.

---

### Common Pitfalls

1. **Build the mask once.** `get_mask` is a pure-Python double loop — O(N²) work. Build it in `Encoder.__init__`, store as a buffer, and reuse it for *every* layer and *every* batch. Rebuilding per forward pass dominates runtime.
2. **Mask polarity.** This document uses **`True = masked out`** (matching `masked_fill(mask, -1e9)`). `get_mask` internally builds `1 = allowed` and inverts at the end — do not forget the `(1 - mask).bool()`. Inverting it silently makes every node attend to exactly the nodes it should ignore, and the model still trains (badly).
3. **The `+1` predict token.** With the FC head the encoder is built with `input_size = L + 1`, and the caller must append the zero token and the first future covariate row. Forgetting either gives a shape mismatch between `all_size[0]` and the embedded sequence, or silently shifts the forecast by one step.
4. **Divisibility.** `all_size[s+1] = floor(all_size[s] / C[s])`. If `prod(window_size)` does not divide `L(+1)`, the tail nodes at each level are absorbed by the *last* coarse node (both in the conv, which drops the remainder, and in `get_mask`, which extends the last parent's child range to the end of the level). Verify `all_size[-1] >= 2`, otherwise the coarsest scale is degenerate. Pad the input if you want exact trees.
5. **`inner_size` must be odd.** The code uses `inner_window = inner_size // 2` and the window `[i-inner_window, i+inner_window]` — for even `A` this yields `A+1` neighbours, an off-by-one relative to the paper's `|j-l| ≤ (A-1)/2`.
6. **BatchNorm axis in `ConvLayer`.** `nn.BatchNorm1d(c_in)` and `nn.Conv1d` both expect `[B, channels, time]`. The CSCM must `permute(0,2,1)` on the way in and `transpose(1,2)` on the way out. Applying BatchNorm on `[B, T, D]` normalizes over the *time* axis and breaks for variable-length batches.
7. **Concatenation order must match `all_size`.** `Bottleneck_Construct` outputs `[level0 | level1 | … | level_{S-1}]` in exactly that order; `get_mask` and `refer_points` assume the same layout with offsets `sum(all_size[:s])`. Any reordering corrupts the graph silently.
8. **`refer_points` gather.** `torch.gather(seq_enc, 1, idx)` needs `idx` of the *same rank and last-dim size* as `seq_enc`: expand `[1,L,S,1] → [B,L,S,d_model] → view(B, L*S, d_model)`. Then reshape the gathered result to `[B, L, S*d_model]` — **not** `[B, S, L*d_model]`.
9. **FC head reads only the last position.** `self.encoder(...)[:, -1, :]` — the predict token. Using the mean or the whole sequence changes the model.
10. **`d_model` is not necessarily `n_head * d_k`.** With `n_head = 4, d_k = 128` it happens to match (`4*128 = 512`), but the long-range runs use `n_head = 6, d_k = 128`, so `w_qs` projects `512 → 768` and `fc` maps `768 → 512`. Never hard-code `d_k = d_model // n_head`; take `d_k`/`d_v` as explicit constructor arguments.
11. **Memory.** The naive masked PAM still materializes an `[B, H, N, N]` score matrix. For `L > ~2000` you need the TVM/sparse kernel (or a gather-based implementation using `get_q_k`) to realize the advertised O(L).
12. **LR decays by 10× every epoch** (`StepLR(1, gamma=0.1)`) with only 5 epochs — effectively almost all learning happens in epochs 1–2. Do not substitute a cosine schedule without re-tuning.

---

### [UNVERIFIED] items

* Exact train/val/test ratio inside `Dataset_Custom` for Electricity / App Flow (stated above as 0.7/0.1/0.2) — `data_loader.py` was not read directly.
* The precise construction of the 4 ETT time features (`month/day/weekday/hour` scaled to `[-0.5,0.5]`) is inherited from Informer's `timeFeatures`; `TimeFeatureEmbedding` hard-codes `d_inp = 4`, which is confirmed, the individual feature formulas are the standard Informer ones.
* Whether the single-step head outputs `(mu, sigma)` of a Gaussian (paper says "mean and variance") vs a point estimate in the released `single_step_main.py`.
* The attention decoder's first layer using no mask at all is inferred from `Decoder.forward` (no `slf_attn_mask` argument passed) — confirmed in code, but the paper does not state it explicitly.
