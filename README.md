# Transformer Options Pricing

Forecast-driven option valuation. Four long-sequence forecasting transformers (Informer, Autoformer, FEDformer, Pyraformer) are trained on a stock's recent price history and macro indicators. They predict daily log returns over a short horizon. The forecast spot price is then used to value near-the-money options with Black-Scholes and compute their Greeks. See *ex_input_output.md* for example pipeline usage. Transformer models were recreated from the following papers: 

* [Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting](https://arxiv.org/abs/2106.13008) (NeurIPS 2021)
* [Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://arxiv.org/abs/2012.07436) (AAAI 2021)
* [Pyraformer: Low-Complexity Pyramidal Attention for Long-Range Time Series Modeling and Forecasting](https://www.google.com/search?q=https://openreview.net/forum%3Fid%3D0EXmFzVa5ge) (ICLR 2022)
* [FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting](https://arxiv.org/abs/2201.12740) (ICML 2022)
* [Transformer Architectures for Option Pricing: Valuation, Prediction, and Trading](https://www.google.com/search?q=https://doi.org/10.1007/978-3-032-25035-3_18) (ICAART 2025 / Springer)

The pipeline is coordinated by two Claude agents (LangChain + LangGraph), with deterministic validation gates between them:

```
[Market Data Agent] --> [Schema Gate] --> [Transformer Models] --> [Greeks/Validation Agent] --> [Final Valuation]
        ^                    |
        +---- rejected ------+   (up to 3 attempts)
```

1. **Market Data Agent** picks a ticker, history window and features from your query. It pulls daily data from Yahoo Finance (`yfinance`) and FRED, then cleans and aligns it into a feature panel.
2. **Schema Gate** re-checks the panel in code, looking at value ranges, calendar gaps, NaNs and minimum row count. If the panel fails, the errors go back to the agent so it can try again.
3. **Transformer Models** train small CPU-sized JAX versions of each model on the panel (chronological train/validation split, early stopping, then a refit on all data) and forecast `log_return` for `pred_len` trading days.
4. **Greeks/Validation Agent** sanity-checks each model's forecast and drops bad models. It prices near-the-money contracts, checks the Greeks for no-arbitrage violations, and looks up events such as earnings inside the horizon. It can also run Python on the run artifacts. It ends by submitting an approve/reject verdict.
5. **Final Valuation** values the contracts using the mean forecast of the accepted models and writes the results to disk.

## Repository layout

| Path | Contents |
| --- | --- |
| `pipeline/orchestration.py` | Entry point. Builds the LangGraph pipeline and provides the CLI. |
| `pipeline/models/` | JAX/Equinox implementations of the four transformers. Each file runs a smoke test on its own. |
| `pipeline/utils/forecasting.py` | Feature-panel construction, per-model fitting and forecasting. |
| `pipeline/utils/training.py` | Training loop, prediction, feature scaling and checkpoints. |
| `pipeline/utils/windows.py` | Sliding windows and leakage-safe train/validation/test splits. |
| `pipeline/evaluation/` | Ad-hoc evaluation CLI: metrics, baselines, plots and a markdown report. |
| `pipeline/tests/` | pytest suite. |
| `pipeline/utils/greeks.py` | Black-Scholes prices/Greeks, contract loading and sanity checks. |
| `pipeline/utils/schemas.py` | Pydantic schemas and the panel validation gate. |
| `pipeline/utils/agent_tools.py` | Tools exposed to the agents. |
| `pipeline/utils/prompts.py` | Agent system prompts. |
| `pipeline/skills/` | Short checklists the agents can load on demand. |
| `pipeline/utils/company_data.py`, `macro_indicators.py`, `load_data.py` | Data loaders for Yahoo Finance and FRED. |
| `long_sequence_forecasting_transformers_outlines.md` | Reference spec the model implementations follow. |
| `prediction CSVs/` | Saved model predictions from earlier experiments. |
| `valuation runs/` | Output of each pipeline run, in a timestamped folder. |
| `evaluation runs/`, `checkpoints/` | Evaluation output and saved models (git-ignored). |

## Setup

Requires Python 3.12.

```bash
pip install -r requirements.txt
```

JAX's CPU wheels install natively on Windows, macOS and Linux; no GPU is needed.

Create a `.env` file in the repo root (it is already in `.gitignore`):

```
ANTHROPIC_API_KEY=your-anthropic-key
FRED_API_KEY=your-fred-key
```

- `ANTHROPIC_API_KEY` is required to run the agents.
- `FRED_API_KEY` is required for the macro features (10Y yield, breakeven inflation, CPI). If the FRED pull fails, the market data agent can drop those features.

## Running the pipeline

Run from the `pipeline/` directory, passing a plain-English request:

```bash
cd pipeline
python orchestration.py "Value near-the-money AAPL options over the next 10 trading days"
```

Options:

| Flag | Default | Description |
| --- | --- | --- |
| `--model` | `claude-opus-5` | Claude model used by both agents. |
| `--models` | all four | Subset of transformers to train, e.g. `--models informer autoformer`. |
| `--pred-len` | `10` | Forecast horizon in trading days (1–60). |
| `--epochs` | `100` | Maximum training epochs; early stopping usually ends sooner. |
| `--batch-size` | `32` | Minibatch size. |
| `--lr` | `1e-3` | Peak learning rate (warmup then cosine decay). |
| `--weight-decay` | `1e-4` | AdamW weight decay. |
| `--patience` | `10` | Epochs without validation improvement before stopping. |
| `--val-frac` | `0.15` | Share of windows held out (newest data) for early stopping. |
| `--no-refit` | off | Skip retraining on train+validation before forecasting. |
| `--save-checkpoints` | off | Save each trained model to `checkpoints/<TICKER>/<model>/<timestamp>/`. |
| `--no-code` | off | Disables the validation agent's `run_python` tool. |

Example with a quicker run:

```bash
python orchestration.py "Price MSFT options two weeks out" --models informer fedformer --epochs 30
```

Agent progress is printed as it runs, and a JSON summary is printed at the end.

### Output

Each run writes to `valuation runs/<YYYYMMDD_HHMMSS>/`:

- `panel.csv`: the validated feature panel the models were trained on.
- `forecasts.csv`: each model's predicted log returns and implied closes.
- `valuation.csv`: one row per contract, with market mid, Black-Scholes price and Greeks today and at the horizon, a low/high value band across models, expected P&L and model-vs-market difference.
- `summary.json`: the verdict, the accepted models, horizon spot, and the issues and notes from the validation agent.

### Using it from Python

```python
from orchestration import run_pipeline
from utils.schemas import ForecastConfig

state, ctx = run_pipeline("Value AAPL options over 10 days", ForecastConfig(pred_len=10))
print(state["summary"])
ctx.valuation  # pandas DataFrame of valued contracts
```

## Running a single model

Each model file runs a forward pass on random inputs. It prints the output shape and checks that inference is deterministic:

```bash
cd pipeline
python -m models.informer
```

## How training works

`run_forecasts` (in `utils/forecasting.py`) trains each model as follows:

1. Cuts the panel into sliding windows. Each window has `seq_len` input days and the next `pred_len` days as the target.
2. Splits the windows chronologically. The newest `val_frac` become the validation set, with a purge gap of `pred_len` windows so training targets never overlap validation targets.
3. Standardises features using statistics from the training rows only.
4. Trains with AdamW on shuffled minibatches, using warmup and cosine learning-rate decay and gradient clipping. The loss weights the `log_return` column at `target_weight` (default 0.5) and spreads the rest over the other features. Each epoch is one jit-compiled `lax.scan`, so a run compiles once, and compiled programs are cached in `.jax_cache/` across runs.
5. Stops once validation MSE on `log_return` stops improving and restores the best weights.
6. Retrains from scratch on train+validation for the best epoch count, then forecasts. Disable this with `--no-refit`.

On a laptop CPU, training all four models on ~2 years of data takes roughly a minute.

## Evaluating the models

The evaluation CLI trains each model (or loads saved checkpoints) and scores it on held-out data. It compares each model against four baselines: random walk (zero return), historical mean, AR(1), and ridge regression on lagged returns. Run it from `pipeline/`:

```bash
cd pipeline
python -m evaluation --ticker AAPL --history-months 24                 # chronological 70/15/15 holdout
python -m evaluation --ticker AAPL --mode walkforward --folds 4        # expanding-window walk-forward
python -m evaluation --panel "../valuation runs/<run>/panel.csv"       # a saved panel, no network
python -m evaluation --ticker AAPL --save-checkpoints                  # also save the trained models
python -m evaluation --panel panel.csv --checkpoint ../checkpoints/AAPL/informer   # score a saved model
```

Pulled panels are cached in `evaluation runs/_cache/` for the day; pass `--refresh` to re-pull. Every `ForecastConfig` setting can be overridden (`--seq-len`, `--pred-len`, `--epochs`, `--lr`, ...). Run `python -m evaluation --help` for the full list.

Each run writes to `evaluation runs/<YYYYMMDD_HHMMSS>/`:

- `report.md`: a verdict per model, a ranked metrics table, efficiency stats and the plots.
- `metrics.json` / `metrics.csv`: per model and baseline:
  - RMSE, MAE and out-of-sample R² vs a zero forecast
  - next-day and cumulative directional hit rate
  - next-day information coefficient
  - cumulative-return error
  - implied-price RMSE/MAPE
  - Diebold-Mariano tests vs the random walk
  - training time, parameter count and prediction latency
- `per_horizon.csv` and `predictions.csv`: error by horizon step, and every forecast next to the actual value.
- `plots/`: loss curves, forecast vs actual (next day and cumulative), RMSE by horizon, scatter, residual histograms, a model comparison and implied close vs actual.

Daily returns are close to unpredictable, so the random walk is a hard benchmark. A model is only worth trusting if it beats it out of sample. Walk-forward mode gives the most reliable read.

## Tests

```bash
python -m pytest pipeline/tests -q
```

The suite covers output shapes, gradient flow, determinism, overfitting a small batch, layer and mask correctness, leakage-safe splits, early stopping, exact checkpoint round-trips, an end-to-end `run_forecasts`, and the metrics and baselines. The JAX ports were also checked against the original PyTorch implementations, matching to within 1e-6, before those were removed.

## Notes

- Models are kept small (`d_model=64`, 1–2 layers) so training finishes in seconds on CPU. The forecasts are meant for experimentation, not trading.
- Deviations from the reference implementations, all needed because the models process one sample at a time and batch with `vmap`:
  - Autoformer uses per-sample time-delay aggregation in both training and inference.
  - Pyraformer's coarser-scale convs and Informer's distilling layers (only built when `e_layers > 1`) use LayerNorm instead of BatchNorm.
  - FEDformer's decoder Fourier blocks are sized for `label_len + pred_len`; the reference used `seq_len // 2 + pred_len`, which is only correct when `label_len == seq_len / 2`.
- Pricing uses European Black-Scholes, but listed US equity options are American-style. The validation agent may flag the bias this causes in deep in-the-money puts.

## FRED API Key Use

https://fred.stlouisfed.org/docs/api/terms_of_use.html

*This product uses the FRED API but is not endorsed or certified by the Federal Reserve Bank of St. Louis.*
