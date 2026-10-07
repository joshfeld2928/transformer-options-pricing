# Transformer Options Pricing

Forecast-driven option valuation. Four long-sequence forecasting transformers (Informer, Autoformer, FEDformer, Pyraformer) are trained on a stock's recent price history and macro indicators. They predict daily log returns over a short horizon. The forecast spot price is then used to value near-the-money options with Black-Scholes and compute their Greeks. Transformer models were recreated from the following papers: 

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
3. **Transformer Models** train small CPU-sized versions of each model on the panel and forecast `log_return` for `pred_len` trading days.
4. **Greeks/Validation Agent** sanity-checks each model's forecast and drops bad models. It prices near-the-money contracts, checks the Greeks for no-arbitrage violations, and looks up events such as earnings inside the horizon. It can also run Python on the run artifacts. It ends by submitting an approve/reject verdict.
5. **Final Valuation** values the contracts using the mean forecast of the accepted models and writes the results to disk.

## Repository layout

| Path | Contents |
| --- | --- |
| `pipeline/orchestration.py` | Entry point. Builds the LangGraph pipeline and provides the CLI. |
| `pipeline/inference/` | PyTorch implementations of the four transformers. Each file can run on its own. |
| `pipeline/utils/forecasting.py` | Feature-panel construction, model training and forecasting. |
| `pipeline/utils/greeks.py` | Black-Scholes prices/Greeks, contract loading and sanity checks. |
| `pipeline/utils/schemas.py` | Pydantic schemas and the panel validation gate. |
| `pipeline/utils/agent_tools.py` | Tools exposed to the agents. |
| `pipeline/utils/prompts.py` | Agent system prompts. |
| `pipeline/skills/` | Short checklists the agents can load on demand. |
| `pipeline/utils/company_data.py`, `macro_indicators.py`, `load_data.py` | Data loaders for Yahoo Finance and FRED. |
| `long_sequence_forecasting_transformers_outlines.md` | Reference spec the model implementations follow. |
| `prediction CSVs/` | Saved model predictions from earlier experiments. |
| `valuation runs/` | Output of each pipeline run, in a timestamped folder. |

## Setup

Requires Python 3.12.

```bash
pip install torch numpy pandas scipy yfinance fredapi python-dotenv pydantic langchain langchain-anthropic langgraph
```

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
| `--epochs` | `8` | Training epochs per transformer. |
| `--no-code` | off | Disables the validation agent's `run_python` tool. |

Example with a quicker run:

```bash
python orchestration.py "Price MSFT options two weeks out" --models informer fedformer --epochs 4
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

state, ctx = run_pipeline("Value AAPL options over 10 days", ForecastConfig(pred_len=10, epochs=8))
print(state["summary"])
ctx.valuation  # pandas DataFrame of valued contracts
```

## Running a single model

Each transformer file runs one forward pass on random tensors. This is a quick way to check that a model builds and gives output of the right shape:

```bash
cd pipeline/inference
python informer_inference.py
```

## Notes

- Models are kept small (`d_model=64`, 1–2 layers) so training finishes in seconds on CPU. The forecasts are meant for experimentation, not trading.
- Pricing uses European Black-Scholes, but listed US equity options are American-style. The validation agent may flag the bias this causes in deep in-the-money puts.

## FRED API Key Use

https://fred.stlouisfed.org/docs/api/terms_of_use.html

*This product uses the FRED API but is not endorsed or certified by the Federal Reserve Bank of St. Louis.*
