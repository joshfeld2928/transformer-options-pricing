"""
LangChain tools for the pipeline agents.

Tools close over a shared ``RunContext`` so DataFrames stay in Python and only
compact JSON summaries go to the model. Every tool that accepts agent input
validates it against a schema from ``schemas.py`` before acting.
"""

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool
from pydantic import ValidationError

from .forecasting import build_feature_panel as _build_panel, forecasts_frame
from .greeks import greek_bound_violations, load_contracts, sanity_check_forecasts, value_contracts
from .schemas import ForecastConfig, MarketDataRequest, ValidationReport, validate_feature_panel

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"

# Anthropic server-side tools; run on Anthropic's infrastructure.
WEB_SEARCH = {"type": "web_search_20250305", "name": "web_search", "max_uses": 3}
WEB_FETCH = {"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 3}

_MAX_TOOL_OUTPUT = 4000
_CODE_TIMEOUT_S = 60


@dataclass
class RunContext:
    """Artifacts shared between agents and graph nodes for one run."""

    config: ForecastConfig
    output_dir: Path
    risk_free_rate: float = 0.04
    request: MarketDataRequest | None = None
    panel: pd.DataFrame | None = None
    forecasts: dict = field(default_factory=dict)
    train_losses: dict = field(default_factory=dict)
    contracts: pd.DataFrame | None = None
    valuation: pd.DataFrame | None = None
    validation: ValidationReport | None = None
    last_close: float | None = None

    def rate(self):
        """Latest 10Y yield from the panel as a decimal, else the default."""
        if self.panel is not None and "10Y_Treasury_Yield" in self.panel:
            return float(self.panel["10Y_Treasury_Yield"].iloc[-1]) / 100
        return self.risk_free_rate


def _errors(exc: ValidationError) -> str:
    return json.dumps({"ok": False, "errors": [
        f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()
    ]})


def _clip(text: str) -> str:
    return text if len(text) <= _MAX_TOOL_OUTPUT else text[:_MAX_TOOL_OUTPUT] + "\n...[truncated]"


def make_load_skill():
    @tool
    def load_skill(name: str) -> str:
        """Load a skill's instructions by name. Available: market_data, greeks_validation."""
        path = SKILLS_DIR / f"{Path(name).stem}.md"
        if not path.is_file():
            return f"No skill '{name}'. Available: {[p.stem for p in SKILLS_DIR.glob('*.md')]}"
        return path.read_text(encoding="utf-8")

    return load_skill


def market_data_tools(ctx: RunContext):
    @tool
    def build_feature_panel(ticker: str, history_months: int = 6, features: list[str] | None = None) -> str:
        """Pull, clean and align daily underlying + macro features for a ticker,
        then validate them for the transformers. Returns the validation report."""
        try:
            args = {"ticker": ticker, "history_months": history_months}
            if features is not None:
                args["features"] = features
            request = MarketDataRequest(**args)
        except ValidationError as exc:
            return _errors(exc)

        try:
            panel = _build_panel(request)
        except Exception as exc:  # noqa: BLE001 - reported back to the agent
            return json.dumps({"ok": False, "errors": [f"data pull failed: {exc}"]})

        report = validate_feature_panel(panel, request, ctx.config)
        if report.ok:
            ctx.request, ctx.panel = request, panel
        return report.model_dump_json()

    @tool
    def describe_panel() -> str:
        """Summary statistics of the current validated feature panel."""
        if ctx.panel is None:
            return "No validated panel yet; call build_feature_panel."
        return _clip(ctx.panel.describe().round(4).to_string())

    return [build_feature_panel, describe_panel, make_load_skill(), WEB_SEARCH]


def greeks_tools(ctx: RunContext, allow_code=True):
    @tool
    def get_forecasts() -> str:
        """Each model's forecast, training loss, and deterministic sanity checks."""
        summary = {
            name: {
                "horizon_close": round(f.implied_close[-1], 4),
                "cum_log_return": round(sum(f.predicted_log_return), 5),
                "train_loss": round(ctx.train_losses.get(name, float("nan")), 5),
            }
            for name, f in ctx.forecasts.items()
        }
        return json.dumps({
            "ticker": ctx.request.ticker,
            "last_close": round(ctx.last_close, 4),
            "horizon": next(iter(ctx.forecasts.values())).dates[-1],
            "models": summary,
            "sanity": sanity_check_forecasts(ctx.forecasts, ctx.last_close),
        })

    @tool
    def price_contracts(accepted_models: list[str], max_expiries: int = 2, moneyness_band: float = 0.1) -> str:
        """Black-Scholes value and Greeks of near-the-money contracts, today and
        at the forecast horizon, using the accepted models' mean forecast."""
        try:
            report = ValidationReport(approved=True, accepted_models=accepted_models,
                                      max_expiries=max_expiries, moneyness_band=moneyness_band)
        except ValidationError as exc:
            return _errors(exc)
        missing = set(report.accepted_models) - set(ctx.forecasts)
        if missing:
            return json.dumps({"ok": False, "errors": [f"no forecast for {sorted(missing)}"]})

        table = run_valuation(ctx, report)
        if table.empty:
            return "No contracts survive the horizon; widen max_expiries."
        cols = ["contract_symbol", "option_type", "strike", "mid_price", "price_now",
                "price_horizon", "delta_now", "gamma_now", "vega_now", "theta_now", "expected_pnl"]
        return _clip(f"{len(table)} contracts\n" + table[cols].round(4).to_string(index=False))

    @tool
    def check_greeks() -> str:
        """Deterministic no-arbitrage checks on the last priced table."""
        if ctx.valuation is None or ctx.valuation.empty:
            return "Nothing priced yet; call price_contracts."
        return json.dumps(greek_bound_violations(ctx.valuation))

    @tool
    def submit_validation(approved: bool, accepted_models: list[str], issues: list[str] | None = None,
                          notes: str = "", max_expiries: int = 2, moneyness_band: float = 0.1) -> str:
        """Submit the final verdict. Required exactly once before finishing."""
        try:
            report = ValidationReport(approved=approved, accepted_models=accepted_models, issues=issues or [],
                                      notes=notes, max_expiries=max_expiries, moneyness_band=moneyness_band)
        except ValidationError as exc:
            return _errors(exc)
        missing = set(report.accepted_models) - set(ctx.forecasts)
        if missing:
            return json.dumps({"ok": False, "errors": [f"no forecast for {sorted(missing)}"]})
        ctx.validation = report
        return "Submitted."

    tools = [get_forecasts, price_contracts, check_greeks, submit_validation, make_load_skill(), WEB_SEARCH, WEB_FETCH]
    if allow_code:
        tools.append(_run_python_tool(ctx))
    return tools


def _run_python_tool(ctx: RunContext):
    @tool
    def run_python(code: str) -> str:
        """Run Python in a subprocess. Working dir holds panel.csv, forecasts.csv
        and valuation.csv (if priced). Print what you need; 60s timeout."""
        write_artifacts(ctx)
        try:
            proc = subprocess.run([sys.executable, "-c", code], cwd=ctx.output_dir, capture_output=True,
                                  text=True, timeout=_CODE_TIMEOUT_S, env={**os.environ, "MPLBACKEND": "Agg"})
        except subprocess.TimeoutExpired:
            return f"Timed out after {_CODE_TIMEOUT_S}s."
        return _clip(f"exit {proc.returncode}\n{proc.stdout}\n{proc.stderr}".strip())

    return run_python


def run_valuation(ctx: RunContext, report: ValidationReport) -> pd.DataFrame:
    """Load contracts (cached per expiry count) and value them; stores the table."""
    cached = ctx.contracts is not None and ctx.contracts.attrs.get("max_expiries") == report.max_expiries
    if not cached:
        horizon = next(iter(ctx.forecasts.values())).dates[-1]
        ctx.contracts = load_contracts(ctx.request.ticker, ctx.last_close, min_expiry=horizon,
                                       max_expiries=report.max_expiries, moneyness_band=0.5)
        ctx.contracts.attrs["max_expiries"] = report.max_expiries
    contracts = ctx.contracts[ctx.contracts["moneyness"].between(1 - report.moneyness_band,
                                                                 1 + report.moneyness_band)]
    ctx.valuation = value_contracts(contracts, ctx.forecasts, report.accepted_models,
                                    as_of=ctx.panel.index[-1], spot=ctx.last_close, rate=ctx.rate())
    return ctx.valuation


def write_artifacts(ctx: RunContext):
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    if ctx.panel is not None:
        ctx.panel.to_csv(ctx.output_dir / "panel.csv")
    if ctx.forecasts:
        forecasts_frame(ctx.forecasts).to_csv(ctx.output_dir / "forecasts.csv", index=False)
    if ctx.valuation is not None:
        ctx.valuation.to_csv(ctx.output_dir / "valuation.csv", index=False)
