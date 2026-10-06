"""
Pydantic schemas and hard validation gates for the agent pipeline.

Anything an agent hands to the transformer models passes through
``MarketDataRequest`` (the arguments) and ``validate_feature_panel`` (the
resulting data). The gate is enforced in code, never left to the LLM.
"""

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

MODEL_NAMES = ("autoformer", "fedformer", "informer", "pyraformer")
ModelName = Literal["autoformer", "fedformer", "informer", "pyraformer"]

UNDERLYING_FEATURES = ("log_return", "underlying_close", "realized_vol")
MACRO_FEATURES = (
    "VIX",
    "10Y_Treasury_Yield",
    "Daily_Breakeven_Inflation",
    "Official_Inflation_YoY",
)
ALLOWED_FEATURES = UNDERLYING_FEATURES + MACRO_FEATURES
DEFAULT_FEATURES = list(ALLOWED_FEATURES)
TARGET = "log_return"

# Plausible value ranges per feature: (low, high), both exclusive.
_FEATURE_BOUNDS = {
    "log_return": (-0.5, 0.5),
    "underlying_close": (0.0, 1e7),
    "realized_vol": (-1e-9, 5.0),
    "VIX": (0.0, 200.0),
    "10Y_Treasury_Yield": (-5.0, 30.0),
    "Daily_Breakeven_Inflation": (-5.0, 20.0),
    "Official_Inflation_YoY": (-20.0, 50.0),
}

# Fri -> Tue over a long weekend is 4 days; anything past a week is a hole.
_MAX_CALENDAR_GAP_DAYS = 7
# Training windows required beyond one seq_len + pred_len span.
MIN_TRAINING_WINDOWS = 20


class PanelValidationError(ValueError):
    """Raised when a feature panel fails the pre-model gate."""


class MarketDataRequest(BaseModel):
    """Arguments the market data agent may request."""

    ticker: str = Field(pattern=r"^[A-Z][A-Z0-9.\-]{0,9}$")
    history_months: int = Field(6, ge=3, le=60)
    features: list[str] = Field(default_factory=lambda: list(DEFAULT_FEATURES))

    @field_validator("ticker", mode="before")
    @classmethod
    def _upper(cls, v):
        return v.strip().upper() if isinstance(v, str) else v

    @field_validator("features")
    @classmethod
    def _known_features(cls, v):
        unknown = sorted(set(v) - set(ALLOWED_FEATURES))
        if unknown:
            raise ValueError(f"unknown features {unknown}; allowed: {list(ALLOWED_FEATURES)}")
        if TARGET not in v:
            raise ValueError(f"features must include the target '{TARGET}'")
        if len(set(v)) != len(v):
            raise ValueError("features contain duplicates")
        return v


class ForecastConfig(BaseModel):
    """Window sizes and training settings for the transformer step."""

    models: list[ModelName] = Field(default_factory=lambda: list(MODEL_NAMES), min_length=1)
    seq_len: int = Field(60, ge=16, le=512)
    label_len: int = Field(30, ge=1)
    pred_len: int = Field(10, ge=1, le=60)
    epochs: int = Field(8, ge=1, le=500)
    lr: float = Field(1e-3, gt=0, lt=1)
    seed: int = 42

    @model_validator(mode="after")
    def _label_fits(self):
        if self.label_len > self.seq_len:
            raise ValueError("label_len must not exceed seq_len")
        return self

    @property
    def min_rows(self) -> int:
        return self.seq_len + self.pred_len + MIN_TRAINING_WINDOWS - 1


class PanelReport(BaseModel):
    ok: bool
    n_rows: int
    start: str | None = None
    end: str | None = None
    columns: list[str] = []
    errors: list[str] = []
    warnings: list[str] = []


def validate_feature_panel(
    panel: pd.DataFrame, request: MarketDataRequest, config: ForecastConfig
) -> PanelReport:
    """Check a feature panel is fit to feed the transformers. Never raises."""
    errors, warnings = [], []

    if not isinstance(panel, pd.DataFrame) or panel.empty:
        return PanelReport(ok=False, n_rows=0, errors=["panel is empty or not a DataFrame"])

    idx = panel.index
    if not isinstance(idx, pd.DatetimeIndex):
        errors.append("index must be a DatetimeIndex")
    else:
        if not idx.is_monotonic_increasing:
            errors.append("index is not sorted ascending")
        if idx.has_duplicates:
            errors.append(f"{int(idx.duplicated().sum())} duplicate dates")
        if len(idx) > 1:
            max_gap = int(idx.to_series().diff().dt.days.max())
            if max_gap > _MAX_CALENDAR_GAP_DAYS:
                errors.append(f"calendar gap of {max_gap} days (max {_MAX_CALENDAR_GAP_DAYS})")
        staleness = (pd.Timestamp.today().normalize() - idx.max()).days
        if staleness > _MAX_CALENDAR_GAP_DAYS:
            warnings.append(f"latest row is {staleness} days old")

    if list(panel.columns) != list(request.features):
        errors.append(f"columns {list(panel.columns)} != requested {list(request.features)}")

    non_numeric = [c for c in panel.columns if not pd.api.types.is_numeric_dtype(panel[c])]
    if non_numeric:
        errors.append(f"non-numeric columns: {non_numeric}")
    else:
        values = panel.to_numpy(dtype=float)
        bad = ~np.isfinite(values)
        if bad.any():
            cols = panel.columns[bad.any(axis=0)].tolist()
            errors.append(f"{int(bad.sum())} NaN/inf values in {cols}")

        for col in panel.columns:
            low, high = _FEATURE_BOUNDS.get(col, (-np.inf, np.inf))
            series = panel[col].dropna()
            out = series[(series <= low) | (series >= high)]
            if len(out):
                errors.append(f"{col}: {len(out)} values outside ({low}, {high})")
            if series.std() == 0:
                warnings.append(f"{col} is constant")

        # Alignment check: the return must match the close it came from.
        if {"log_return", "underlying_close"} <= set(panel.columns) and len(panel) > 1:
            implied = np.log(panel["underlying_close"]).diff().iloc[1:]
            drift = (implied - panel["log_return"].iloc[1:]).abs().max()
            if drift > 1e-6:
                errors.append(f"log_return misaligned with underlying_close (max diff {drift:.2e})")

    if len(panel) < config.min_rows:
        errors.append(
            f"{len(panel)} rows < {config.min_rows} needed for seq_len={config.seq_len}, "
            f"pred_len={config.pred_len}; raise history_months"
        )

    has_dates = isinstance(idx, pd.DatetimeIndex) and len(idx)
    return PanelReport(
        ok=not errors,
        n_rows=len(panel),
        start=str(idx.min().date()) if has_dates else None,
        end=str(idx.max().date()) if has_dates else None,
        columns=list(panel.columns),
        errors=errors,
        warnings=warnings,
    )


def require_valid_panel(panel, request, config) -> PanelReport:
    """``validate_feature_panel`` that raises ``PanelValidationError`` on failure."""
    report = validate_feature_panel(panel, request, config)
    if not report.ok:
        raise PanelValidationError("; ".join(report.errors))
    return report


class ModelForecast(BaseModel):
    """One model's forecast; rejects non-finite or wrong-length output."""

    model: ModelName
    dates: list[str]
    predicted_log_return: list[float]
    implied_close: list[float]

    @model_validator(mode="after")
    def _consistent(self):
        n = len(self.dates)
        if n == 0 or n != len(self.predicted_log_return) or n != len(self.implied_close):
            raise ValueError("forecast arrays must be non-empty and equal length")
        if not np.isfinite(self.predicted_log_return + self.implied_close).all():
            raise ValueError("forecast contains NaN/inf")
        if min(self.implied_close) <= 0:
            raise ValueError("implied_close must be positive")
        return self


class ValidationReport(BaseModel):
    """Verdict submitted by the Greeks/validation agent."""

    approved: bool
    accepted_models: list[ModelName] = Field(min_length=1)
    moneyness_band: float = Field(0.1, gt=0, le=0.5)
    max_expiries: int = Field(2, ge=1, le=12)
    issues: list[str] = []
    notes: str = ""
