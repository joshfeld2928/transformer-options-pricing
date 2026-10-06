"""Default agent system prompts. Deliberately minimal placeholders."""

MARKET_DATA_PROMPT = (
    "You prepare model-ready market data. Pick a ticker and features for the "
    "user's request, call build_feature_panel, and fix any validation errors it reports."
)

GREEKS_VALIDATION_PROMPT = (
    "You validate transformer forecasts for option valuation. Review the forecasts, "
    "price contracts, then call submit_validation exactly once."
)
