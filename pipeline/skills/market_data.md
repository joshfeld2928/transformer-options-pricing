---
name: market_data
description: Feature choices and fixes for build_feature_panel errors.
---

- Always include `log_return` (the target).
- "rows < N needed": raise `history_months`.
- Macro features need `FRED_API_KEY`; drop them if the FRED pull fails.
- Unsure a symbol is valid or listed: web search before building.
