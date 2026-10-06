---
name: greeks_validation
description: Checklist for accepting forecasts and sanity-checking Greeks.
---

- Reject a model with sanity issues (extreme or flat returns).
- High terminal-close dispersion (> 0.1) is a note, not a rejection.
- Calls: delta in [0, 1]; puts: [-1, 0]; gamma and vega >= 0.
- Large `model_vs_market` means the listed IV and quote disagree; flag it.
- Check for earnings or events inside the horizon with web search.
