To save anyone the time and cost of running this script, I have attached example input and output from the entire pipeline. Note agent feedback, analysis, and incorporation of current events.

*last updated 10/18/2026*

# *Example Terminal Input Command and Prompt*

python pipeline/orchestration.py "Value near-the-money AAPL options over the next 10 trading days"

# *Example Terminal Output*

[market_data]  Panel is built and validated. Here's where things stand.

## Feature panel: AAPL, ready for modeling

| | |
|---|---|
| Ticker | AAPL |
| Rows | 493 daily observations |
| Window | 2024-10-29 → 2026-10-16 (24 months) |
| Features | `log_return` (target), `underlying_close`, `realized_vol`, `VIX`, `10Y_Treasury_Yield` |
| Errors / warnings | None |

**One fix along the way:** my first build was rejected — I'd used common-sense names (`close`, `volume`, `realized_vol_20d`, `vix`, `dgs10`, `high_low_range`) that aren't in the allowed vocabulary. I remapped to the schema's names and dropped `volume` and `high_low_range` entirely, since the panel doesn't support them. The FRED-backed macro pulls (`VIX`, `10Y_Treasury_Yield`) succeeded, so no macro features had to be dropped.

**Panel characteristics relevant to near-the-money pricing:**
- Realized vol averages 26% but spans 9.5% → 78%, so the sample covers both calm and stressed regimes — good for a 10-day horizon.
- Daily log returns are well-behaved (σ ≈ 1.8%), with tails at -9.7% / +14.3%. Those extremes are almost certainly earnings-day gaps.
- Underlying ranged 171 → 343 over the window; the recent level matters for which strikes are actually near-the-money.
- One thing to verify before you trust rate-sensitive Greeks: `10Y_Treasury_Yield` comes through at 0.40–0.53, not 4.0–5.3. That's off by a factor of 10 from percent units. It's internally consistent so it won't hurt a learned model, but if rho or any discounting is computed directly from this column, confirm the scaling first.

## Flag: earnings likely sits inside your horizon

This is the one that matters most for valuing near-the-money options, and the sources disagree.

Ten trading days from today runs through roughly **October 30**.  Wall Street Horizon lists AAPL's next earnings as confirmed for Monday, November 2, 2026, aftermarket close , and  Market Chameleon agrees on November 2 AMC . But  Financhill has it scheduled for October 29 , and  EarningsToday also expects Thursday, October 29 .

So the horizon either **just excludes** (Nov 2) or **just includes** (Oct 29) the event. This isn't a detail you can average over:

- If earnings land on Oct 29, your 10-day terminal distribution is bimodal, not lognormal.  The market is pricing an earnings reaction of roughly ±4.66%, about $15.97  — a single-day move far outside the 1.8% daily σ in the panel.
- If earnings land Nov 2, the options still carry the event premium in their IV even though the event is past your horizon. A model trained on realized returns will read that elevated IV as a disagreement and show a large `model_vs_market` gap — which would be an artifact, not a signal.

The Oct 29 vs Nov 2 ambiguity is worth resolving against Apple's own investor-relations page before you commit to the valuation, since the two cases call for different treatment.

## Before you accept any forecast off this panel

Worth applying the standard checks: reject the model outright if returns come back extreme or suspiciously flat; confirm call deltas fall in [0, 1] and put deltas in [-1, 0]; confirm gamma and vega are non-negative. High terminal-close dispersion (>0.1) is expected here given the earnings overlap — treat it as a note rather than grounds for rejection. And flag any large `model_vs_market` reading, keeping the earnings-timing caveat above in mind as the likely explanation.

I've prepared the data but haven't run a valuation — say the word and I'll proceed, ideally once the earnings date is pinned down.
[schema_gate] ok=True 
Training autoformer on 350 windows (val 64)...
  autoformer: best epoch 2, val target MSE 0.913116455078125, 51.75s
Training fedformer on 350 windows (val 64)...
  fedformer: best epoch 5, val target MSE 0.9896928668022156, 64.19s
Training informer on 350 windows (val 64)...
  informer: best epoch 4, val target MSE 0.853641927242279, 50.38s
Training pyraformer on 350 windows (val 64)...
  pyraformer: best epoch 8, val target MSE 0.889700710773468, 69.51s
[greeks_validation] I've completed the validation. Summary of what I found:

## Accepted: all four models (no rejection criteria triggered)

Sanity checks were empty for autoformer, fedformer, informer, and pyraformer — no extreme or flat return paths. Terminal-close dispersion of 0.0535 is below the 0.1 threshold, so per the checklist it's a note rather than a rejection. The panel is sound: 493 rows over 24 months, `log_return` present, and the final close reconciles exactly with `last_close` of 343.46. Greeks came back completely clean (no delta/gamma/vega violations, no calls below intrinsic).

## Two material issues worth your attention

**1. The event check is the big one.**  Apple confirmed it will report Q4 FY2026 earnings on Monday, November 2, with CEO John Ternus and CFO Kevan Parekh discussing results on the call  —  Ternus's first earnings call after taking over from Tim Cook on September 1, 2026 . That date lands *after* the Oct 30 forecast horizon but *before* both available expiries (Nov 6 and Nov 13). So every contract's IV embeds an earnings jump the forecast path explicitly does not contain. The uniformly negative expected P&L (puts average -$3.83) is event-premium decay, not signal. Note some aggregators still list Oct 29 as the date;  at least one flags it as expected but not yet confirmed , and Apple's own IR page supersedes those.

**2. `model_vs_market` is large and one-sided.** All 54 contracts reprice *above* market mid (mean +$0.80, max +$1.61). Because calls *and* puts are both overpriced, the error is vega-positive — the listed IV is systematically richer than the mids imply, pointing to a stale or mismatched IV-vs-quote snapshot rather than arate or dividend sign error. This is why I'd use the output for relative/risk-surface work but not trust absolute price levels.

I also flagged that validation MSE (0.85–0.99 against a 1.00 naive baseline) implies R² of only ~15% down to ~1%, with fedformer statistically indistinguishable from forecasting zero — and a likely 10x scaling error in the `10Y_Treasury_Yield` column, which is harmless as a scaled feature but must not be passed through asthe discount rate.

My recommendation: re-run after re-syncing the IV and quote snapshots, and add an expiry on or before Oct 30 so at least one contract set is clean of the earnings jump.
{
  "ticker": "AAPL",
  "as_of": "2026-10-16",
  "last_close": 343.4599914550781,
  "approved": true,
  "accepted_models": [
    "autoformer",
    "fedformer",
    "informer",
    "pyraformer"
  ],
  "horizon_date": "2026-10-30",
  "horizon_spot": 347.0319137573242,
  "n_contracts": 54,
  "issues": [
    "model_vs_market is large AND one-sided: all 54/54 contracts reprice ABOVE the market mid (mean +$0.80, max +$1.61; 33 contracts >$0.50). Since both calls and puts are overpriced, the error is vega-positive, i.e. the listed impliedVolatility is systematically higher than the IV implied by the quoted mids (stale/mismatched IV-vs-quote snapshot) rather than a rate/dividend sign error. Absolute price_now levels should not be trusted; use relative/P&L comparisons only until the IVand quote snapshots are re-synced.",
    "Event-horizon mismatch: Apple's Q4 FY2026 earnings is confirmed for Mon Nov 2, 2026 (after market close), which falls AFTER the Oct 30 forecast horizon but BEFORE both available expiries (Nov 6 and Nov 13). Every priced contract's IV (mean 32-34%) embeds an earnings jump that the forecast path explicitly does not contain. Expected_pnl is therefore structurally biased negative (puts average -$3.83, calls -$0.60) as an artifact of decaying event-inflated IV, not as tradeable signal.",
    "Volatility inconsistency: listed IVs (26.8%-44.0%) are roughly double current realized vol (20d = 16.3%, panel realized_vol = 15.9%). Consistent with the earnings premium above, but it means the horizon repricing (which holds IV fixed) overstates remaining option value at Oct 30.",
    "Weak predictive skill: validation MSE on the standardized target is 0.85-0.99 against a naive zero-forecast baseline of 1.00, i.e. R-squared of only ~15% (informer) down to ~1% (fedformer). Fedformer (0.9897) is statistically indistinguishable from predicting zero return, and autoformer early-stopped at epoch 2 of 12. No model was rejected since none triggered a sanity failure, but the ensemble carries little directional information.",
    "Data hygiene (non-blocking): the 10Y_Treasury_Yield column averages 0.438 with a 0.395-0.531 range. If these are percent units this implies a ~0.44% 10Y yield, which is implausible for 2025-26; the series looks mis-scaled by 10x (should be ~4.4%). Harmless as a scaled model feature, but must not be passed through as the risk-free discount rate."
  ],
  "notes": "VALIDATION SUMMARY \u2014 AAPL, 10 trading days, spot 343.46, horizon 2026-10-30.\n\nPanel: 493 rows / 24 months (2024-10-29 to 2026-10-16), log_return target present, no gaps; final underlying_close (343.46) reconciles exactly with last_close. Meets the >=12-month minimum, so early stopping was active rather than fixed-epoch fallback (confirmed: all four models stopped before max epochs).\n\nForecasts: sanity checks empty for all four models - no extreme or flat return paths, so no model met the rejection criteria and all four were accepted into the ensemble. Terminal-close dispersion 0.0535 is below the 0.1 threshold, so it is recorded as a note, not a rejection; still, the terminal spread is wide in dollar terms (autoformer 336.44 bearish vs informer 354.80 bullish, a ~5.5% spread straddling spot). Ensemble horizon_spot of 347.03 was verified as the exact equal-weight mean of the four terminal closes (+1.04% over 10 days). Direction is 3-1 bullish but rests on low-skill models (see issues).\n\nGreeks: check_greeks fully clean - 0 call deltas outside [0,1], 0 put deltas outside [-1,0], 0 negative gamma, 0 negative vega, 0 calls below intrinsic. Deltas, gamma and vega all behave monotonically across the strike ladder, and tau_horizon is correctly shortened versus tau_now on all 54 contracts.\n\nCoverage: 54 contracts, 2 expiries (2026-11-06, 2026-11-13), moneyness 0.904-1.090, within the requested +/-10% band.\n\nBOTTOM LINE: Approved for the Greeks/risk-surface use case, with caveats. The two expiries available both straddle the Nov 2 earnings release while the forecast stops Oct 30, so the expected_pnl column should NOT be read as a trade signal - the uniformly negative put P&L is an event-premium decay artifact. Combined with the one-sided IV-vs-quote disagreement on 100% of contracts, I would re-run after re-syncing the IV and quote snapshots, and ideally add an expiry on or before Oct 30 so at least one contract set is clean of the earnings jump.",
  "output_dir": "..."
}