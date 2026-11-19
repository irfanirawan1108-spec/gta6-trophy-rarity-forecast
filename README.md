# Leonida Rarity Forecaster — GTA VI trophy earn-rate forecast

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/irfanirawan1108-spec/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/irfanirawan1108-spec/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Forecasts every Grand Theft Auto VI PlayStation 5 trophy's Exophase earn rate at the case's
target, as five quantiles (5/25/50/75/95%) in percent, for the
[make.fun GTA VI trophy rarity bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).

## The idea

The house baseline says every rate stays where the last read put it. That is wrong in launch
week for a structural reason the rules themselves document: Exophase's tracked-player count
jumps at each refresh (638 → 3,868 at the first refresh), and every rate but the 100% one falls
when the denominator jumps, because the new players have not earned anything yet. A rate is a
ratio — earners over tracked players — and the two sides move on different clocks: the
denominator surges with sales while the numerator grows only as players actually play.

So `main.py` models the two sides separately and combines them, then blends with a logit-trend
fit:

- **Model A (numerator/denominator):** least-squares slopes of ln(tracked) and ln(earners)
  against time over the last 24 h of reads, extrapolated with growth that decays exponentially
  (2-day time constant — a launch surge cannot compound forever). Forecast rate =
  100 × earners(target) / tracked(target), earners capped at tracked.
- **Model B (logit trend):** least-squares line of logit(rate) against ln(1 + hours since
  release) over the last 24 h, extrapolated with the slope shrunk by H / (H + 1.5 days).
  The median is the logit-average of A and B.
- **Refresh prior (flat early history only):** when every read is identical and the checkpoint
  is before the documented first refresh (2026-11-19T04:30Z, tracked 638 → 3,868), the next
  refresh is a denominator jump, not a flat line: forecast tracked = 3,868 with earners growing
  5%/hour of play. This only fires on flat launch-night histories; a rich history (the final's
  ~2 days of reads) never triggers it.
- **Spread:** random walk 0.5·√days, plus the trend fit's prediction uncertainty, plus half the
  disagreement between the two models, plus 0.7 when the refresh prior fires; floored at 0.35,
  capped at 2.5 logit units.

Python 3.13, standard library only, no network, deterministic (two runs write byte-identical
output), ~0.1 s per case — far under the 8-second cap.

## Try it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
# writes out/p00.out, out/p01.out — one {"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}]} each
```

## Local checks (what was actually measured)

- Practice cases p00/p01: valid shape (55 rows, quantiles 0–100, non-decreasing), two runs
  byte-identical, ~0.06 s per run.
- **Refresh scenario** (the refresh the rules document: earners unchanged, tracked 638 → 3,868):
  this model scores pinball **0.16** against that scenario, the house baseline **1.69**
  (typical miss 0.02 vs 0.73 points, 90% coverage 0.98 vs 0.00). This is a scenario, not the
  practice answer — the practice answers live server-side.
- **Two synthetic final-like backtests** (self-made cases with growing tracked players, per-trophy
  earner growth and discrete refreshes, 4.5-day horizon): this model **0.26 vs house 0.32** and
  **0.20 vs house 0.26**. A 3,700-read dense case runs in 0.17 s.
- A flat-history backtest (train to 01:00Z, answer the unchanged 02:30Z read) scores 0.07 vs the
  house's 0.02 — the honest cost of the wider spread when genuinely nothing moves; the spread
  pays for itself the moment a refresh lands.
