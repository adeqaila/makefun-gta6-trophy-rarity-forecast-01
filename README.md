[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/adeqaila/makefun-gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/adeqaila/makefun-gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecast — Week One

Forecasts how many GTA VI players earn each trophy in week one (target 2026-11-26).

Model (`main.py`, Python 3.13 stdlib only, deterministic, ~0,05 s/case):
median per trophy is a 0,6/0,2/0,2 blend of a ceiling-logistic curve
(rate saturates toward a fitted ceiling), a recency-weighted logit trend,
and persistence on the latest observed rate, with a stale-page guard that
dilutes the median when the source page stops moving.

Entry for the [gta6-trophy-rarity-forecast-01 bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) on make.fun. See `BUILD_NOTES.md` for the full model write-up and local validation.
