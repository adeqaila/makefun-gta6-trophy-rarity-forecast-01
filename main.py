"""gta6-trophy-rarity-forecast-01 entry: forecast each GTA VI trophy's Exophase
earn rate at the case target as five quantiles (0.05/0.25/0.5/0.75/0.95).

Python 3.13, standard library only, no network, deterministic.

Why this shape.  A trophy's rate is earned-count / players-tracked, and both
counts move for two different reasons: players progress through the game
(rates rise) and new players arrive holding almost nothing yet (rates fall).
Over a week the rise saturates -- a trophy's rate approaches the share of
players who will ever earn it -- while a straight line in logit space would
run away to 100%.  So the median is built from a saturating curve, not a
trend line:

  * Ceiling curve (weight 0.6).  Per trophy, r(u) = c * sigmoid(a + b*u)
    with u = ln(1 + hours since launch).  The ceiling c is gridded from just
    above the highest observed rate to 99.99; for each candidate c, logit(r/c)
    is linear in u, so (a, b) come from weighted least squares and the c with
    the smallest residual sum of squares wins.  The history is first
    compressed to its distinct page states (Exophase refreshes every few
    hours, so hundreds of reads are a few dozen plateaus), and reads are
    weighted by recency (24h half-life).
  * Logit trend (weight 0.2).  logit(rate) against ln(1 + hours), same
    weights, same window.  The same family as the house curve baseline; it
    steadies the ceiling fit when a trophy's history is thin.
  * Persistence (weight 0.2).  The last read's rate, as a shrinkage anchor.

  * Stale-page guard.  Exophase refreshes on its own schedule; a page that
    has not changed for hours on launch week is a stale snapshot, not a still
    world (launch night: flat for 4+ hours, then players-tracked jumped
    638 -> 3,868 and every rate but the 100% one fell).  Trust in the fits is
    scaled by how much players-tracked actually moved over the history; when
    the history is flat the median is shifted down -- dilution: fresh players
    arrive holding almost nothing yet -- and the spread is widened a lot.
    When the history shows real movement the guard does nothing.

Spread: the house random-walk term (0.5 * sqrt(horizon in days), scaled by
0.85) combined in quadrature with the stale-page term and a refresh-lag term
(a steep trend means a few hours between the last refresh and the target is
a real logit move).  Clamped to [0.4, 2.2] logit units so a degenerate
history can neither collapse the quantiles nor blow them out.

Output: /output/<id>.out per case, {"trophies": [{"id": n, "q": [...]}]},
exactly one key, five non-decreasing quantiles in [0, 100].
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_SCORES = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Blend weights for the three median estimators (ceiling / trend / persistence).
W_CEIL, W_TREND, W_PERSIST = 0.6, 0.2, 0.2
# Recency half-life (hours) for the weighted fits.
HALF_LIFE_H = 24.0
# House random-walk volatility, logit units per sqrt(day), and our scale on it.
RW_SIGMA, RW_SCALE = 0.5, 0.85
# Stale-page guard: dilution shift per log unit of staleness, and spread floor.
DILUTION_SHIFT = 1.2
STALE_SPREAD = 1.0
# Refresh lag assumed at the target (hours) for the lag term.
LAG_HOURS = 6.0
# Spread clamp (logit units).
SPREAD_MIN, SPREAD_MAX = 0.40, 2.20


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit_pct(x):
    p = clamp_pct(x) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(l):
    if l >= 0:
        return 100.0 / (1.0 + math.exp(-l))
    e = math.exp(l)
    return 100.0 * e / (1.0 + e)


def parse_time(s):
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def wlsq_line(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, resid_sd) or None."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 3:
        return None
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for x, y, w in zip(xs, ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return None
    b = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws)) / sxx
    a = my - b * mx
    dof = max(len(xs) - 2, 1)
    sse = sum(w * (y - (a + b * x)) ** 2 for x, y, w in zip(xs, ys, ws))
    resid = math.sqrt(sse / sw * len(xs) / dof)
    return a, b, resid


def distinct_states(history):
    """Collapse consecutive reads with identical (players, rates) to one point.

    The page only changes at an Exophase refresh, so a run of identical reads
    is one observation, not many; keeping them all would let long plateaus
    dominate the fits and inflate the residuals.  A missing players-tracked
    carries no information and never counts as a change on its own.
    """
    states, prev = [], None
    last_p = None
    for h in history:
        p = h.get("players_tracked")
        if p:
            last_p = p
        key = (last_p, tuple(h["rates"]))
        if key != prev:
            states.append(h)
            prev = key
    return states


def ceiling_median(us, rates, ws, u_target, fallback):
    """Best ceiling-logistic fit r = c*sigmoid(a+b*u); returns logit median.

    For each candidate ceiling c, logit(r/c) is linear in u, so (a, b) come
    from weighted least squares and the c with the smallest weighted SSE wins.
    """
    pts = [(u, r, w) for u, r, w in zip(us, rates, ws) if r is not None and r > 0]
    if len(pts) < 3:
        return fallback, 0.0
    rmax = max(r for _, r, _ in pts)
    if rmax >= 99.9:
        return fallback, 0.0
    lo = math.log(max(rmax * 1.02, rmax + 0.05))
    hi = math.log(99.99)
    ceilings = [math.exp(lo + (hi - lo) * k / 24.0) for k in range(25)] + [99.99]
    best = None
    for c in ceilings:
        if c <= rmax:
            continue
        xs, ys, w2 = [], [], []
        ok = True
        for u, r, w in pts:
            f = r / c
            if f <= 0.0 or f >= 1.0:
                ok = False
                break
            xs.append(u)
            ys.append(math.log(f / (1.0 - f)))
            w2.append(w)
        if not ok:
            continue
        fit = wlsq_line(xs, ys, w2)
        if fit is None:
            continue
        a, b, _ = fit
        sse = sum(w * (y - (a + b * x)) ** 2 for x, y, w in zip(xs, ys, w2))
        if best is None or sse < best[0]:
            best = (sse, c, a, b)
    if best is None:
        return fallback, 0.0
    _, c, a, b = best
    pred = c / (1.0 + math.exp(-(a + b * u_target)))
    return logit_pct(min(max(pred, 0.01), 99.99)), b


def forecast_case(case):
    trophies = case["trophies"]
    history = sorted(case["history"], key=lambda h: parse_time(h["at"]))
    if not history:
        raise ValueError("case has no history")
    release = parse_time(case["game"]["release_date"]) if case.get("game", {}).get("release_date") else parse_time(history[0]["at"])
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    t_cp = (checkpoint - release).total_seconds() / 3600.0
    t_tg = (target - release).total_seconds() / 3600.0
    u_tg = math.log1p(max(t_tg, 0.0))
    horizon_days = max(case["horizon_minutes"], 1) / 1440.0

    states = distinct_states(history)
    times_h = [(parse_time(s["at"]) - release).total_seconds() / 3600.0 for s in states]
    us = [math.log1p(max(t, 0.0)) for t in times_h]
    ws = [0.5 ** ((t_cp - t) / HALF_LIFE_H) for t in times_h]
    last = states[-1]
    last_rates = last["rates"]

    # ---- stale-page guard: how much did players-tracked actually move? ----
    seen_p = [s.get("players_tracked") for s in states if s.get("players_tracked")]
    p_move = math.log(max(seen_p) / min(seen_p)) if len(seen_p) >= 2 and min(seen_p) > 0 else 0.0
    trust = min(1.0, p_move / 0.5)  # 0 when the page never moved, 1 after ~65% growth
    last_change_h = times_h[0]
    prev_p, prev_rates = None, None
    for h in history:
        p = h.get("players_tracked")
        changed = h["rates"] != prev_rates if prev_rates is not None else False
        if p and prev_p and p != prev_p:
            changed = True
        if changed:
            last_change_h = (parse_time(h["at"]) - release).total_seconds() / 3600.0
        if p:
            prev_p = p
        prev_rates = h["rates"]
    stale_h = max(t_cp - last_change_h, 0.0)
    dilution = (1.0 - trust) * DILUTION_SHIFT * math.log1p((stale_h + case["horizon_minutes"] / 60.0) / 2.0)
    dilution = min(dilution, 2.0)

    rw_spread = RW_SCALE * RW_SIGMA * math.sqrt(horizon_days)
    stale_var = ((1.0 - trust) * STALE_SPREAD) ** 2
    lag_u = math.log1p(max(t_tg + LAG_HOURS, 0.0)) - u_tg

    rows = []
    for i, trophy in enumerate(trophies):
        rates = [s["rates"][i] if i < len(s["rates"]) else None for s in states]
        last_rate = last_rates[i] if i < len(last_rates) else None
        if last_rate is None:
            known = [r for r in rates if r is not None]
            last_rate = known[-1] if known else 0.1
        m_persist = logit_pct(last_rate)

        m_ceil, _ = ceiling_median(us, rates, ws, u_tg, m_persist)

        pts = [(u, logit_pct(r), w) for u, r, w in zip(us, rates, ws) if r is not None]
        if len(pts) >= 3:
            fit = wlsq_line([p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts])
            if fit:
                m_trend = fit[0] + fit[1] * u_tg
                slope = fit[1]
            else:
                m_trend, slope = m_persist, 0.0
        else:
            m_trend, slope = m_persist, 0.0

        median = W_CEIL * m_ceil + W_TREND * m_trend + W_PERSIST * m_persist
        # Stale page: the true rates sit below the stale snapshot (dilution),
        # except for a trophy every tracked player already holds.
        if last_rate < 99.99:
            median -= dilution

        spread = math.sqrt(rw_spread ** 2 + stale_var + (abs(slope) * lag_u) ** 2)
        spread = min(max(spread, SPREAD_MIN), SPREAD_MAX)

        qs = [inv_logit_pct(median + z * spread) for z in Z_SCORES]
        qs = [min(100.0, max(0.0, q)) for q in qs]
        qs = sorted(qs)
        rows.append({"id": trophy["id"],
                     "q": [round(q, 4) for q in qs]})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
