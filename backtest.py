"""Signal scoring: dollar-neutral rank-weighted L/S portfolio, 1-day implementation lag.

Convention (identical to the August 2026 post):
  - signal S_t is computed from closes up to and including day t;
  - the position is established at the close of day t+1;
  - it therefore first earns the close-to-close return of day t+2.
Implemented as: w_t = normalize(cs_rank(S)_{t-2}); r_p(t) = sum_i w_{i,t} * r_{i,t}.

Weights: cross-sectional rank in [-0.5, 0.5], demeaned (dollar-neutral),
scaled to unit gross exposure (0.5 long / 0.5 short).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAG = 2  # days between signal observation and first earning day
TRADING_DAYS = 252


def weights_from_signal(sig: pd.DataFrame) -> pd.DataFrame:
    """Rank-weight, demean, unit-gross. Rows with <20 valid names are zeroed."""
    rk = sig.rank(axis=1, pct=True) - 0.5
    rk = rk.sub(rk.mean(axis=1), axis=0)
    gross = rk.abs().sum(axis=1)
    w = rk.div(gross.where(gross > 0), axis=0).fillna(0.0)
    valid = sig.notna().sum(axis=1)
    w[valid < 20] = 0.0
    return w


def portfolio_returns(sig: pd.DataFrame, r1: pd.DataFrame) -> pd.Series:
    """Daily portfolio return. Days before the signal is warm (all-NaN rows, i.e.
    zero gross exposure) are returned as NaN, not 0, so that in-sample and
    out-of-sample Sharpe are computed on the same basis: a 250-day-lookback
    signal is not credited with ~250 days of exact zeros inside the train window."""
    w = weights_from_signal(sig).shift(LAG)
    pr = (w * r1).sum(axis=1, min_count=1)
    live = w.abs().sum(axis=1) > 0
    return pr.where(live)


def sharpe(pr: pd.Series) -> float:
    pr = pr.dropna()
    if len(pr) < 60:
        return 0.0
    sd = pr.std()
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(pr.mean() / sd * np.sqrt(TRADING_DAYS))


def rank_ic(sig: pd.DataFrame, r1: pd.DataFrame) -> float:
    """Mean daily Spearman correlation between lagged signal and returns."""
    s = sig.shift(LAG)
    srk = s.rank(axis=1)
    rrk = r1.rank(axis=1)
    sc = srk.sub(srk.mean(axis=1), axis=0)
    rc = rrk.sub(rrk.mean(axis=1), axis=0)
    num = (sc * rc).sum(axis=1)
    den = np.sqrt((sc**2).sum(axis=1) * (rc**2).sum(axis=1))
    ic = (num / den.replace(0, np.nan))
    return float(ic.mean(skipna=True))


def score(sig: pd.DataFrame, r1: pd.DataFrame, start=None, end=None) -> dict:
    """Score a signal over [start, end] (inclusive labels)."""
    pr = portfolio_returns(sig, r1)
    if start is not None or end is not None:
        pr = pr.loc[start:end]
        sig_w = sig.loc[start:end]
        r1_w = r1.loc[start:end]
    else:
        sig_w, r1_w = sig, r1
    return {
        "sharpe": round(sharpe(pr), 4),
        "ic": round(rank_ic(sig_w, r1_w), 5),
        "ann_vol": round(float(pr.std() * np.sqrt(TRADING_DAYS)), 5),
    }


def portfolio_series(sig: pd.DataFrame, r1: pd.DataFrame, start=None, end=None) -> pd.Series:
    pr = portfolio_returns(sig, r1)
    if start is not None or end is not None:
        pr = pr.loc[start:end]
    return pr
