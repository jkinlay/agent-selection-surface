"""The selection-plus-aggregation surface (SYN-0, zero true alpha).

Two pipelines can log the identical number of backtests and report very
different in-sample Sharpe ratios, depending on how many of the winners they
blend. This isolates that second axis, which no trial-count correction records.

Protocol: on zero-alpha panels, evaluate a fixed library of random expressions
(train and out-of-sample portfolio return series). Then, for each (N, k): draw N
expressions at random, keep the top k by TRAIN Sharpe, equal-weight them, and
record the composite's train and out-of-sample Sharpe. Truth is zero everywhere,
so every point of train Sharpe is selection.

Scoring is deliberately identical to reanalyze.composite_scores, which produced
every book number in results/books.csv: legs are scaled by their own train
volatility, combined by a NaN-skipping mean (so a leg contributes only on days it
is live), and the Sharpe is taken over the days the composite is live. An earlier
version intersected the live windows of the whole library, scoring the surface on
~440 train days against the books' ~940 and inflating every cell by roughly
0.3-1.0 Sharpe. Do not reintroduce that.

Usage: python3 aggregation.py [n_panels]
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

from dsl import Evaluator, random_expr
from backtest import portfolio_series, TRADING_DAYS
from panel import load_full

RESULTS = "results"
N_GRID = [25, 50, 100, 200, 400]
K_GRID = [1, 2, 3, 5, 8, 12, 20]
DRAWS = 400
LIBRARY = 500

warnings.filterwarnings("ignore", category=RuntimeWarning)


def build_library(panel_id, n=LIBRARY, seed=99):
    """Train/OOS portfolio return series for n random expressions, each already
    scaled by its own TRAIN volatility. NaN marks days the signal is not live."""
    px, split = load_full(panel_id)
    r1 = px.pct_change()
    ev = Evaluator(px)
    rng = np.random.default_rng(seed)
    tr, oo = [], []
    while len(tr) < n:
        e = random_expr(rng)
        try:
            pr = portfolio_series(ev.eval(e), r1)
            a, b = pr.iloc[:split], pr.iloc[split:]
            sd = a.dropna().std()
            if sd and sd > 0 and b.dropna().std() > 0 and a.notna().sum() > 400:
                tr.append((a / sd).values)
                oo.append((b / sd).values)
        except Exception:
            pass
    return np.array(tr), np.array(oo)


def _sharpe(x):
    x = x[np.isfinite(x)]
    if len(x) < 60:
        return np.nan
    sd = x.std()
    return float(x.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan


def surface(tr, oo, rng):
    """Sharpe of top-k-of-N equal-weight composites, train and out of sample."""
    sr_tr = np.array([_sharpe(row) for row in tr])
    ok = np.isfinite(sr_tr)
    tr, oo, sr_tr = tr[ok], oo[ok], sr_tr[ok]
    rows, M = [], len(tr)
    for N in N_GRID:
        if N > M:
            continue
        for k in K_GRID:
            if k > N:
                continue
            a, b = [], []
            for _ in range(DRAWS):
                idx = rng.choice(M, N, replace=False)
                top = idx[np.argsort(-sr_tr[idx])[:k]]
                a.append(_sharpe(np.nanmean(tr[top], axis=0)))
                b.append(_sharpe(np.nanmean(oo[top], axis=0)))
            rows.append({"N": N, "k": k,
                         "sr_train": float(np.nanmean(a)),
                         "sr_oos": float(np.nanmean(b)),
                         "sr_train_sd": float(np.nanstd(a))})
    return rows


if __name__ == "__main__":
    n_panels = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    rng = np.random.default_rng(2026)
    allrows = []
    for i in range(n_panels):
        pid = f"SYN0-{i:02d}"
        tr, oo = build_library(pid)
        for r in surface(tr, oo, rng):
            r["panel"] = pid
            allrows.append(r)
        print("done", pid, flush=True)
    df = pd.DataFrame(allrows)
    df.to_csv(os.path.join(RESULTS, "aggregation_surface_bypanel.csv"), index=False)
    agg = df.groupby(["N", "k"]).agg(
        sr_train=("sr_train", "mean"), sr_oos=("sr_oos", "mean"),
        sr_train_se=("sr_train", lambda s: s.std() / np.sqrt(len(s))),
        n_panels=("sr_train", "size")).reset_index()
    agg.to_csv(os.path.join(RESULTS, "aggregation_surface.csv"), index=False)
    print(agg.pivot(index="k", columns="N", values="sr_train").round(2).to_string())
    print("\nOOS (truth = 0):")
    print(agg.pivot(index="k", columns="N", values="sr_oos").round(2).to_string())
