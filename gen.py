"""Synthetic panel generator, calibrated to the real panel's gross statistics.

Base model (zero-alpha):
    r_{i,t} = beta_i * f_m(t) + sum_k B_{ik} g_k(t) + sigma_i * eps_{i,t}
  - f_m: market factor with 2-state Markov regime-switching volatility
  - g_k: 3 latent factors, own regime processes
  - eps: Student-t(5), unit-scaled; sigma_i lognormal across names
  E[r_{i,t+1} | F_t] = 0 for every i, so the true Sharpe of ANY signal-driven
  dollar-neutral portfolio is zero by construction (vol is predictable; means are not).

Planted alpha (SYN-A): two mechanisms, each calibrated so that its ORACLE
signal — evaluated through the actual DSL/backtest pipeline, with its
implementation lag — earns a target annualized Sharpe on the train window:
  - canon-shaped: 5-day cross-sectional reversal (oracle: neg(ret(5)))
  - anti-canon:   LONG high idiosyncratic kurtosis, orthogonalized to a canon
                  block (oracle: cs_rank(kurt(15))). The literature, if anything,
                  prices lottery-like names the other way; this plants alpha
                  where the prior points away from it.
Calibration constants are stored in calibration_targets.json and fixed across panels.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

N_NAMES = 400
T_TRAIN = 1000
T_OOS = 250
K_LATENT = 3


def _regime_vol(T, rng, lo, hi, p_stay_lo=0.985, p_stay_hi=0.95):
    state = np.zeros(T, dtype=int)
    s = 0
    for t in range(T):
        state[t] = s
        u = rng.random()
        if s == 0 and u > p_stay_lo:
            s = 1
        elif s == 1 and u > p_stay_hi:
            s = 0
    return np.where(state == 0, lo, hi)


def base_returns(T, rng):
    """Zero-alpha panel returns (T x N_NAMES)."""
    vol_m = _regime_vol(T, rng, 0.008, 0.022)
    f_m = rng.standard_normal(T) * vol_m + 0.0002
    G = np.empty((T, K_LATENT))
    for k in range(K_LATENT):
        vk = _regime_vol(T, rng, 0.004, 0.010)
        G[:, k] = rng.standard_normal(T) * vk
    beta = rng.normal(1.0, 0.30, N_NAMES)
    B = rng.normal(0.0, 0.7, (N_NAMES, K_LATENT))
    sigma = np.exp(rng.normal(np.log(0.020), 0.35, N_NAMES))
    eps = rng.standard_t(5, (T, N_NAMES)) / np.sqrt(5 / 3)
    r = np.outer(f_m, beta) + G @ B.T + eps * sigma
    return np.clip(r, -0.5, 1.0)


def _px_from_r(r, rng):
    p0 = np.exp(rng.uniform(np.log(10), np.log(300), r.shape[1]))
    return pd.DataFrame(np.cumprod(1 + r, axis=0) * p0)


def _canon_block(px: pd.DataFrame):
    r1 = px.pct_change()
    feats = [
        px.pct_change(5),
        px.pct_change(20),
        r1.rolling(20).std(),
        r1.rolling(20).skew(),
    ]
    Z = []
    for f in feats:
        z = f.sub(f.mean(axis=1), axis=0).div(f.std(axis=1), axis=0)
        Z.append(z.fillna(0.0).values)
    return np.stack(Z, axis=-1)  # T x N x 4


def _anti_canon_score(px: pd.DataFrame):
    """cs-z of kurt(15), orthogonalized day-by-day to the canon block."""
    r1 = px.pct_change()
    k = r1.rolling(15).kurt()
    kz = k.sub(k.mean(axis=1), axis=0).div(k.std(axis=1), axis=0).fillna(0.0).values
    C = _canon_block(px)
    T = kz.shape[0]
    out = np.zeros_like(kz)
    for t in range(T):
        X = C[t]
        y = kz[t]
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ coef
        sd = res.std()
        out[t] = res / sd if sd > 0 else 0.0
    return out


def _reversal_score(px: pd.DataFrame):
    f = px.pct_change(5)
    z = f.sub(f.mean(axis=1), axis=0).div(f.std(axis=1), axis=0)
    return (-z).fillna(0.0).values


def make_panel(seed: int, kappa_rev: float = 0.0, kappa_anti: float = 0.0):
    """Generate one panel. Returns (px, split_idx). Plant is applied with the
    same information lag as the backtester (score at t affects return at t+LAG),
    so the oracle's achievable Sharpe already reflects implementation lag."""
    from backtest import LAG

    rng = np.random.default_rng(seed)
    T = T_TRAIN + T_OOS
    r = base_returns(T, rng)
    if kappa_rev or kappa_anti:
        px0 = _px_from_r(r, np.random.default_rng(seed + 1))
        boost = np.zeros_like(r)
        if kappa_rev:
            s = _reversal_score(px0)
            boost[LAG:] += kappa_rev * s[:-LAG]
        if kappa_anti:
            s = _anti_canon_score(px0)
            boost[LAG:] += kappa_anti * s[:-LAG]
        r = np.clip(r + boost, -0.5, 1.0)
    px = _px_from_r(r, np.random.default_rng(seed + 1))
    px.index = pd.RangeIndex(T)
    return px, T_TRAIN


def panel_stats(px: pd.DataFrame) -> dict:
    r1 = px.pct_change()
    corr_proxy = r1.corrwith(r1.mean(axis=1), axis=0)
    return {
        "names": px.shape[1],
        "days": px.shape[0],
        "ann_vol_median": float(r1.std().median() * np.sqrt(252)),
        "avg_corr_with_ew": float(corr_proxy.mean()),
        "daily_kurt_median": float(r1.kurt().median()),
    }


def calibrate_kappas(targets=(0.5, 1.0), n_panels=3, seed0=7000, verbose=True):
    """Find kappa for each mechanism/strength so the DSL-evaluated oracle earns
    the target annualized train-window Sharpe (averaged over n_panels)."""
    from dsl import Evaluator, parse
    from backtest import score

    oracles = {"rev": "neg(ret(5))", "anti": "cs_rank(kurt(15))"}

    def oracle_sr(mech, kappa, seed):
        kw = {"kappa_rev": kappa if mech == "rev" else 0.0,
              "kappa_anti": kappa if mech == "anti" else 0.0}
        px, split = make_panel(seed, **kw)
        ev = Evaluator(px)
        sig = ev.eval(parse(oracles[mech]))
        r1 = px.pct_change()
        return score(sig, r1, end=split - 1)["sharpe"]

    out = {}
    for mech in oracles:
        for tgt in targets:
            lo, hi = 0.0, 0.02
            for _ in range(8):
                mid = 0.5 * (lo + hi)
                srs = [oracle_sr(mech, mid, seed0 + i) for i in range(n_panels)]
                m = float(np.mean(srs))
                if m < tgt:
                    lo = mid
                else:
                    hi = mid
            out[f"{mech}_{tgt}"] = {"kappa": round(0.5 * (lo + hi), 6), "achieved_sr": m}
            if verbose:
                print(mech, tgt, out[f"{mech}_{tgt}"])
    return out


if __name__ == "__main__":
    cal = calibrate_kappas()
    with open("calibration_targets.json", "w") as f:
        json.dump(cal, f, indent=2)
    print(json.dumps(cal, indent=2))
