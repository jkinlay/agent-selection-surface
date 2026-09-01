"""Analysis part 2: bridge statistics, look-ahead screen, block bootstrap,
dollar haircut, canon-span regression, concept mapping.

Usage: python3 analyze2.py <step>
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

from analyze import (all_runs, arm_of, composite, implied_n, null_pool,
                     expected_max, li_ji_meff, read_run, RESULTS)
from dsl import Evaluator, parse
from backtest import portfolio_series, sharpe, TRADING_DAYS
from panel import load_full
from canon import CANON

RNG = np.random.default_rng(20260901)


def _trial_returns(run, px, r1, end, cap=120, rng=None):
    """Portfolio return series for up to `cap` of a run's trials (train window)."""
    rng = rng or np.random.default_rng(7)
    ev = Evaluator(px)
    exprs = list(run["trials"])
    if len(exprs) > cap:
        exprs = [exprs[i] for i in rng.choice(len(exprs), cap, replace=False)]
    cols = {}
    for e in exprs:
        try:
            pr = portfolio_series(ev.eval(parse(e)), r1, end=end)
            if pr.dropna().std() > 0:
                cols[e] = pr
        except Exception:
            pass
    return pd.DataFrame(cols)


def step_bridge():
    """Trial-process statistics: SYN-0 vs REAL for the agent arm (and OPT for contrast).
    These are the statistics the calibration consumes; matching them licenses transfer."""
    rows = []
    for r in all_runs():
        arm = arm_of(r["run"])
        if arm not in ("AGENT", "OPT-medium"):
            continue
        px, split = load_full(r["panel"])
        r1 = px.pct_change()
        R = _trial_returns(r, px, r1, end=split - 1).dropna(axis=1, how="all")
        R = R.loc[R.notna().all(axis=1)]
        if R.shape[1] < 10 or len(R) < 100:
            continue
        C = R.corr().values
        iu = np.triu_indices_from(C, 1)
        srs = np.array(list(r["trials"].values()))
        # adaptivity: mean trial SR by batch, first vs last third
        bm = [np.mean([x["sharpe"] for x in b["results"] if "sharpe" in x])
              for b in r["batches"] if any("sharpe" in x for x in b["results"])]
        rows.append({
            "run": r["run"], "arm": arm,
            "setting": "REAL" if r["panel"] == "REAL" else r["panel"].split("-")[0],
            "n_logged": len(r["trials"]),
            "mean_abs_rho": float(np.abs(C[iu]).mean()),
            "sd_trial_sr": float(srs.std()),
            "adaptivity": float(bm[-1] - bm[0]) if len(bm) > 1 else np.nan,
            "li_ji_meff": li_ji_meff(C),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "bridge.csv"), index=False)
    print(df.groupby(["arm", "setting"])[
        ["n_logged", "mean_abs_rho", "sd_trial_sr", "adaptivity", "li_ji_meff"]]
        .agg(["mean", "count"]).round(3).to_string())
    return df


def _block_bootstrap_index(n, block=21, rng=None):
    rng = rng or RNG
    idx = []
    while len(idx) < n:
        s = rng.integers(0, max(n - block, 1))
        idx.extend(range(s, min(s + block, n)))
    return np.array(idx[:n])


def step_bootstrap(block=21, reps=2000):
    """Joint moving-block bootstrap over the shared REAL OOS path.
    Resamples the same time blocks for all runs simultaneously."""
    px, split = load_full("REAL")
    r1 = px.pct_change()
    series = {}
    for r in all_runs():
        if r["panel"] != "REAL":
            continue
        argmax3 = sorted(r["trials"], key=r["trials"].get, reverse=True)[:3]
        c = composite(argmax3, px, r1, start=split, scale_window=split - 1)
        if c is not None:
            series[(arm_of(r["run"]), r["run"])] = c.values
    canon_c = composite(list(CANON.values()), px, r1, start=split, scale_window=split - 1)
    series[("CANON", "canon")] = canon_c.values
    n = min(len(v) for v in series.values())
    M = {k: v[-n:] for k, v in series.items()}
    arms = sorted({k[0] for k in M})
    draws = {a: [] for a in arms}
    for _ in range(reps):
        idx = _block_bootstrap_index(n, block)
        for a in arms:
            vals = [sharpe(pd.Series(v[idx])) for (aa, _), v in M.items() if aa == a]
            draws[a].append(np.mean(vals))
    out = {a: {"mean": float(np.mean(d)), "lo": float(np.percentile(d, 2.5)),
               "hi": float(np.percentile(d, 97.5))} for a, d in draws.items()}
    json.dump({"block": block, "reps": reps, "oos_days": int(n), "arms": out},
              open(os.path.join(RESULTS, f"bootstrap_b{block}.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))
    return out


def _continuations(px, split, M=200, block=21, demean=False, seed=11):
    """Block-bootstrap continuations of the TRAIN panel: futures the model cannot
    have seen. Returns a list of price panels of OOS length."""
    rng = np.random.default_rng(seed)
    r_train = px.iloc[:split].pct_change().iloc[1:]
    if demean:
        r_train = r_train - r_train.mean()
    n_oos = len(px) - split
    burn = 300
    out = []
    T = len(r_train)
    for _ in range(M):
        idx = []
        while len(idx) < burn + n_oos:
            s = rng.integers(0, T - block)
            idx.extend(range(s, s + block))
        idx = np.array(idx[:burn + n_oos])
        r = r_train.values[idx]
        p = pd.DataFrame(np.cumprod(1 + r, axis=0) * 100.0, columns=px.columns)
        out.append((p, burn))
    return out


def step_lookahead(M=120):
    """One-sided screen: Delta = (R_agent - S_agent) - (R_canon - S_canon).
    Delta > 0 is conservative evidence of pretraining look-ahead; Delta <= 0 uninformative."""
    px, split = load_full("REAL")
    r1 = px.pct_change()
    agent_sets = {}
    for r in all_runs():
        if r["panel"] != "REAL" or arm_of(r["run"]) != "AGENT":
            continue
        agent_sets[r["run"]] = sorted(r["trials"], key=r["trials"].get, reverse=True)[:3]
    real = {k: sharpe(composite(v, px, r1, start=split, scale_window=split - 1))
            for k, v in agent_sets.items()}
    real["canon"] = sharpe(composite(list(CANON.values()), px, r1, start=split,
                                     scale_window=split - 1))
    for demean in (False, True):
        conts = _continuations(px, split, M=M, demean=demean)
        syn = {k: [] for k in real}
        for p, burn in conts:
            pr1 = p.pct_change()
            for k, v in list(agent_sets.items()) + [("canon", list(CANON.values()))]:
                c = composite(v, p, pr1, start=burn, scale_window=burn - 1)
                if c is not None:
                    syn[k].append(sharpe(c))
        S = {k: float(np.mean(v)) for k, v in syn.items()}
        gaps = {k: real[k] - S[k] for k in real}
        deltas = {k: gaps[k] - gaps["canon"] for k in agent_sets}
        res = {"demean": demean, "M": M, "real": real, "syn_mean": S,
               "gap": gaps, "delta": deltas,
               "delta_mean": float(np.mean(list(deltas.values())))}
        json.dump(res, open(os.path.join(RESULTS,
                  f"lookahead_demean{int(demean)}.json"), "w"), indent=2)
        print(f"demean={demean}: canon gap {gaps['canon']:+.3f}, "
              f"agent gap mean {np.mean([gaps[k] for k in agent_sets]):+.3f}, "
              f"Delta mean {res['delta_mean']:+.3f}")
    return res


def step_canon_span():
    """Canon-spanned R^2 of agent reported composites (REAL), plus the same for
    random expressions as the null the August post's failed metric demands."""
    px, split = load_full("REAL")
    r1 = px.pct_change()
    ev = Evaluator(px)
    B = pd.concat({k: portfolio_series(ev.eval(parse(e)), r1).iloc[:split]
                   for k, e in CANON.items()}, axis=1).dropna()
    X = np.column_stack([np.ones(len(B)), B.values])
    rows = []
    for r in all_runs():
        if r["panel"] != "REAL" or arm_of(r["run"]) != "AGENT":
            continue
        a3 = sorted(r["trials"], key=r["trials"].get, reverse=True)[:3]
        ys = composite(a3, px, r1, end=split - 1, scale_window=split - 1).loc[B.index]
        m = ys.notna().values
        y = ys.values[m]
        Xm = X[m]
        beta, *_ = np.linalg.lstsq(Xm, y, rcond=None)
        resid = y - Xm @ beta
        r2 = 1 - resid.var() / y.var()
        alpha = beta[0] * TRADING_DAYS / (resid.std() * np.sqrt(TRADING_DAYS))
        rows.append({"run": r["run"], "canon_r2": float(r2),
                     "resid_ir": float(alpha)})
    pool = json.load(open(os.path.join(RESULTS, "null_pool_REAL.json")))
    rng = np.random.default_rng(3)
    rnd = [pool[i]["expr"] for i in rng.choice(len(pool), 40, replace=False)]
    r2s = []
    for e in rnd:
        try:
            ys = portfolio_series(ev.eval(parse(e)), r1).loc[B.index]
            m = ys.notna().values
            if m.sum() < 200 or np.nanstd(ys.values) == 0:
                continue
            y = ys.values[m]
            Xm = X[m]
            beta, *_ = np.linalg.lstsq(Xm, y, rcond=None)
            r2s.append(float(1 - (y - Xm @ beta).var() / y.var()))
        except Exception:
            pass
    df = pd.DataFrame(rows)
    out = {"agent": df.to_dict("records"),
           "agent_mean_r2": float(df.canon_r2.mean()),
           "random_mean_r2": float(np.mean(r2s)), "random_n": len(r2s)}
    json.dump(out, open(os.path.join(RESULTS, "canon_span.json"), "w"), indent=2)
    print(f"agent canon R2 mean {out['agent_mean_r2']:.3f}; "
          f"random-expression null {out['random_mean_r2']:.3f} (n={len(r2s)})")
    return out


def step_agreement():
    """Cross-run correlation of reported composites (REAL) — the August-post metric,
    recomputed here on this design's runs."""
    px, split = load_full("REAL")
    r1 = px.pct_change()
    cols = {}
    for r in all_runs():
        if r["panel"] != "REAL":
            continue
        a3 = sorted(r["trials"], key=r["trials"].get, reverse=True)[:3]
        cols[r["run"]] = composite(a3, px, r1, end=split - 1, scale_window=split - 1)
    C = pd.DataFrame(cols).corr()
    ag = [c for c in C.columns if c.startswith("AG-")]
    op = [c for c in C.columns if c.startswith("OPT")]
    def m(sub):
        v = C.loc[sub, sub].values
        iu = np.triu_indices_from(v, 1)
        return float(v[iu].mean()) if len(sub) > 1 else np.nan
    out = {"agent_mean_rho": m(ag), "opt_mean_rho": m(op),
           "n_agent": len(ag), "n_opt": len(op)}
    json.dump(out, open(os.path.join(RESULTS, "agreement.json"), "w"), indent=2)
    print(out)
    return out


if __name__ == "__main__":
    step = sys.argv[1]
    {"bridge": step_bridge, "boot": step_bootstrap, "look": step_lookahead,
     "span": step_canon_span, "agree": step_agreement}[step]()
