"""Consolidated re-analysis on the corrected evaluation basis.

Two bases are used deliberately and are NOT interchangeable:

  * HARNESS basis (zero-padded pre-warm-up days) — what the searcher actually saw
    and selected on. All trial-count, null-pool and effective-N work stays here,
    self-consistently: the run logs and the random-expression pools were both
    produced by the harness.
  * EVALUATION basis (warm-start; pre-warm-up days are NaN, not 0) — how reported
    books are scored. In-sample and out-of-sample are then on the same footing,
    which the harness basis was not: a 250-day-lookback signal was previously
    credited with ~250 exact zeros inside the train window but none out of sample.

Selection always uses the harness-basis scores in the run logs (that is the
historical fact of what the searcher chose); only the scoring of the chosen books
uses the evaluation basis.

Usage: python3 reanalyze.py <step>
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

from analyze import all_runs, arm_of, RESULTS
from dsl import Evaluator, parse, complexity
from backtest import portfolio_series, sharpe, weights_from_signal, TRADING_DAYS
from panel import load_full
from canon import CANON

_CACHE = {}


def _panel(pid):
    if pid not in _CACHE:
        px, split = load_full(pid)
        _CACHE[pid] = (px, px.pct_change(), split, Evaluator(px))
    return _CACHE[pid]


def series_for(pid, exprs):
    px, r1, split, ev = _panel(pid)
    out = []
    for e in exprs:
        try:
            pr = portfolio_series(ev.eval(parse(e)), r1)
            if pr.dropna().std() > 0:
                out.append(pr)
        except Exception:
            pass
    return out, split


def composite_scores(pid, exprs):
    """Equal-weight composite of vol-scaled legs; scaling uses TRAIN data only."""
    ser, split = series_for(pid, exprs)
    if not ser:
        return None
    scaled = []
    for pr in ser:
        s = pr.iloc[:split].dropna().std()
        if s and s > 0:
            scaled.append(pr / s)
    if not scaled:
        return None
    comp = pd.concat(scaled, axis=1).mean(axis=1)
    return {"sr_is": sharpe(comp.iloc[:split]), "sr_oos": sharpe(comp.iloc[split:]),
            "series": comp, "split": split}


def _picks(r):
    """argmax-3 by the harness-basis score the searcher saw (primary rule)."""
    return sorted(r["trials"], key=r["trials"].get, reverse=True)[:3]


def turnover(pid, exprs):
    px, r1, split, ev = _panel(pid)
    ts = []
    for e in exprs:
        try:
            w = weights_from_signal(ev.eval(parse(e)))
            ts.append(float(w.diff().abs().sum(axis=1).iloc[:split].mean()))
        except Exception:
            pass
    return float(np.mean(ts)) if ts else np.nan


# --------------------------------------------------------------------------- main tables

def step_books():
    """Every completed run's reported book, scored on the evaluation basis."""
    rows = []
    for r in all_runs():
        pid = r["panel"]
        for rule, exprs in (("argmax", _picks(r)), ("chosen", r["report"]["top3_chosen"])):
            sc = composite_scores(pid, exprs)
            if sc is None:
                continue
            rows.append({
                "run": r["run"], "arm": arm_of(r["run"]), "panel": pid,
                "setting": "REAL" if pid == "REAL" else pid.split("-")[0],
                "ckpt": "B" if r["run"].startswith("AGB-") else "A",
                "rule": rule, "sr_is": sc["sr_is"], "sr_oos": sc["sr_oos"],
                "decay": sc["sr_is"] - sc["sr_oos"],
                "n_logged": len(r["trials"]),
                "legs": float(np.mean([_legs(e) for e in exprs])),
                "cplx": float(np.mean([complexity(parse(e)) for e in exprs])),
                "turnover": turnover(pid, exprs) if rule == "argmax" else np.nan,
            })
        print("  scored", r["run"], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "books.csv"), index=False)
    return df


def _legs(expr):
    ast = parse(expr)

    def cnt(a):
        if isinstance(a, int):
            return 0
        if a[0] == "add":
            return cnt(a[1]) + cnt(a[2])
        if a[0] in ("cs_z", "cs_rank", "neg", "delay", "ts_z", "abs_", "sign"):
            return max(1, cnt(a[1]))
        return 1

    return cnt(ast)


def step_canon_composite():
    """The no-selection control, scored EXACTLY like every other book:
    a single equal-weight composite of the 12 published anomalies."""
    sc = composite_scores("REAL", list(CANON.values()))
    ind = []
    px, r1, split, ev = _panel("REAL")
    for name, e in CANON.items():
        pr = portfolio_series(ev.eval(parse(e)), r1)
        ind.append({"name": name, "expr": e, "sr_train": sharpe(pr.iloc[:split]),
                    "sr_oos": sharpe(pr.iloc[split:])})
    d = pd.DataFrame(ind)
    out = {"composite_sr_is": sc["sr_is"], "composite_sr_oos": sc["sr_oos"],
           "individual_mean_train": float(d.sr_train.mean()),
           "individual_mean_oos": float(d.sr_oos.mean()),
           "individual_oos_se": float(d.sr_oos.std() / np.sqrt(len(d))),
           "individual": ind,
           "turnover": turnover("REAL", list(CANON.values()))}
    json.dump(out, open(os.path.join(RESULTS, "canon_composite.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "individual"}, indent=2))
    return out


def step_emax():
    """The textbook benchmark: E[max Sharpe] over N iid draws of this grammar on
    zero-alpha data, at the agent's realized median logged trial count."""
    from analyze import expected_max, null_pool
    L = pd.read_csv(os.path.join(RESULTS, "run_logs.csv"))
    out = {}
    for setting, pid in (("SYN0", "SYN0-00"), ("REAL", "REAL")):
        pool = null_pool(pid)
        sub = L[(L.arm == "AGENT") & (L.panel.str.startswith(setting.replace("REAL", "REAL")))]
        n_med = float(sub.n_logged.median()) if len(sub) else np.nan
        out[setting] = {"pool_panel": pid, "pool_n": int(len(pool)),
                        "pool_max": float(pool.max()),
                        "agent_median_n_logged": n_med,
                        "E_max_at_agent_N": expected_max(pool, n_med),
                        "E_max_at_1000": expected_max(pool, 1000)}
    # censoring rate by arm
    L["censored"] = L.n_star >= 1e7
    out["censoring_rate_by_arm"] = {a: {"censored": int(g.censored.sum()), "n": int(len(g)),
                                        "rate": round(float(g.censored.mean()), 3)}
                                    for a, g in L.groupby("arm")}
    json.dump(out, open(os.path.join(RESULTS, "emax.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))
    return out


def step_deep_pool(depth=6, n=1500, pid="SYN0-00", seed=555):
    """Control demanded by review: a null pool drawn from DEEPER random expressions,
    matching the agent's structural complexity rather than the shallow prior."""
    from dsl import random_expr
    rng = np.random.default_rng(seed)
    px, r1, split, ev = _panel(pid)
    out = []
    while len(out) < n:
        e = random_expr(rng, max_depth=depth)
        try:
            pr = portfolio_series(ev.eval(e), r1).iloc[:split]
            if pr.dropna().std() > 0:
                out.append({"sharpe": round(sharpe(pr), 4),
                            "cplx": complexity(e), "legs": _legs_ast(e)})
        except Exception:
            pass
    s = np.array([r["sharpe"] for r in out])
    res = {"depth": depth, "n": n, "panel": pid, "max": float(s.max()),
           "p99": float(np.percentile(s, 99)), "p95": float(np.percentile(s, 95)),
           "mean_cplx": float(np.mean([r["cplx"] for r in out])),
           "mean_legs": float(np.mean([r["legs"] for r in out]))}
    json.dump({"summary": res, "pool": out},
              open(os.path.join(RESULTS, f"deep_pool_d{depth}_{pid}.json"), "w"))
    print(json.dumps(res, indent=2))
    return res


def _legs_ast(a):
    if isinstance(a, int):
        return 0
    if a[0] == "add":
        return _legs_ast(a[1]) + _legs_ast(a[2])
    if a[0] in ("cs_z", "cs_rank", "neg", "delay", "ts_z", "abs_", "sign"):
        return max(1, _legs_ast(a[1]))
    return 1


if __name__ == "__main__":
    step = sys.argv[1]
    {"books": step_books, "canon": step_canon_composite, "emax": step_emax,
     "deep": step_deep_pool}[step]()
