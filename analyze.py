"""Analysis: trial counts, implied effective N, overfit calibration, decay,
look-ahead screen, bridge statistics, dollar haircut.

Everything downstream of the LLM calls runs deterministically from seeds.
Usage: python3 analyze.py <step>
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

from dsl import Evaluator, parse, complexity
from backtest import portfolio_series, sharpe, TRADING_DAYS
from panel import load_full, load_train

RUNS = "runs"
RESULTS = "results"
RNG = np.random.default_rng(20260901)


# --------------------------------------------------------------------------- log parsing

def read_run(run_id):
    d = os.path.join(RUNS, run_id)
    cfg = json.load(open(os.path.join(d, "config.json")))
    trials, batches, journals = {}, [], []
    for line in open(os.path.join(d, "log.jsonl")):
        rec = json.loads(line)
        if rec.get("kind") == "eval_batch":
            batches.append(rec)
            for r in rec["results"]:
                if "sharpe" in r:
                    trials.setdefault(r["expr"], r["sharpe"])
        elif rec.get("kind") == "journal":
            journals.append(rec["text"])
    rep_p = os.path.join(d, "report.json")
    rep = json.load(open(rep_p)) if os.path.exists(rep_p) else None
    return {"run": run_id, "arm": cfg["arm"], "panel": cfg["panel"],
            "trials": trials, "batches": batches, "report": rep, "journals": journals}


def all_runs(prefix=None):
    out = []
    for r in sorted(os.listdir(RUNS)):
        if r.startswith(("SMOKE", "ABANDONED")):
            continue
        if prefix and not r.startswith(prefix):
            continue
        if os.path.exists(os.path.join(RUNS, r, "report.json")):
            out.append(read_run(r))
    return out


def arm_of(run_id):
    if run_id.startswith("AG-") or run_id.startswith("AGB-"):
        return "AGENT"
    return read_run(run_id)["arm"]


# --------------------------------------------------------------------------- effective N

def null_pool(panel_id):
    p = os.path.join(RESULTS, f"null_pool_{panel_id}.json")
    return np.array([r["sharpe"] for r in json.load(open(p))])


def expected_max(pool, n):
    """E[max of n iid draws from the empirical null pool] — exact order-statistic form.

    For the empirical distribution on sorted values x_(1)..x_(m), each with mass 1/m,
    P(max <= x_(k)) = (k/m)^n, so E[max] = sum_k x_(k) [(k/m)^n - ((k-1)/m)^n].
    Exact and O(m); no simulation, no memory blowup.
    """
    x = np.sort(np.asarray(pool, dtype=float))
    m = len(x)
    k = np.arange(1, m + 1)
    w = np.power(k / m, n) - np.power((k - 1) / m, n)
    return float(np.dot(x, w))


# The empirical pool caps what this estimator can express: E[max] -> max(pool) as
# n -> infinity, so an observed best above the pool maximum is not identified upward.
N_CAP = 1e7


def implied_n(best_sr, pool, lo=1.0, hi=N_CAP):
    """Smallest N whose expected max matches the observed best IS Sharpe.
    Returns N_CAP (censored) when the observed best exceeds what the null pool
    can express — reported as a lower bound, never as a point estimate."""
    if best_sr <= expected_max(pool, 1):
        return 1.0
    if best_sr >= expected_max(pool, hi):
        return float(N_CAP)
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        if expected_max(pool, mid) < best_sr:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def li_ji_meff(corr):
    """Li & Ji (2005) eigenvalue-based effective number of independent tests."""
    ev = np.linalg.eigvalsh(corr)
    ev = np.clip(ev, 0, None)
    return float(sum((e >= 1) + (e - np.floor(e)) for e in ev if e > 0))


# --------------------------------------------------------------------------- portfolios

def composite(exprs, px, r1, start=None, end=None, scale_window=None):
    """Equal-weight composite of signals, each scaled to equal vol on the scale window."""
    ev = Evaluator(px)
    series = []
    for e in exprs:
        pr = portfolio_series(ev.eval(parse(e)), r1)
        s = pr.loc[:scale_window].std() if scale_window is not None else pr.std()
        if s and not np.isnan(s) and s > 0:
            series.append(pr / s)
    if not series:
        return None
    comp = pd.concat(series, axis=1).mean(axis=1)
    return comp.loc[start:end] if (start is not None or end is not None) else comp


# --------------------------------------------------------------------------- step 1

def step_logs():
    """Trial counts and effective N for every completed run."""
    rows = []
    for r in all_runs():
        pool = null_pool(r["panel"])
        srs = np.array(list(r["trials"].values()))
        if len(srs) == 0:
            continue
        best = float(srs.max())
        rep = r["report"]
        chosen_sr = max(rep["chosen_sr_train"]) if rep["chosen_sr_train"] else np.nan
        rows.append({
            "run": r["run"], "arm": arm_of(r["run"]), "panel": r["panel"],
            "n_logged": len(r["trials"]), "n_batches": len(r["batches"]),
            "best_is_sr": best, "chosen_best_sr": chosen_sr,
            "n_star": implied_n(best, pool),
            "n_star_chosen": implied_n(chosen_sr, pool) if not np.isnan(chosen_sr) else np.nan,
            "null_p95": float(np.percentile(pool, 95)),
            "mean_complexity": float(np.mean([complexity(parse(e)) for e in list(r["trials"])[:400]])),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "run_logs.csv"), index=False)
    print(df.groupby("arm")[["n_logged", "best_is_sr", "n_star"]].agg(["mean", "median", "count"]).round(2))
    return df


# --------------------------------------------------------------------------- step 2

def step_syn_decay():
    """SYN-0 and SYN-A: IS vs OOS of reported composites (argmax-3 and chosen-3)."""
    rows = []
    for r in all_runs():
        if r["panel"] == "REAL":
            continue
        px, split = load_full(r["panel"])
        r1 = px.pct_change()
        chosen = r["report"]["top3_chosen"]
        argmax3 = sorted(r["trials"], key=r["trials"].get, reverse=True)[:3]
        for label, exprs in (("chosen", chosen), ("argmax", argmax3)):
            c_is = composite(exprs, px, r1, end=split - 1, scale_window=split - 1)
            c_oos = composite(exprs, px, r1, start=split, scale_window=split - 1)
            if c_is is None or c_oos is None:
                continue
            rows.append({"run": r["run"], "arm": arm_of(r["run"]), "panel": r["panel"],
                         "setting": r["panel"].split("-")[0], "rule": label,
                         "sr_is": sharpe(c_is), "sr_oos": sharpe(c_oos),
                         "decay": sharpe(c_is) - sharpe(c_oos)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "syn_decay.csv"), index=False)
    print(df[df.rule == "argmax"].groupby(["setting", "arm"])[["sr_is", "sr_oos", "decay"]]
          .agg(["mean", "count"]).round(3).to_string())
    return df


# --------------------------------------------------------------------------- step 3

def step_real():
    """REAL: reported composites train vs OOS; canon placebo; agreement."""
    px, split = load_full("REAL")
    r1 = px.pct_change()
    rows = []
    for r in all_runs():
        if r["panel"] != "REAL":
            continue
        chosen = r["report"]["top3_chosen"]
        argmax3 = sorted(r["trials"], key=r["trials"].get, reverse=True)[:3]
        for label, exprs in (("chosen", chosen), ("argmax", argmax3)):
            c_is = composite(exprs, px, r1, end=split - 1, scale_window=split - 1)
            c_oos = composite(exprs, px, r1, start=split, scale_window=split - 1)
            rows.append({"run": r["run"], "arm": arm_of(r["run"]), "rule": label,
                         "sr_is": sharpe(c_is), "sr_oos": sharpe(c_oos),
                         "decay": sharpe(c_is) - sharpe(c_oos)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "real_decay.csv"), index=False)
    print(df[df.rule == "argmax"].groupby("arm")[["sr_is", "sr_oos", "decay"]]
          .agg(["mean", "count"]).round(3).to_string())
    return df


def step_canon():
    from arms import canon_placebo
    rows = canon_placebo("REAL")
    df = pd.DataFrame(rows)
    print(df[["name", "sr_train", "sr_oos"]].to_string(index=False))
    print("mean train %.3f  mean oos %.3f  mean decay %.3f" %
          (df.sr_train.mean(), df.sr_oos.mean(), (df.sr_train - df.sr_oos).mean()))
    return df


if __name__ == "__main__":
    os.makedirs(RESULTS, exist_ok=True)
    step = sys.argv[1] if len(sys.argv) > 1 else "logs"
    {"logs": step_logs, "syn": step_syn_decay, "real": step_real,
     "canon": step_canon}[step]()
