"""Analysis part 3: planted-alpha capture, minimum detectable effects,
dollar haircut arithmetic, and the summary tables used in the post.

Usage: python3 analyze3.py <step>
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

from analyze import all_runs, arm_of, composite, RESULTS
from dsl import Evaluator, parse
from backtest import portfolio_series, sharpe
from panel import load_full


def step_capture():
    """Did the searcher find the planted mechanisms?

    Each SYN-A panel carries two planted effects of equal calibrated strength:
      canon-shaped  : 5-day cross-sectional reversal   (oracle neg(ret(5)))
      anti-canon    : high 15-day return kurtosis      (oracle cs_rank(kurt(15)))
    Capture = correlation of the run's reported composite with each oracle's
    portfolio return series, measured on the OOS window (out of sample for the
    searcher, and where only true structure survives).
    """
    oracles = {"canon_plant": "neg(ret(5))", "anti_plant": "cs_rank(kurt(15))"}
    rows = []
    for r in all_runs():
        if not r["panel"].startswith("SYNA"):
            continue
        px, split = load_full(r["panel"])
        r1 = px.pct_change()
        ev = Evaluator(px)
        a3 = sorted(r["trials"], key=r["trials"].get, reverse=True)[:3]
        comp = composite(a3, px, r1, start=split, scale_window=split - 1)
        row = {"run": r["run"], "arm": arm_of(r["run"]), "panel": r["panel"],
               "strength": "0.5" if "SYNA05" in r["panel"] else "1.0",
               "sr_oos": sharpe(comp)}
        for k, e in oracles.items():
            o = portfolio_series(ev.eval(parse(e)), r1).loc[comp.index]
            row[k] = float(np.corrcoef(comp.values, o.values)[0, 1])
            row[k + "_sr"] = sharpe(o)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "capture.csv"), index=False)
    print(df.groupby(["strength", "arm"])[
        ["canon_plant", "anti_plant", "sr_oos"]].agg(["mean", "count"]).round(3).to_string())
    print("\noracle OOS Sharpes (truth ceiling):")
    print(df.groupby("strength")[["canon_plant_sr", "anti_plant_sr"]].mean().round(3).to_string())
    return df


def step_mde():
    """Minimum detectable effects for the headline contrasts, from the realized
    across-panel dispersion. Reported alongside every comparison in the post."""
    d = pd.read_csv(os.path.join(RESULTS, "syn_decay.csv"))
    d = d[d.rule == "argmax"]
    out = {}
    for setting in ["SYN0", "SYNA05", "SYNA10"]:
        s = d[d.setting == setting]
        for arm in sorted(s.arm.unique()):
            pp = s[s.arm == arm].groupby("panel")["decay"].mean()
            out[f"{setting}:{arm}"] = {"n_panels": int(len(pp)),
                                       "mean_decay": round(float(pp.mean()), 3),
                                       "sd_across_panels": round(float(pp.std()), 3),
                                       "se": round(float(pp.std() / np.sqrt(len(pp))), 3)}
    # paired AGENT vs OPT-medium on the same SYN-0 panels
    s = d[d.setting == "SYN0"]
    a = s[s.arm == "AGENT"].groupby("panel")["decay"].mean()
    o = s[s.arm == "OPT-medium"].groupby("panel")["decay"].mean()
    common = a.index.intersection(o.index)
    diff = (a[common] - o[common])
    n = len(diff)
    se = float(diff.std() / np.sqrt(n))
    out["paired_AGENT_minus_OPTmedium_SYN0"] = {
        "n_panels": int(n), "mean_diff": round(float(diff.mean()), 3),
        "se": round(se, 3), "t": round(float(diff.mean() / se), 2),
        "mde_80pct_power": round(2.8 * se, 3)}
    json.dump(out, open(os.path.join(RESULTS, "mde.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))
    return out


BOOK_NOTIONAL = 100e6
TARGET_VOL = 0.10


def step_dollars():
    """Selection-haircut arithmetic. NOT a forecast, NOT strategy P&L.

    Given a book run at TARGET_VOL on $BOOK_NOTIONAL, one unit of annualized
    Sharpe is worth TARGET_VOL * NOTIONAL of expected annual return. The
    overstatement is (face IS Sharpe - calibrated expectation) * that unit.
    The calibrated expectation comes from the SYN-0 arm: on data with zero true
    predictability, this pipeline's reported book realizes OOS Sharpe ~0, so the
    entire face IS Sharpe of a comparable book is selection.
    """
    B = pd.read_csv(os.path.join(RESULTS, "books.csv"))
    B = B[B.rule == "argmax"]
    syn = B[(B.setting == "SYN0") & (B.arm == "AGENT") & (B.ckpt == "A")]
    real = B[(B.setting == "REAL") & (B.arm == "AGENT")]
    unit = TARGET_VOL * BOOK_NOTIONAL          # $ per unit of annualized Sharpe
    face = float(real.sr_is.mean())
    # True Sharpe is zero on SYN-0 by construction, so the manufactured component
    # is the reported in-sample Sharpe itself. (Subtracting the realised OOS mean
    # would subtract sampling noise from a quantity whose expectation is zero.)
    syn_haircut = float(syn.sr_is.mean())
    priced = face - syn_haircut
    realized = float(real.sr_oos.mean())
    boot = json.load(open(os.path.join(RESULTS, "bootstrap_b21.json")))["arms"]["AGENT"]
    out = {
        "book_notional_usd": BOOK_NOTIONAL, "target_vol": TARGET_VOL,
        "usd_per_sharpe_unit": unit,
        "face_is_sharpe": round(face, 2),
        "syn0_measured_haircut_sharpe": round(syn_haircut, 2),
        "priced_sharpe_after_haircut": round(priced, 2),
        "overstatement_sharpe": round(syn_haircut, 2),
        "overstatement_usd_per_year": round(syn_haircut * unit, -5),
        "overstatement_bps_of_notional": round(syn_haircut * TARGET_VOL * 1e4, -1),
        "realized_oos_sharpe_one_draw": round(realized, 2),
        "realized_oos_block_bootstrap_95ci": [round(boot["lo"], 2), round(boot["hi"], 2)],
        "label": ("Selection-haircut arithmetic given the SYN-0 calibration. "
                  "Gross of costs. Relative overstatement only — not a forecast, "
                  "not strategy P&L."),
    }
    json.dump(out, open(os.path.join(RESULTS, "dollars.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))
    return out


def step_cio_table():
    """The forwardable table: per REAL run, what was reported vs what was logged."""
    logs = pd.read_csv(os.path.join(RESULTS, "run_logs.csv"))
    real = pd.read_csv(os.path.join(RESULTS, "real_decay.csv"))
    real = real[real.rule == "argmax"]
    m = logs.merge(real[["run", "sr_is", "sr_oos", "decay"]], on="run")
    m = m[m.panel == "REAL"].sort_values(["arm", "run"])
    cols = ["run", "arm", "n_logged", "n_star", "sr_is", "sr_oos", "decay"]
    m[cols].to_csv(os.path.join(RESULTS, "cio_table.csv"), index=False)
    print(m[cols].to_string(index=False))
    return m


if __name__ == "__main__":
    step = sys.argv[1]
    {"capture": step_capture, "mde": step_mde, "dollars": step_dollars,
     "cio": step_cio_table}[step]()
