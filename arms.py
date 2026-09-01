"""Mechanical search arms, schedule-matched to the agent (12 rounds x <= 25 candidates,
feedback once per round), plus the no-feedback arms and null pools.

All arms write the same runs/<id>/log.jsonl + report.json format as the harness,
with report = argmax-3 by train Sharpe (mechanical arms have no judgment channel).
"""
from __future__ import annotations

import datetime
import json
import os

import numpy as np

from dsl import Evaluator, parse, unparse, random_expr, mutate, crossover
from backtest import score
from canon import CANON, jitter_template
from panel import load_train

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

N_ROUNDS = 12
BATCH = 25

# OPT selection-pressure sweep: (tournament size, elite count, p_crossover)
OPT_SETTINGS = {
    "soft":   {"k": 2, "elite": 1, "p_x": 0.3},
    "medium": {"k": 4, "elite": 2, "p_x": 0.5},
    "hard":   {"k": 8, "elite": 5, "p_x": 0.6},
}


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class _Scorer:
    def __init__(self, panel_id):
        self.px = load_train(panel_id)
        self.r1 = self.px.pct_change()
        self.ev = Evaluator(self.px)

    def score_exprs(self, exprs):
        out = []
        for e in exprs:
            try:
                ast = parse(e) if isinstance(e, str) else e
                s = self.ev.eval(ast)
                if s.iloc[300:].notna().sum(axis=1).median() < 20:
                    out.append({"expr": unparse(ast) if not isinstance(e, str) else e,
                                "error": "degenerate"})
                    continue
                sc = score(s, self.r1)
                out.append({"expr": unparse(ast) if not isinstance(e, str) else e,
                            "sharpe": sc["sharpe"], "ic": sc["ic"]})
            except Exception as ex:
                out.append({"expr": str(e)[:80], "error": str(ex)[:120]})
        return out


def _write_run(run_id, panel_id, arm, batches, note=""):
    d = os.path.join(RUNS, run_id)
    os.makedirs(d, exist_ok=False)
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"run": run_id, "panel": panel_id, "arm": arm}, f, indent=2)
    seen = {}
    with open(os.path.join(d, "log.jsonl"), "w") as f:
        f.write(json.dumps({"kind": "init", "panel": panel_id, "arm": arm,
                            "ts": _now(), "note": note}) + "\n")
        for i, res in enumerate(batches, start=1):
            f.write(json.dumps({"kind": "eval_batch", "batch": i, "note": "",
                                "results": res, "ts": _now()}) + "\n")
            for r in res:
                if "sharpe" in r:
                    seen[r["expr"]] = r["sharpe"]
    top = sorted(seen, key=seen.get, reverse=True)[:3]
    rep = {"top3_chosen": top, "chosen_sr_train": [seen[e] for e in top],
           "rationale": f"argmax-3 by train Sharpe ({arm})",
           "n_batches_used": len(batches)}
    with open(os.path.join(d, "report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    return rep


def run_rs(panel_id, seed, run_id=None):
    """Random search: 12 x 25 draws from the grammar prior."""
    rng = np.random.default_rng(seed)
    sc = _Scorer(panel_id)
    batches = [sc.score_exprs([random_expr(rng) for _ in range(BATCH)])
               for _ in range(N_ROUNDS)]
    run_id = run_id or f"RS-{panel_id}-s{seed}"
    return _write_run(run_id, panel_id, "RS", batches)


def run_opt(panel_id, seed, setting="medium", run_id=None):
    """Evolutionary search at the agent's interaction schedule:
    population 25, 12 generations, feedback once per generation."""
    cfgs = OPT_SETTINGS[setting]
    rng = np.random.default_rng(seed)
    sc = _Scorer(panel_id)
    pop = [random_expr(rng) for _ in range(BATCH)]
    batches, scored = [], []
    for g in range(N_ROUNDS):
        res = sc.score_exprs(pop)
        batches.append(res)
        pairs = [(parse(r["expr"]), r["sharpe"]) for r in res if "sharpe" in r]
        scored.extend(pairs)
        if g == N_ROUNDS - 1:
            break
        pool = sorted(scored, key=lambda t: t[1], reverse=True)
        elite = [a for a, _ in pool[:cfgs["elite"]]]
        nxt = list(elite)
        while len(nxt) < BATCH:
            def tourney():
                idx = rng.integers(0, len(pool), size=min(cfgs["k"], len(pool)))
                best = max(idx, key=lambda i: pool[i][1])
                return pool[best][0]
            a = tourney()
            if rng.random() < cfgs["p_x"]:
                child = crossover(a, tourney(), rng)
            else:
                child = mutate(a, rng)
            nxt.append(child)
        pop = nxt
    run_id = run_id or f"OPT{setting[0].upper()}-{panel_id}-s{seed}"
    return _write_run(run_id, panel_id, f"OPT-{setting}", batches)


def run_canon_sampler(panel_id, seed, run_id=None):
    """Canon-shaped draws, no feedback: prior direction without optimization."""
    rng = np.random.default_rng(seed)
    sc = _Scorer(panel_id)
    batches = [sc.score_exprs([jitter_template(rng) for _ in range(BATCH)])
               for _ in range(N_ROUNDS)]
    run_id = run_id or f"CS-{panel_id}-s{seed}"
    return _write_run(run_id, panel_id, "CANON-SAMPLER", batches)


def build_null_pool(panel_id, n=2500, seed=123):
    """One-off pool of random-expression train scores: the null per-trial
    IS-Sharpe distribution for this panel (feeds the implied-N* estimator)."""
    rng = np.random.default_rng(seed)
    sc = _Scorer(panel_id)
    out = []
    while len(out) < n:
        res = sc.score_exprs([random_expr(rng) for _ in range(50)])
        out.extend([r for r in res if "sharpe" in r])
    out = out[:n]
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"null_pool_{panel_id}.json"), "w") as f:
        json.dump(out, f)
    return len(out)


def canon_placebo(panel_id_hold="REAL"):
    """The 12 canonical anomalies, train vs OOS, no selection (REAL only)."""
    from panel import load_full
    from backtest import score as bscore

    px, split = load_full(panel_id_hold)
    r1 = px.pct_change()
    ev = Evaluator(px)
    rows = []
    for name, e in CANON.items():
        sig = ev.eval(parse(e))
        tr = bscore(sig, r1, end=split - 1)
        oo = bscore(sig, r1, start=split)
        rows.append({"name": name, "expr": e, "sr_train": tr["sharpe"],
                     "sr_oos": oo["sharpe"], "ic_train": tr["ic"], "ic_oos": oo["ic"]})
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "canon_placebo_REAL.json"), "w") as f:
        json.dump(rows, f, indent=2)
    return rows
