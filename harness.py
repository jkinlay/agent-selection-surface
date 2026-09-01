"""Instrumented backtest harness — the agent's ONLY interface to the data.

The agent never sees prices, dates, tickers, or the holdout. It submits DSL
expressions and receives train-window scores. Every submission is logged.

CLI:
  python3 harness.py init    --run R --panel P
  python3 harness.py eval    --run R --exprs 'e1; e2; ...' [--note '...']
  python3 harness.py journal --run R --text '...'
  python3 harness.py report  --run R --top 'e1; e2; e3' [--rationale '...']
  python3 harness.py status  --run R

Limits: <= 25 expressions per eval call, <= 12 eval calls per run.
Invalid expressions are logged but return errors and do not count as trials.
Reported signals must already have been evaluated in this run.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
MAX_BATCHES = 12
MAX_PER_BATCH = 25


def _rd(run):
    d = os.path.join(RUNS, run)
    if not os.path.isdir(d):
        sys.exit(f"unknown run {run!r}; init first")
    return d


def _cfg(run):
    with open(os.path.join(_rd(run), "config.json")) as f:
        return json.load(f)


def _log_path(run):
    return os.path.join(_rd(run), "log.jsonl")


def _append(run, rec):
    rec["ts"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with open(_log_path(run), "a") as f:
        f.write(json.dumps(rec) + "\n")


def _read_log(run):
    p = _log_path(run)
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def _n_batches(run):
    return sum(1 for r in _read_log(run) if r.get("kind") == "eval_batch")


def cmd_init(a):
    d = os.path.join(RUNS, a.run)
    os.makedirs(d, exist_ok=False)
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"run": a.run, "panel": a.panel, "arm": a.arm}, f, indent=2)
    _append(a.run, {"kind": "init", "panel": a.panel, "arm": a.arm})
    print(f"run {a.run} ready (panel {a.panel}). {MAX_BATCHES} eval calls of up to "
          f"{MAX_PER_BATCH} expressions each. Scores are TRAIN-window only.")


def _evaluate(run, exprs):
    from dsl import Evaluator, parse
    from backtest import score
    from panel import load_train

    cfg = _cfg(run)
    px = load_train(cfg["panel"])
    r1 = px.pct_change()
    ev = Evaluator(px)
    out = []
    for e in exprs:
        e = e.strip()
        if not e:
            continue
        try:
            ast = parse(e)
            sig = ev.eval(ast)
            if sig.iloc[300:].notna().sum(axis=1).median() < 20:
                out.append({"expr": e, "error": "degenerate: <20 valid names on a typical day"})
                continue
            sc = score(sig, r1)
            out.append({"expr": e, "sharpe": sc["sharpe"], "ic": sc["ic"]})
        except Exception as ex:
            out.append({"expr": e, "error": str(ex)[:120]})
    return out


def cmd_eval(a):
    if os.path.exists(os.path.join(_rd(a.run), "report.json")):
        sys.exit("run already reported; no further evals")
    nb = _n_batches(a.run)
    if nb >= MAX_BATCHES:
        sys.exit(f"eval budget exhausted ({MAX_BATCHES} calls)")
    exprs = [e for e in a.exprs.split(";") if e.strip()]
    if len(exprs) > MAX_PER_BATCH:
        sys.exit(f"max {MAX_PER_BATCH} expressions per call; got {len(exprs)}")
    res = _evaluate(a.run, exprs)
    _append(a.run, {"kind": "eval_batch", "batch": nb + 1, "note": a.note or "",
                    "results": res})
    for r in res:
        if "error" in r:
            print(f"ERROR   {r['expr']}  [{r['error']}]")
        else:
            print(f"SR={r['sharpe']:+.3f}  IC={r['ic']:+.4f}  {r['expr']}")
    print(f"[batch {nb + 1}/{MAX_BATCHES}]")


def cmd_journal(a):
    _append(a.run, {"kind": "journal", "text": a.text})
    print("noted")


def cmd_report(a):
    d = _rd(a.run)
    if os.path.exists(os.path.join(d, "report.json")):
        sys.exit("already reported")
    top = [e.strip() for e in a.top.split(";") if e.strip()]
    if len(top) != 3:
        sys.exit("report exactly 3 expressions, separated by ';'")
    seen = {}
    for rec in _read_log(a.run):
        if rec.get("kind") == "eval_batch":
            for r in rec["results"]:
                if "sharpe" in r:
                    seen[r["expr"]] = r["sharpe"]
    for e in top:
        if e not in seen:
            sys.exit(f"{e!r} was never successfully evaluated in this run")
    rep = {"top3_chosen": top, "chosen_sr_train": [seen[e] for e in top],
           "rationale": a.rationale or "", "n_batches_used": _n_batches(a.run)}
    with open(os.path.join(d, "report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    _append(a.run, {"kind": "report", **rep})
    print("report accepted; run closed. Thank you.")


def cmd_status(a):
    nb = _n_batches(a.run)
    done = os.path.exists(os.path.join(_rd(a.run), "report.json"))
    n_valid = len({r["expr"] for rec in _read_log(a.run) if rec.get("kind") == "eval_batch"
                   for r in rec["results"] if "sharpe" in r})
    print(f"batches used: {nb}/{MAX_BATCHES}; unique valid expressions: {n_valid}; "
          f"reported: {done}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init"); p.add_argument("--run", required=True)
    p.add_argument("--panel", required=True); p.add_argument("--arm", default="AGENT")
    p = sub.add_parser("eval"); p.add_argument("--run", required=True)
    p.add_argument("--exprs", required=True); p.add_argument("--note", default="")
    p = sub.add_parser("journal"); p.add_argument("--run", required=True)
    p.add_argument("--text", required=True)
    p = sub.add_parser("report"); p.add_argument("--run", required=True)
    p.add_argument("--top", required=True); p.add_argument("--rationale", default="")
    p = sub.add_parser("status"); p.add_argument("--run", required=True)
    a = ap.parse_args()
    {"init": cmd_init, "eval": cmd_eval, "journal": cmd_journal,
     "report": cmd_report, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    main()
