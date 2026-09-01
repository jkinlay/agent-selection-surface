"""Leakage audit of agent free text, and the run manifest. Writes
results/leakage_audit.json and results/runs_manifest.json."""
import json, os, re
from analyze import all_runs

YEAR = re.compile(r"\b(19|20)\d{2}\b")
MONTH = re.compile(r"\b(january|february|march|april|june|july|august|september|"
                   r"october|november|december)\b", re.I)
EVENT = re.compile(r"\b(covid|pandemic|nasdaq|s&p\s?500|sp500|dot-?com|gfc|"
                   r"financial crisis|rate hike[s]?|bear market|bull market|"
                   r"tech bubble|taper|quantitative easing|fomc)\b", re.I)
TICK = re.compile(r"\b(AAPL|MSFT|NVDA|TSLA|AMZN|GOOGL?|META|SPY|QQQ|IWM)\b")


def leakage():
    hits, n = [], 0
    for r in all_runs():
        if not r["run"].startswith(("AG-", "AGB-")):
            continue
        rep = r["report"] or {}
        texts = list(r["journals"]) + [b.get("note", "") for b in r["batches"]] \
            + [rep.get("rationale", "")]
        for t in texts:
            if not t:
                continue
            n += 1
            for rx, kind in ((YEAR, "year"), (MONTH, "month"),
                             (EVENT, "event"), (TICK, "ticker")):
                for m in rx.finditer(t):
                    hits.append({"run": r["run"], "kind": kind, "match": m.group(0),
                                 "context": t[max(0, m.start() - 60):m.start() + 60]})
    json.dump({"fields_scanned": n, "hits": hits},
              open("results/leakage_audit.json", "w"), indent=2)
    print(f"agent free-text fields scanned: {n}; date/month/event/ticker hits: {len(hits)}")


def manifest():
    rows = []
    for d in sorted(os.listdir("runs")):
        p = os.path.join("runs", d)
        cfg_p = os.path.join(p, "config.json")
        cfg = json.load(open(cfg_p)) if os.path.exists(cfg_p) else {}
        done = os.path.exists(os.path.join(p, "report.json"))
        if d.startswith("ABANDONED"):
            disp, why = "abandoned", (
                "operator timing probe contaminated the run before the agent started; "
                "replaced by AG-REAL-06" if "operator-probe" in d else
                "interrupted mid-run by a provider rate limit; re-run from scratch "
                "on the same panel")
        elif d.startswith("SMOKE"):
            disp, why = "excluded", "harness smoke test, never an experimental run"
        elif done:
            disp, why = "included", ""
        else:
            disp, why = "excluded", "did not complete the protocol"
        rows.append({"run": d, "panel": cfg.get("panel"), "arm": cfg.get("arm"),
                     "disposition": disp, "reason": why})
    json.dump(rows, open("results/runs_manifest.json", "w"), indent=2)
    from collections import Counter
    print("manifest:", dict(Counter(r["disposition"] for r in rows)), f"total {len(rows)}")


if __name__ == "__main__":
    leakage()
    manifest()
