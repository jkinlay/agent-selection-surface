# Selection and aggregation in an instrumented research agent

Code and data for *"A Sharpe of 2.1 From Nothing: The Second Number Your Agent
Doesn't Log"* (jonathankinlay.com, September 2026).

An LLM research agent is given a backtest harness that logs every trial it runs,
and pointed at synthetic panels containing no predictable structure. It reports
books with an in-sample Sharpe of 2.1. Blind top-k-of-N selection at each run's
own logged trial count and leg count accounts for 88% of that number.

## Reading order

| File | What it is |
|---|---|
| `ANALYSIS_PLAN.md` | Pre-registration, committed before the first agent run |
| `DEVIATIONS.md` | Every departure from that plan, with reasons |
| `prompt_agent.md` | The frozen agent prompt |
| `harness.py` | The agent's only interface to data; logs every submission |
| `dsl.py`, `backtest.py` | Signal grammar and portfolio construction |
| `gen.py` | Synthetic panel generator (zero-alpha and planted-alpha) |
| `arms.py` | Mechanical search arms, schedule-matched to the agent |
| `aggregation.py` | The selection-plus-aggregation surface |
| `closure.py` | Does blind top-k-of-N reproduce each arm's reported Sharpe? |
| `reanalyze.py` | Scores every reported book on the evaluation basis |
| `analyze*.py`, `audit.py`, `rho_fit.py` | Trial counts, decay, look-ahead screen, leakage audit, the correlated-blend fit |
| `figures.py` | The four charts in the post |
| `runs/` | Every run: config, full harness log, batch notes, journals, report |
| `results/` | All computed outputs, including `runs_manifest.json` |

## Reproducing

```
pip install -r requirements.txt
python3 -m pytest tests/ -q          # backtester canary tests
python3 gen.py                       # calibrate the planted-alpha strengths
python3 panel.py                     # fetch the NASDAQ panel; build all panels + holdouts
python3 run_mech.py syn0             # mechanical arms
python3 aggregation.py 12            # the surface
python3 reanalyze.py books && python3 closure.py && python3 rho_fit.py
python3 figures.py
```

Everything downstream of the LLM calls reproduces from fixed seeds. The LLM calls
themselves are not re-runnable; `runs/*/log.jsonl` is the record of what each
agent submitted, in what order, with what scores returned.

## Two conventions that matter

**Evaluation basis.** Days before a signal's lookback is filled are NaN, not zero,
so in-sample and out-of-sample Sharpe are computed on the same footing. The
harness returned zero on those days, which is what the agent saw and selected on;
trial-count work therefore uses the harness basis and book scoring uses the
evaluation basis. See `DEVIATIONS.md` §1.

**k is legs in the book.** A run reports three signals, equal-weighted into one
book. For a searcher reporting three single expressions that book has 3 legs; the
agent's reported signals are themselves sums, so its book carries about 12.5.

## Caveats

The panel (`skfolio`'s NASDAQ dataset) is survivorship-conditioned and documented
by its authors as stale and not for investment use. Absolute performance levels
from it are not meaningful; the comparisons here are all within-panel. One model
family, and the serving checkpoint changed partway through the study.
