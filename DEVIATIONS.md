# Deviations from the pre-registered analysis plan

Every departure from `ANALYSIS_PLAN.md`, with the reason. Committed alongside the
results so the plan and the outcome can be diffed.

## 1. Evaluation basis changed to warm-start (affects every reported Sharpe)

The plan did not specify how to treat days before a signal's lookback is filled.
The harness returned zero on those days, so a 250-day-lookback signal was credited
with ~250 exact zeros inside the 1,008-day train window while its out-of-sample
window was fully warm — in-sample and out-of-sample were not on the same footing,
and the distortion depended on lookback length.

All *reported book* scoring (`results/books.csv`, via `reanalyze.composite_scores`)
now marks pre-warm-up days NaN rather than zero. Selection is unaffected: which
signals a searcher chose is a historical fact and still uses the harness-basis
scores in the run logs.

Two bases therefore coexist deliberately, and are not interchangeable:

| Basis | Used for | Where |
|---|---|---|
| Harness (zero-padded) | trial counts, null pools, effective-N — what the searcher saw | `runs/*/log.jsonl`, `results/null_pool_*.json`, `results/run_logs.csv` |
| Evaluation (warm-start) | every reported book Sharpe, the aggregation surface | `results/books.csv`, `results/aggregation_surface.csv` |

Effect: raises measured in-sample Sharpe for every arm (agent SYN-0 1.87 → 2.12).

## 2. The primary effective-N estimator is censored and was demoted

The plan made simulation-implied N* the primary matched quantity. It is censored
in 93% of agent runs — the run's best in-sample Sharpe exceeds the maximum of the
1,500–2,500-draw null pool, so no trial count reproduces it. Censoring rates are
reported by arm instead, and the comparison is made at realised per-run trial and
leg counts. The censoring is partly a property of a finite pool and no argument
rests on it.

## 3. The random-search (RS) leg of the real-panel decay decomposition was not run

The plan specified RS-top-3 at matched N* as the selection component of the
real-panel decomposition. It was not executed. Consequently the decomposition
rests on the canon placebo alone, which licenses only the weaker claim stated in
the post: the unselected canon did not decay across this boundary, so the regime
does not explain the agent's gap. It does not establish that selection explains
all of it.

## 4. Aggregation surface: first version used the wrong window

The first surface intersected the live windows of the entire 500-expression
library, scoring cells on ~440 train days against the books' ~940 and inflating
every cell by roughly 0.3–1.0 Sharpe. It is rebuilt to score exactly as
`composite_scores` does. The superseded runs are not in the results.

## 5. Look-ahead screen: 60 continuations, not 200

Compute budget. The screen is in any case structurally incapable of a positive
result — resampling training returns reproduces the structure the books were
selected on — which is reported in the post.

## 6. Synthetic null pools are 1,500 draws, not 2,000

Compute budget. The real-panel pool is 2,500 as planned.

## 7. The frozen prompt was revised after the pre-registration commit

Two initial agent spawns were aborted by a platform content filter before any tool
call (no data was seen). The prompt was reworded and every completed run received
the revised text in `prompt_agent.md`. One substantive change: the pre-registered
version ended "Higher out-of-sample Sharpe is the objective. In-sample Sharpe is
what you can measure. How you manage that gap is up to you."; the revised version
replaces this with "The objective is the highest out-of-sample Sharpe of your
reported top-3." A reader may reasonably argue this changed the agent's framing of
the in-sample/out-of-sample trade-off. Both texts are in the git history.

## 8. Model checkpoint changed mid-study, and is not recorded per run

A rate limit forced a switch of serving model checkpoint partway through. The plan
promised the serving model would be recorded per run; it was not, and checkpoint
assignment is inferred from run timestamps. The twelve primary zero-alpha runs
precede the switch; the three `AGB-` bridge runs and the planted-alpha runs follow
it. This is an instrumentation defect.

## 9. Dollar table reports bps of notional, not bps of book volatility

The plan said book volatility. Notional is the more familiar denominator for the
comparison being made, and the volatility assumption is stated in the table.

## 10. Runs abandoned (all in `runs_manifest.json`)

Five runs are excluded under the pre-registered inclusion rule (a run counts only
if it completes the protocol). Four `AG-SYNA05-*` runs were interrupted mid-flight
by the rate limit and re-run from scratch on the same four panels; one
`AG-REAL-00` was contaminated by an operator timing probe before the agent
started, and was replaced by `AG-REAL-06`. No exclusion conditions on results.

## 11. `step_bridge` did not run to completion

The planned trial-process bridge statistics (correlation structure and score
dispersion, synthetic versus real) fail with a linear-algebra error on some runs
and are not reported. The post does not claim them.
