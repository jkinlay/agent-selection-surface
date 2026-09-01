# Pre-registered analysis plan

**Committed before any agent run. The git history is the timestamp.**

Study: instrument a minimal LLM research agent with a backtest harness that logs every
trial; calibrate on synthetic panels (truth known) which trial count — reported, logged,
or effective — prices the selection gain in what the agent reports; illustrate on a real
panel. Companion mechanical arms isolate optimization pressure from LLM priors.

## Fixed inputs

- Real data: skfolio NASDAQ panel, SHA-256
  `92e507b787d77367ef595f55de2920792b088c19dcad16609e85d23c9c5a1fd8`;
  median-price ≥ $5 filter → 1,280 names; train 2018-01-02→2021-12-31 (1,008 d),
  OOS 2022-01-03→2023-05-31 (354 d). Survivorship-conditioned; levels biased;
  within-panel comparisons are the objects of interest.
- DSL grammar: fixed in the August 2026 post, predates this hypothesis. Lookback menu
  {3,5,10,15,20,40,60,120,250}. Backtest: dollar-neutral rank L/S, 1-day implementation
  lag (signal day t, first earning day t+2), gross of costs. Canary tests in `tests/`.
- Synthetic generator: `gen.py`; calibration constants in `calibration_targets.json`
  (verified oracle train-window Sharpe, mean ± SE over 12 panels: rev 0.553±0.180 /
  0.923±0.181; anti 0.651±0.097 / 1.122±0.100 at nominal strengths 0.5 / 1.0).
  Per-panel realized oracle Sharpe is additionally recorded and used as local truth.
- Model arm: Claude (session-configured `claude-fable-5`; serving model may differ —
  recorded as such), one model family; per-run timestamps in logs. LLM calls are not
  re-runnable; the harness log and journals are the reproducibility artifact.

## Arms and run counts

12 rounds × ≤25 candidates, feedback once per round, for every adaptive arm
(adaptivity is matched in ROUNDS, the Dwork et al. scaling, not evaluations):

- AGENT: 12 runs on SYN-0 (independent zero-alpha panels), 4+4 on SYN-A
  (strengths ~0.5/~1.0, both mechanisms planted), 6 on REAL. Frozen prompt in
  `prompt_agent.md`.
- OPT (evolutionary, schedule-matched): settings soft/medium/hard
  (`arms.py:OPT_SETTINGS`); SYN-0: 3 settings × 3 reps/panel; SYN-A: medium × 3/panel;
  REAL: 3 settings × 5 reps.
- RS: replicate runs constructed by drawing 12×25 from the per-panel null pools
  (2,000 syn / 2,500 REAL random expressions).
- CANON-SAMPLER: canon-shaped, jittered, no feedback; 2/panel syn, 5 REAL.
- CANON placebo (REAL): the 12 fixed canonical anomalies, no selection.

Inclusion rule: a run is included iff it completes the protocol (exactly 3 reported
signals, previously evaluated, within budget). Every attempted run — including crashed,
degenerate, and smoke runs — appears in `runs_manifest.json` with its disposition.
Exclusion never conditions on results. Smoke runs are labelled `SMOKE-` and excluded by
rule.

## Definitions (fixed)

- **N_reported** = 3.
- **N_logged** = count of unique, successfully evaluated expressions in the run's log
  (syntax errors and degenerate signals do not count; duplicates count once).
- **N\* (implied effective N, primary)** = the N at which the expected maximum of N iid
  draws from the panel's null per-trial IS-Sharpe distribution (empirical, from that
  panel's RS pool) equals the run's expected best IS Sharpe, estimated by resampling;
  reported per run (from its realized best) and per arm (from pooled runs).
- **M_eff (secondary, closed form)** = Li–Ji eigenvalue effective number on the
  correlation matrix of the run's trial signals (daily lagged portfolio returns).
- Display only: average-linkage hierarchical clustering on d = √((1−ρ)/2), cut at
  ρ = 0.5, with a 0.3–0.7 threshold-sensitivity curve (this is threshold clustering,
  not ONC, and is not used for inference).
- **Selection rule**: argmax-3 by train Sharpe is PRIMARY everywhere (mechanical arms
  have no judgment channel); the agent's chosen-3 is reported as a judgment delta.
- Composite book: equal-weight the 3 signals' daily lagged portfolio returns, each
  scaled to equal train vol; run-level composite is the unit of analysis.
- Overfit gain (SYN-0): best/composite IS Sharpe (true SR = 0 ⇒ E[OOS] = 0; calibrating
  the IS max and calibrating decay are the same estimand — counted once).
- Decay (SYN-A, REAL): IS − OOS Sharpe of the composite (secondary: rank IC).

## Pre-registered comparisons and outcome cells

Primary comparison at matched **N\*** (matched N_logged secondary — at equal N_logged
the processes differ in trial dependence, confounding prior direction with search
concentration). All expected-max / expected-top-3 quantities by simulation from
empirical trial processes; never the iid closed form. AGENT yield curve: trials pooled
across runs (caveat: across-run process variance understated at these run counts).
Minimum detectable effects reported alongside every comparison; where MDE exceeds
plausible effects (all REAL horse races), results are presented as measurement, not test.

Outcome cells for AGENT vs OPT overfit gain on SYN-0, all interpretable and stated in
advance: (1) AGENT < OPT → prior-as-regularizer; bill shifts from overfitting to
crowding; may invert at production trial budgets — stated concession. (2) AGENT ≈ OPT →
the loop is an optimizer; telemetry is the only defense. (3) AGENT > OPT → reported
with the decay decomposition before interpretation.

SYN-A: search efficiency = fraction of planted (rev vs anti) alpha captured by reported
composite OOS Sharpe (fresh simulated continuations, truth known); haircut-vs-truth
curve across strengths {0, ~0.5, ~1.0}. Prediction (registered): the agent captures the
canon-shaped plant at a higher rate than the anti-canon plant; the mechanical arms show
no such asymmetry.

REAL decay decomposition (descriptive): AGENT vs RS-top-3 at matched N\* (selection
component) vs CANON placebo (regime + post-publication component). Uncertainty: joint
moving-block bootstrap, 21-day blocks (10/42 sensitivity), resampling the shared OOS
path across all runs simultaneously.

Look-ahead screen (REAL, one-sided): Δ = (R_agent − S_agent) − (R_canon − S_canon),
R = real-OOS composite Sharpe, S = mean over 200 block-bootstrap continuations of the
train panel (train data only). Selection carryover biases Δ downward ⇒ Δ > 0 is
conservative evidence of pretraining look-ahead; Δ ≤ 0 is uninformative. Robustness:
per-stock train-demeaned continuations. Canon-spanned R² of agent picks reported beside Δ.

Dollar table: selection-haircut arithmetic only — face IS expectation vs priced
expectation at N_reported/N_logged/N\* under the SYN-0 calibration, expressed in bps of
book volatility on a $100M book at 10% target vol, one significant figure, gross of
costs, no absolute P&L, no compounding, no realized-OOS column.

## Leakage audit

All agent-visible artifacts are anonymized (integer IDs, no dates/tickers/market names;
synthetic and real runs use the identical prompt). Journals and notes are scanned for
date/ticker/regime references; any hit is reported verbatim in the post. Holdout files
do not exist on disk during agent runs (created afterwards from seeds / source data);
the source CSV is relocated outside the working tree for the duration of the runs.

## What this study cannot show (registered in advance)

Production pipelines run 1–2 orders more trials with holdout gates this design lacks —
everything a real stack adds either raises N_logged or is a gate whose value the same
telemetry would prove. One model family; one minimal single-loop researcher; the floor,
not the ceiling. REAL results are illustration: one OOS path, shared across runs, inside
the model's pretraining corpus. No absolute alpha claims from a survivorship panel.
