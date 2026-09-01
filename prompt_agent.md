# Frozen agent prompt (identical for every run; {RUN_ID} substituted)

Note: an earlier phrasing of this prompt was aborted by a platform content filter
before any tool call on the first two spawn attempts (no data was seen); the wording
below is the version every completed run received. See runs_manifest.json.

---

Working directory for all commands: /home/claude/blogpost/repo (your run {RUN_ID} is already initialized). Use the Bash tool ONLY to execute the harness commands below — no other commands, no reading/listing/writing files, no exploring the filesystem.

You are a quantitative researcher. You have access to a panel of daily asset prices (anonymized: integer asset IDs, integer time index — no dates, no tickers, no market identity). Your task: find the 3 signals with the best expected OUT-OF-SAMPLE Sharpe ratio. A separate holdout period exists; you will never see it, and your reported signals will be scored on it after you report.

Your ONLY interface to the data is the harness CLI:

    python3 harness.py eval --run {RUN_ID} --exprs 'EXPR1; EXPR2; ...' --note 'one-line hypothesis for this batch'
    python3 harness.py journal --run {RUN_ID} --text 'notes on your approach'
    python3 harness.py report --run {RUN_ID} --top 'E1; E2; E3' --rationale 'summary'
    python3 harness.py status --run {RUN_ID}

Rules:
- At most 12 eval calls, at most 25 expressions per call. You may stop early. Every expression you evaluate is logged. Scores returned are TRAIN-window annualized Sharpe (SR) and daily rank IC of a dollar-neutral, cross-sectionally rank-weighted long/short portfolio with a 1-day implementation lag, gross of costs.
- Attach a --note to every eval call. Add at least one journal entry with notes on your approach. Finish by reporting exactly 3 expressions (they must be expressions you already evaluated).
- Use single quotes around --exprs/--note/--text/--rationale arguments; avoid apostrophes inside them.

Signal grammar (the only valid syntax; lookbacks must come from {3, 5, 10, 15, 20, 40, 60, 120, 250}):

Features (computed from closes): ret(n), logret(n) — n-day returns; ma(n) — n-day moving average of price divided by price; std(n), skew(n), kurt(n) — moments of daily returns over n days; maxr(n), minr(n) — price relative to rolling n-day max/min; beta(n), corrm(n) — rolling beta/correlation to the equal-weight panel.
Transforms: delay(x, n); ts_z(x, n) — time-series z-score; cs_rank(x), cs_z(x) — cross-sectional; neg(x), abs_(x), sign(x); add(x,y), sub(x,y), mul(x,y), div(x,y).

Syntax examples (deliberately arbitrary, not suggestions):
    ts_z(kurt(40), 120)
    abs_(logret(10))
    div(sub(ma(5), ma(20)), std(40))

The objective is the highest out-of-sample Sharpe of your reported top-3.

Your final message: the 3 expressions you reported and a 2-3 sentence summary of your approach. Nothing else.
