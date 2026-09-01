"""Canary tests: the backtester must reward look-ahead and must not leak it.

Run: python3 -m pytest tests/ -q
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from dsl import Evaluator, parse, random_expr, unparse
from backtest import portfolio_returns, sharpe, score, weights_from_signal, LAG


def _null_panel(seed=0, T=900, N=150):
    rng = np.random.default_rng(seed)
    r = rng.standard_normal((T, N)) * 0.02
    px = pd.DataFrame(np.cumprod(1 + r, axis=0) * 50.0)
    return px, px.pct_change()


def test_leaky_signal_scores_huge():
    """A signal equal to TOMORROW's return, evaluated with no lag, must be enormous.
    This proves the scorer can see leakage when leakage exists."""
    px, r1 = _null_panel(1)
    fwd = r1.shift(-(LAG))          # exactly undoes the implementation lag
    sr = sharpe(portfolio_returns(fwd, r1))
    assert sr > 10, f"leaky canary failed: SR={sr}"


def test_lagged_copy_is_null():
    """The same information, properly lagged one extra day, must NOT be huge."""
    px, r1 = _null_panel(1)
    fwd_lagged = r1.shift(-(LAG - 1))   # one day short of leaking
    sr = sharpe(portfolio_returns(fwd_lagged, r1))
    assert abs(sr) < 2.0, f"lag canary failed: SR={sr}"


def test_random_signals_are_null_on_null_panel():
    """On an iid panel, random DSL expressions must average SR ~ 0."""
    px, r1 = _null_panel(2)
    ev = Evaluator(px)
    rng = np.random.default_rng(3)
    srs = []
    for _ in range(40):
        e = random_expr(rng)
        try:
            srs.append(score(ev.eval(e), r1)["sharpe"])
        except Exception:
            pass
    m = np.mean(srs)
    assert abs(m) < 0.25, f"mean SR of random signals on null panel = {m}"


def test_weights_dollar_neutral_unit_gross():
    px, r1 = _null_panel(4)
    ev = Evaluator(px)
    sig = ev.eval(parse("cs_rank(ret(20))"))
    w = weights_from_signal(sig).iloc[300]
    assert abs(w.sum()) < 1e-9
    assert abs(w.abs().sum() - 1.0) < 1e-9


def test_parser_roundtrip_and_rejects():
    e = random_expr(np.random.default_rng(5))
    assert parse(unparse(e)) == e
    for bad in ["__import__('os')", "ret(7)", "ma(5", "close", "add(ret(5))", "ts_z(ret(5), 7)"]:
        try:
            parse(bad)
            raised = False
        except Exception:
            raised = True
        assert raised, f"parser accepted {bad!r}"


def test_no_future_info_in_features():
    """Perturbing prices after day t must not change any feature at day t."""
    px, _ = _null_panel(6, T=400, N=60)
    px2 = px.copy()
    px2.iloc[300:] *= 1.5
    for expr in ["ret(20)", "ts_z(std(20), 60)", "cs_rank(sub(ma(5), ma(20)))",
                 "beta(60)", "maxr(120)", "kurt(15)"]:
        a, b = Evaluator(px).eval(parse(expr)), Evaluator(px2).eval(parse(expr))
        pd.testing.assert_frame_equal(a.iloc[:299], b.iloc[:299])
