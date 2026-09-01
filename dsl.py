"""Price-only signal DSL: grammar, safe parser, cached vectorized evaluator.

Expressions are strings in a small functional grammar, e.g.
    cs_rank(sub(ma(5), ma(20)))
    neg(ret(5))
    ts_z(std(20), 60)

Feature nodes (matrix-valued, causal, computed from adjusted closes only):
    ret(n)      n-day simple return
    logret(n)   n-day log return
    ma(n)       n-day moving average of close, divided by close (scale-free)
    std(n)      n-day std of daily returns
    skew(n)     n-day skewness of daily returns
    kurt(n)     n-day excess kurtosis of daily returns
    maxr(n)     close / rolling n-day max of close
    minr(n)     close / rolling n-day min of close
    beta(n)     n-day rolling beta to the equal-weight panel return
    corrm(n)    n-day rolling correlation with the equal-weight panel return

Transforms:
    delay(x, n)   x lagged n days
    ts_z(x, n)    time-series z-score of x over n days
    cs_rank(x)    cross-sectional rank in [-0.5, 0.5]
    cs_z(x)       cross-sectional z-score
    neg(x), abs_(x), sign(x)
    add(x,y), sub(x,y), mul(x,y), div(x,y)

Lookbacks are restricted to LOOKBACKS. Grammar fixed in the August 2026 post
(predates the hypothesis tested here); see ANALYSIS_PLAN.md.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd

LOOKBACKS = [3, 5, 10, 15, 20, 40, 60, 120, 250]

FEATURES = ["ret", "logret", "ma", "std", "skew", "kurt", "maxr", "minr", "beta", "corrm"]
UNARY = ["neg", "abs_", "sign", "cs_rank", "cs_z"]
BINARY = ["add", "sub", "mul", "div"]
PARAM_TRANSFORMS = ["delay", "ts_z"]

_TOKEN = re.compile(r"\s*([A-Za-z_][A-Za-z_0-9]*|\d+|[(),])")


class DSLError(ValueError):
    pass


# ----------------------------------------------------------------------------- parsing

def tokenize(s: str):
    pos, out = 0, []
    while pos < len(s):
        m = _TOKEN.match(s, pos)
        if not m:
            raise DSLError(f"bad token at position {pos}: {s[pos:pos+10]!r}")
        out.append(m.group(1))
        pos = m.end()
    return out


def parse(s: str):
    """Parse to a nested tuple AST: (op, arg1, arg2, ...). Lookbacks are ints."""
    toks = tokenize(s)
    pos = 0

    def expect(t):
        nonlocal pos
        if pos >= len(toks) or toks[pos] != t:
            raise DSLError(f"expected {t!r} at token {pos} in {s!r}")
        pos += 1

    def node():
        nonlocal pos
        if pos >= len(toks):
            raise DSLError(f"unexpected end of expression: {s!r}")
        t = toks[pos]
        pos += 1
        if t.isdigit():
            return int(t)
        if t in FEATURES:
            expect("(")
            n = node()
            expect(")")
            if not isinstance(n, int) or n not in LOOKBACKS:
                raise DSLError(f"{t} lookback must be one of {LOOKBACKS}")
            return (t, n)
        if t in UNARY:
            expect("(")
            x = node()
            expect(")")
            if isinstance(x, int):
                raise DSLError(f"{t} takes an expression, not an integer")
            return (t, x)
        if t in BINARY:
            expect("(")
            x = node()
            expect(",")
            y = node()
            expect(")")
            if isinstance(x, int) or isinstance(y, int):
                raise DSLError(f"{t} takes two expressions")
            return (t, x, y)
        if t in PARAM_TRANSFORMS:
            expect("(")
            x = node()
            expect(",")
            n = node()
            expect(")")
            if isinstance(x, int) or not isinstance(n, int):
                raise DSLError(f"{t}(expr, int) malformed")
            if t == "ts_z" and n not in LOOKBACKS:
                raise DSLError(f"ts_z window must be one of {LOOKBACKS}")
            if t == "delay" and not (1 <= n <= 250):
                raise DSLError("delay must be in [1, 250]")
            return (t, x, n)
        raise DSLError(f"unknown operator {t!r}")

    ast = node()
    if pos != len(toks):
        raise DSLError(f"trailing tokens in {s!r}")
    if isinstance(ast, int):
        raise DSLError("expression is a bare integer")
    return ast


def unparse(ast) -> str:
    if isinstance(ast, int):
        return str(ast)
    op, *args = ast
    return f"{op}({', '.join(unparse(a) for a in args)})"


def complexity(ast) -> int:
    """Number of operator nodes."""
    if isinstance(ast, int):
        return 0
    return 1 + sum(complexity(a) for a in ast[1:])


# ----------------------------------------------------------------------------- evaluation

class Evaluator:
    """Evaluates ASTs on a close-price panel with feature caching.

    px: DataFrame (T x N) of adjusted closes. All features are causal:
    the value at row t uses information up to and including t.
    """

    def __init__(self, px: pd.DataFrame):
        self.px = px
        self.r1 = px.pct_change()
        self.mkt = self.r1.mean(axis=1)  # equal-weight panel return
        self._cache: dict = {}

    def _feature(self, op: str, n: int) -> pd.DataFrame:
        key = (op, n)
        if key in self._cache:
            return self._cache[key]
        px, r1 = self.px, self.r1
        if op == "ret":
            v = px.pct_change(n)
        elif op == "logret":
            v = np.log(px).diff(n)
        elif op == "ma":
            v = px.rolling(n).mean() / px
        elif op == "std":
            v = r1.rolling(n).std()
        elif op == "skew":
            v = r1.rolling(max(n, 5)).skew()
        elif op == "kurt":
            v = r1.rolling(max(n, 5)).kurt()
        elif op == "maxr":
            v = px / px.rolling(n).max()
        elif op == "minr":
            v = px / px.rolling(n).min()
        elif op in ("beta", "corrm"):
            m = self.mkt
            cov = r1.rolling(n).cov(m)
            if op == "beta":
                v = cov.div(m.rolling(n).var(), axis=0)
            else:
                v = cov.div(r1.rolling(n).std().mul(m.rolling(n).std(), axis=0))
        else:  # pragma: no cover
            raise DSLError(op)
        v = v.replace([np.inf, -np.inf], np.nan)
        self._cache[key] = v
        return v

    def eval(self, ast) -> pd.DataFrame:
        if isinstance(ast, int):
            raise DSLError("bare integer")
        op, *args = ast
        if op in FEATURES:
            return self._feature(op, args[0])
        if op == "delay":
            return self.eval(args[0]).shift(args[1])
        if op == "ts_z":
            x = self.eval(args[0])
            mu = x.rolling(args[1]).mean()
            sd = x.rolling(args[1]).std()
            return ((x - mu) / sd).replace([np.inf, -np.inf], np.nan)
        if op == "cs_rank":
            x = self.eval(args[0])
            return x.rank(axis=1, pct=True) - 0.5
        if op == "cs_z":
            x = self.eval(args[0])
            mu = x.mean(axis=1)
            sd = x.std(axis=1)
            return x.sub(mu, axis=0).div(sd, axis=0).replace([np.inf, -np.inf], np.nan)
        if op == "neg":
            return -self.eval(args[0])
        if op == "abs_":
            return self.eval(args[0]).abs()
        if op == "sign":
            return np.sign(self.eval(args[0]))
        if op in BINARY:
            x, y = self.eval(args[0]), self.eval(args[1])
            if op == "add":
                return x + y
            if op == "sub":
                return x - y
            if op == "mul":
                return x * y
            if op == "div":
                return (x / y).replace([np.inf, -np.inf], np.nan)
        raise DSLError(f"unknown op {op}")


# ----------------------------------------------------------------------------- random generation

def random_expr(rng: np.random.Generator, max_depth: int = 3):
    """Sample a random AST from the grammar prior (used by RS and as OPT seed)."""
    def leaf():
        op = FEATURES[rng.integers(len(FEATURES))]
        n = LOOKBACKS[rng.integers(len(LOOKBACKS))]
        return (op, int(n))

    def rec(depth):
        if depth >= max_depth:
            return leaf()
        u = rng.random()
        if u < 0.35:
            return leaf()
        if u < 0.55:
            op = UNARY[rng.integers(len(UNARY))]
            return (op, rec(depth + 1))
        if u < 0.70:
            op = PARAM_TRANSFORMS[rng.integers(len(PARAM_TRANSFORMS))]
            if op == "delay":
                return (op, rec(depth + 1), int(rng.integers(1, 21)))
            return (op, rec(depth + 1), int(LOOKBACKS[rng.integers(len(LOOKBACKS))]))
        op = BINARY[rng.integers(len(BINARY))]
        return (op, rec(depth + 1), rec(depth + 1))

    return rec(0)


def mutate(ast, rng: np.random.Generator):
    """Mutate an AST: perturb a lookback, swap an operator, or replace a subtree."""
    nodes = []

    def collect(a, path=()):
        if isinstance(a, int):
            return
        nodes.append(path)
        for i, ch in enumerate(a[1:], start=1):
            collect(ch, path + (i,))

    collect(ast)
    target = nodes[rng.integers(len(nodes))] if nodes else ()

    def rebuild(a, path):
        if path == target:
            u = rng.random()
            if isinstance(a, tuple) and a[0] in FEATURES and u < 0.5:
                return (a[0], int(LOOKBACKS[rng.integers(len(LOOKBACKS))]))
            if isinstance(a, tuple) and a[0] in FEATURES and u < 0.8:
                return (FEATURES[rng.integers(len(FEATURES))], a[1])
            return random_expr(rng, max_depth=2)
        if isinstance(a, int):
            return a
        return tuple([a[0]] + [rebuild(ch, path + (i,)) for i, ch in enumerate(a[1:], start=1)])

    return rebuild(ast, ())


def crossover(a, b, rng: np.random.Generator):
    """Replace a random subtree of a with a random subtree of b."""
    def subtrees(t):
        out = []

        def rec(x):
            if isinstance(x, int):
                return
            out.append(x)
            for ch in x[1:]:
                rec(ch)

        rec(t)
        return out

    donor_pool = subtrees(b)
    if not donor_pool:
        return a
    donor = donor_pool[rng.integers(len(donor_pool))]
    paths = []

    def collect(x, path=()):
        if isinstance(x, int):
            return
        paths.append(path)
        for i, ch in enumerate(x[1:], start=1):
            collect(ch, path + (i,))

    collect(a)
    target = paths[rng.integers(len(paths))]

    def rebuild(x, path):
        if path == target:
            return donor
        if isinstance(x, int):
            return x
        return tuple([x[0]] + [rebuild(ch, path + (i,)) for i, ch in enumerate(x[1:], start=1)])

    return rebuild(a, ())
