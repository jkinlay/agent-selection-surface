"""Twelve canonical price-only anomalies, written in the DSL (as in the August 2026 post).

Used two ways:
  - CANON placebo (REAL): scored train -> OOS with NO selection; the regime +
    post-publication decay yardstick for known signals.
  - CANON-SAMPLER arm: canon templates with jittered parameters, no feedback;
    separates prior-direction from optimization pressure.
"""
import numpy as np

CANON = {
    "mom_12_1":   "sub(ret(250), ret(20))",   # Jegadeesh & Titman (1993) / Carhart 12-1
    "str_1m":     "neg(ret(20))",             # Jegadeesh (1990) one-month reversal
    "str_1w":     "neg(ret(5))",              # one-week reversal
    "lowvol_60":  "neg(std(60))",             # Ang, Hodrick, Xing & Zhang (2006)
    "lowvol_20":  "neg(std(20))",
    "bab_120":    "neg(beta(120))",           # Frazzini & Pedersen (2014), rank version
    "high_52w":   "maxr(250)",                # George & Hwang (2004)
    "iskew_120":  "neg(skew(120))",           # Boyer, Mitton & Vorkink (2010)
    "iskew_60":   "neg(skew(60))",
    "trend_120":  "logret(120)",              # Moskowitz, Ooi & Pedersen (2012), cs version
    "int_mom":    "sub(ret(250), ret(120))",  # intermediate momentum (Novy-Marx 2012)
    "mom_accel":  "sub(ret(60), ret(120))",
}

_LOOKS = [3, 5, 10, 15, 20, 40, 60, 120, 250]


def jitter_template(rng: np.random.Generator) -> str:
    """One canon-shaped expression with jittered lookbacks / an optional wrapper."""
    key = list(CANON)[rng.integers(len(CANON))]
    e = CANON[key]
    out, i = [], 0
    while i < len(e):
        if e[i].isdigit():
            j = i
            while j < len(e) and e[j].isdigit():
                j += 1
            n = int(e[i:j])
            idx = _LOOKS.index(n)
            idx = int(np.clip(idx + rng.integers(-1, 2), 0, len(_LOOKS) - 1))
            out.append(str(_LOOKS[idx]))
            i = j
        else:
            out.append(e[i])
            i += 1
    e2 = "".join(out)
    u = rng.random()
    if u < 0.3:
        e2 = f"cs_rank({e2})"
    elif u < 0.45:
        w = _LOOKS[rng.integers(4, len(_LOOKS))]
        e2 = f"ts_z({e2}, {w})"
    return e2
