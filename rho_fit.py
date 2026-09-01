"""Fit the correlated-average form to the aggregation surface.

The gain from blending k selected legs is sqrt(k / (1 + (k-1) rho_bar)), not
sqrt(k), because the legs are correlated. Fitted column by column (fixed N),
over the monotone region k <= 12.

Usage: python3 rho_fit.py
"""
import json

import numpy as np
import pandas as pd

S = pd.read_csv("results/aggregation_surface.csv")
piv = S.pivot(index="k", columns="N", values="sr_train")
grid = np.arange(0.02, 0.99, 0.002)
out = {}
for kmax in (12, 20):
    sub = piv.loc[piv.index <= kmax]
    ks = np.array(sub.index, float)
    for N in sub.columns:
        obs = (sub[N] / sub.loc[1, N]).values
        rho = float(min(grid, key=lambda r: ((np.sqrt(ks / (1 + (ks - 1) * r)) - obs) ** 2).sum()))
        out[f"kmax{kmax}_N{N}"] = {"rho": round(rho, 3), "ceiling": round(1 / np.sqrt(rho), 2)}
json.dump(out, open("results/rho_fit.json", "w"), indent=2)
for k, v in out.items():
    print(f"{k:14s} rho {v['rho']:.2f}  ceiling {v['ceiling']:.2f}x")
