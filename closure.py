"""The closure: does blind top-k-of-N selection reproduce each arm's reported Sharpe?

k is defined unambiguously as the TOTAL number of additive legs in the reported
book: the book is an equal-weight composite of 3 reported signals, each of which
may itself be a sum of legs, so k = 3 x (mean legs per reported signal). For the
mechanical arms that is 3; for the agent on zero-alpha panels it is about 12.5.
N is the run's own logged trial count. Both are read from results/books.csv.

The surface is interpolated in log N and log k. Both are computed on the same
evaluation basis (see aggregation.py).

Usage: python3 closure.py
"""
import json
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

S = pd.read_csv("results/aggregation_surface.csv")
piv = S.pivot(index="k", columns="N", values="sr_train")
ks = np.array(piv.index, float)
Ns = np.array(piv.columns, float)
f = RegularGridInterpolator((np.log(ks), np.log(Ns)), piv.values,
                            bounds_error=False, fill_value=None)

B = pd.read_csv("results/books.csv")
B = B[(B.rule == "argmax") & (B.setting == "SYN0")]
B = B[~((B.arm == "AGENT") & (B.ckpt == "B"))]
B = B.copy()
B["book_legs"] = 3.0 * B.legs           # 3 reported signals x legs per signal

rows = []
for arm, g in B.groupby("arm"):
    pred = [float(f([[np.log(np.clip(r.book_legs, ks.min(), ks.max())),
                      np.log(np.clip(r.n_logged, Ns.min(), Ns.max()))]])[0])
            for r in g.itertuples()]
    rows.append({"arm": arm, "n_runs": len(g),
                 "median_N_logged": float(g.n_logged.median()),
                 "median_book_legs": float(g.book_legs.median()),
                 "predicted_blind": float(np.mean(pred)),
                 "actual": float(g.sr_is.mean()),
                 "actual_minus_predicted": float(g.sr_is.mean() - np.mean(pred))})
out = pd.DataFrame(rows).sort_values("arm")
print(out.round(2).to_string(index=False))

# the leg axis on its own, at the agent's own median trial count
ag = B[B.arm == "AGENT"]
n_med = float(ag.n_logged.median())
at1 = float(f([[np.log(3.0), np.log(n_med)]])[0])
atk = float(f([[np.log(ag.book_legs.median()), np.log(n_med)]])[0])
mech = B[B.arm.str.startswith("OPT")].sr_is.mean()
summary = {
    "agent_median_N": n_med,
    "agent_median_book_legs": float(ag.book_legs.median()),
    "blind_at_3_legs": round(at1, 3),
    "blind_at_agent_legs": round(atk, 3),
    "leg_axis_alone": round(atk - at1, 3),
    "agent_minus_pooled_mechanical_actual": round(float(ag.sr_is.mean() - mech), 3),
    "by_arm": rows,
}
print("\n" + json.dumps({k: v for k, v in summary.items() if k != "by_arm"}, indent=2))
json.dump(summary, open("results/closure.json", "w"), indent=2)
out.to_csv("results/closure_by_arm.csv", index=False)
