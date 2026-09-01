"""Panel preparation and custody.

REAL: skfolio NASDAQ panel (adjusted closes, 1,455 names, 2018-01-02 to 2023-05-31).
Names with median price < $5 dropped (as in the August 2026 post). The panel is
survivorship-conditioned by construction (fully dense over 5.4y); levels are biased,
within-panel comparisons are the objects of interest.

Custody: the file exposed to search arms (panels/) contains the TRAIN slice only,
values-only, integer row/column indices, no dates or tickers. The OOS slice lives
in holdout/ and is touched only by analysis code after runs are complete.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd

REAL_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "nasdaq.csv.gz")
TRAIN_END = "2021-12-31"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_real(out_dir="panels", hold_dir="holdout"):
    px = pd.read_csv(REAL_SRC, index_col=0, parse_dates=True)
    keep = px.median() >= 5.0
    px = px.loc[:, keep]
    dates = px.index
    train_len = int((dates <= TRAIN_END).sum())
    arr = px.values.astype(np.float64)
    np.savez_compressed(os.path.join(out_dir, "REAL.npz"), px=arr[:train_len])
    np.savez_compressed(os.path.join(hold_dir, "REAL.npz"), px=arr, train_len=train_len)
    meta = {
        "source": "skfolio nasdaq_dataset (github.com/skfolio/skfolio-datasets)",
        "source_sha256": sha256(REAL_SRC),
        "names_total": 1455,
        "names_kept": int(keep.sum()),
        "filter": "median adjusted close >= $5",
        "dates": [str(dates[0].date()), str(dates[-1].date())],
        "train_len_days": train_len,
        "oos_len_days": int(len(dates) - train_len),
        "train_end": TRAIN_END,
        "note": "panels/REAL.npz is train-only, values-only, integer-indexed (no dates/tickers)",
    }
    with open(os.path.join(out_dir, "REAL.meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_train(panel_id, panel_dir="panels") -> pd.DataFrame:
    a = np.load(os.path.join(panel_dir, f"{panel_id}.npz"))["px"]
    return pd.DataFrame(a)


def load_full(panel_id, hold_dir="holdout"):
    z = np.load(os.path.join(hold_dir, f"{panel_id}.npz"))
    px = pd.DataFrame(z["px"])
    return px, int(z["train_len"])


def make_syn_panels(arm: str, n: int, seed0: int, kappa_rev=0.0, kappa_anti=0.0,
                    out_dir="panels", hold_dir="holdout"):
    """Generate n synthetic panels for an arm; train slice to panels/, full to holdout/."""
    from gen import make_panel

    manifest = []
    for i in range(n):
        seed = seed0 + i
        px, split = make_panel(seed, kappa_rev=kappa_rev, kappa_anti=kappa_anti)
        pid = f"{arm}-{i:02d}"
        np.savez_compressed(os.path.join(out_dir, f"{pid}.npz"), px=px.values[:split])
        np.savez_compressed(os.path.join(hold_dir, f"{pid}.npz"), px=px.values, train_len=split)
        manifest.append({"panel": pid, "seed": seed,
                         "kappa_rev": kappa_rev, "kappa_anti": kappa_anti})
    with open(os.path.join(out_dir, f"{arm}.manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest
