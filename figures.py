"""Figures for the post. Palette validated with the dataviz validator
(slots blue #2a78d6 / orange #eb6834 / aqua #1baf7a; all checks pass, light mode;
aqua carries a contrast WARN, so every aqua mark is directly labelled).

Usage: python3 figures.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS, FIGS = "results", "figures"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8880"
SURFACE = "#fcfcfb"
GRID = "#e6e5e0"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10.5,
    "axes.edgecolor": GRID, "axes.linewidth": 1.0,
    "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.8,
    "legend.frameon": False, "figure.dpi": 160,
})


def _style(ax, title=None, sub=None, xlabel=None, ylabel=None):
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", fontsize=12.5, color=INK, pad=24 if sub else 8,
                     fontweight="semibold")
    if sub:
        ax.text(0, 1.012, sub, transform=ax.transAxes, fontsize=9.5, color=INK2,
                va="bottom")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)


# --------------------------------------------------------------------- fig 1

def fig_surface():
    """Hero: the selection-plus-aggregation surface on zero-alpha data."""
    df = pd.read_csv(os.path.join(RESULTS, "aggregation_surface.csv"))
    piv = df.pivot(index="k", columns="N", values="sr_train")
    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    Ns = list(piv.columns)
    shades = ["#cfe0f5", "#a8c7ec", "#7aa9e2", "#4c8bd9", BLUE]
    for i, N in enumerate(Ns):
        ax.plot(piv.index, piv[N], "-o", color=shades[i], lw=2.0, ms=6,
                mec=SURFACE, mew=1.6, zorder=3, label=f"{N} trials logged")
        ax.annotate(f"N={N}", (piv.index[-1], piv[N].iloc[-1]), textcoords="offset points",
                    xytext=(8, -1), fontsize=9, color=INK2 if i < 4 else BLUE, va="center")
    ax.axhline(0, color=INK3, lw=1.2, zorder=2)
    ax.text(1.05, 0.05, "true Sharpe = 0 everywhere on this data",
            fontsize=9, color=INK2, style="italic")
    try:
        a = float(piv.loc[12, 100]); b = float(piv.loc[1, 400])
        ax.annotate("", xy=(12, a), xytext=(1, b),
                    arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1.4,
                                    ls=(0, (5, 3)), shrinkA=7, shrinkB=7), zorder=6)
        for xx, yy in ((12, a), (1, b)):
            ax.plot(xx, yy, "o", ms=11, mfc="none", mec=ORANGE, mew=2.0, zorder=7)
        ax.annotate(f"100 backtests, top 12 blended: {a:.2f}\n400 backtests, single best: {b:.2f}",
                    xy=(5.0, (a + b) / 2 - 0.30), fontsize=9, color=ORANGE, ha="center",
                    va="top", zorder=7)
    except Exception:
        pass
    _style(ax, "The same log, two very different numbers",
           "Reported in-sample Sharpe of a top-k-of-N composite, on synthetic panels with zero true predictability",
           "legs blended into the reported book (k)", "reported in-sample Sharpe")
    ax.set_xticks(list(piv.index))
    ax.set_xlim(0.4, 22.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig1_surface.png"), bbox_inches="tight")
    plt.close(fig)
    print("fig1 ok")


# --------------------------------------------------------------------- fig 2

def fig_arms():
    """In-sample vs out-of-sample by arm, synthetic (truth=0) and real."""
    B = pd.read_csv(os.path.join(RESULTS, "books.csv"))
    B = B[B.rule == "argmax"]
    syn = B[(B.setting == "SYN0") & ~((B.arm == "AGENT") & (B.ckpt == "B"))]
    real = B[B.setting == "REAL"]
    order = ["CANON-SAMPLER", "OPT-soft", "OPT-medium", "OPT-hard", "AGENT"]
    names = {"CANON-SAMPLER": "canon sampler\n(no feedback)", "OPT-soft": "evolutionary\n(soft)",
             "OPT-medium": "evolutionary\n(medium)", "OPT-hard": "evolutionary\n(hard)",
             "AGENT": "LLM agent"}
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), sharey=True)
    for ax, (d, ttl, sub) in zip(axes, [
            (syn, "Synthetic panels — true Sharpe is zero",
             "12 zero-alpha panels; every point of in-sample Sharpe is selection"),
            (real, "Real panel — NASDAQ 2018–2023",
             "train 2018–21, holdout 2022–23; one shared out-of-sample path")]):
        g = d.groupby("arm").agg(is_m=("sr_is", "mean"), oos_m=("sr_oos", "mean"),
                                 n=("sr_is", "size"),
                                 is_se=("sr_is", lambda s: s.std() / np.sqrt(len(s))),
                                 oos_se=("sr_oos", lambda s: s.std() / np.sqrt(len(s))))
        arms = [a for a in order if a in g.index]
        y = np.arange(len(arms))
        for i, a in enumerate(arms):
            r = g.loc[a]
            col = ORANGE if a == "AGENT" else BLUE
            ax.plot([r.oos_m, r.is_m], [i, i], color=GRID, lw=3, zorder=1,
                    solid_capstyle="round")
            ax.errorbar(r.is_m, i, xerr=r.is_se, fmt="o", color=col, ms=9,
                        mec=SURFACE, mew=1.8, ecolor=col, elinewidth=1.6,
                        capsize=3, zorder=4)
            ax.errorbar(r.oos_m, i, xerr=r.oos_se, fmt="o", color=SURFACE, ms=9,
                        mec=col, mew=2.2, ecolor=col, elinewidth=1.6,
                        capsize=3, zorder=4)
            ax.annotate(f"{r.is_m:.2f}", (r.is_m, i), textcoords="offset points",
                        xytext=(0, 11), ha="center", fontsize=9, color=INK)
            ax.annotate(f"{r.oos_m:+.2f}", (r.oos_m, i), textcoords="offset points",
                        xytext=(0, -17), ha="center", fontsize=9, color=INK2)
        ax.axvline(0, color=INK3, lw=1.1, zorder=2)
        ax.set_yticks(y)
        ax.set_yticklabels([names[a] for a in arms], fontsize=9.5)
        ax.set_ylim(-0.7, len(arms) - 0.3)
        _style(ax, ttl, sub, "annualized Sharpe of the reported book")
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
    axes[0].plot([], [], "o", color=ORANGE, mec=SURFACE, mew=1.6, label="in sample (filled)")
    axes[0].plot([], [], "o", color=SURFACE, mec=ORANGE, mew=2.2, label="out of sample (hollow)")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig2_arms.png"), bbox_inches="tight")
    plt.close(fig)
    print("fig2 ok")


# --------------------------------------------------------------------- fig 3

def fig_nstar():
    """What the log says vs what the selection is worth."""
    L = pd.read_csv(os.path.join(RESULTS, "run_logs.csv"))
    L = L[~L.run.str.startswith("AGB-")]          # primary checkpoint only
    L["censored"] = L.n_star >= 1e7
    pools = {}
    for p in ["SYN0-00", "REAL"]:
        pools[p] = [r["sharpe"] for r in
                    json.load(open(os.path.join(RESULTS, f"null_pool_{p}.json")))]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4))

    ax = axes[0]
    arms = ["CANON-SAMPLER", "OPT-soft", "OPT-medium", "OPT-hard", "AGENT"]
    lbl = {"CANON-SAMPLER": "canon sampler", "OPT-soft": "evo (soft)",
           "OPT-medium": "evo (medium)", "OPT-hard": "evo (hard)", "AGENT": "LLM agent"}
    x = np.arange(len(arms))
    Ls = L[L.panel.str.startswith("SYN0")]
    logged = [Ls[Ls.arm == a].n_logged.median() for a in arms]
    ax.bar(x, logged, width=0.6, color=[ORANGE if a == "AGENT" else BLUE for a in arms],
           zorder=3)
    for i, v in enumerate(logged):
        ax.annotate(f"{v:.0f}", (i, v), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=9.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl[a] for a in arms], fontsize=9, rotation=18, ha="right")
    _style(ax, "What the harness log records",
           "median unique backtests per run, zero-alpha panels", None, "backtests logged")

    # Right: each run's best in-sample Sharpe against the best its OWN panel's
    # 1,500-draw random pool reached. Positive means the searcher went beyond
    # anything blind sampling of the same grammar found on that panel.
    ax = axes[1]
    pmax = {f"SYN0-{i:02d}": max(r["sharpe"] for r in json.load(
        open(os.path.join(RESULTS, f"null_pool_SYN0-{i:02d}.json")))) for i in range(12)}
    Ls = Ls.copy()
    Ls["excess"] = Ls.best_is_sr - Ls.panel.map(pmax)
    rng = np.random.default_rng(4)
    strip = ["CANON-SAMPLER", "OPT-soft", "OPT-medium", "OPT-hard", "AGENT"]
    for i, a in enumerate(strip):
        v = Ls[Ls.arm == a].excess.values
        if not len(v):
            continue
        col = ORANGE if a == "AGENT" else BLUE
        ax.plot(v, i + rng.normal(0, 0.07, len(v)), "o", color=col, ms=7,
                mec=SURFACE, mew=1.2, alpha=0.85, zorder=4)
        ax.plot(np.median(v), i, "|", color=INK, ms=20, mew=2, zorder=5)
        ax.annotate(f"{(v > 0).sum()}/{len(v)} beat the pool", (max(v) + 0.06, i),
                    fontsize=8.8, color=INK2, va="center")
    ax.axvline(0, color=INK3, lw=1.3, zorder=3)
    ax.set_yticks(range(len(strip)))
    ax.set_yticklabels([lbl[a] for a in strip], fontsize=9)
    ax.set_ylim(-0.6, len(strip) - 0.4)
    ax.set_xlim(-1.35, 1.5)
    _style(ax, "What each searcher reached past it",
           "each run's best in-sample Sharpe minus the best of its own panel's 1,500 random draws",
           "excess over the panel's random-search maximum", None)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig3_nstar.png"), bbox_inches="tight")
    plt.close(fig)
    print("fig3 ok")


# --------------------------------------------------------------------- fig 4

def fig_canon():
    """Selected picks decay; unselected canon does not."""
    cc = json.load(open(os.path.join(RESULTS, "canon_composite.json")))
    real = pd.read_csv(os.path.join(RESULTS, "books.csv"))
    real = real[(real.rule == "argmax") & (real.setting == "REAL")]
    ag = real[real.arm == "AGENT"]
    op = real[real.arm.str.startswith("OPT")]
    rows = [("12 published anomalies\n(no selection)", cc["composite_sr_is"],
             cc["composite_sr_oos"], AQUA),
            ("evolutionary search\n(selected, 3 settings)", op.sr_is.mean(), op.sr_oos.mean(), BLUE),
            ("LLM agent\n(selected)", ag.sr_is.mean(), ag.sr_oos.mean(), ORANGE)]
    # Small multiples: the three books converge out of sample, so a single slope
    # panel collides. One panel per book keeps every label legible.
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 4.3), sharey=True)
    for ax, (lab, a, b, c) in zip(axes, rows):
        ax.plot([0, 1], [a, b], "-o", color=c, lw=2.6, ms=10, mec=SURFACE, mew=2,
                zorder=3, clip_on=False)
        ax.annotate(f"{a:.2f}", (0, a), textcoords="offset points", xytext=(0, 13),
                    ha="center", fontsize=11, color=INK, fontweight="semibold")
        ax.annotate(f"{b:+.2f}", (1, b), textcoords="offset points", xytext=(0, 13),
                    ha="center", fontsize=11, color=INK, fontweight="semibold")
        ax.annotate(f"{b - a:+.2f}", (0.5, (a + b) / 2), textcoords="offset points",
                    xytext=(0, -20), ha="center", fontsize=9.5, color=c)
        ax.axhline(0, color=INK3, lw=1.1, zorder=2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["train\n2018–21", "holdout\n2022–23"], fontsize=9.5)
        ax.set_xlim(-0.35, 1.35)
        ax.set_title(lab, loc="left", fontsize=10.5, color=c, pad=26,
                     fontweight="semibold")
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("annualized Sharpe", fontsize=10)
    axes[0].set_ylim(-0.5, 3.5)
    fig.suptitle("Selection decays across this boundary. The unselected canon did not.",
                 x=0.005, y=1.10, ha="left", fontsize=12.5, color=INK,
                 fontweight="semibold")
    fig.text(0.005, 1.02, "Same panel, same backtester, same holdout; each book is one "
             "equal-weight composite", fontsize=9.5, color=INK2, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig4_canon.png"), bbox_inches="tight")
    plt.close(fig)
    print("fig4 ok")


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    fig_arms()
    fig_nstar()
    fig_canon()
    try:
        fig_surface()
    except FileNotFoundError:
        print("fig1 skipped (aggregation surface not ready)")
