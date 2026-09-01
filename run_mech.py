"""Driver: null pools + mechanical arms for all panels, parallel across panels.

Usage: python3 run_mech.py <group>   where group in {syn0, syna, real}
"""
import sys
from multiprocessing import Pool

from arms import build_null_pool, run_opt, run_canon_sampler


def do_syn0(pid):
    build_null_pool(pid, n=1500, seed=123)
    for setting in ("soft", "medium", "hard"):
        for rep in range(3):
            run_opt(pid, seed=9000 + rep, setting=setting)
    for rep in range(2):
        run_canon_sampler(pid, seed=9500 + rep)
    return pid


def do_syna(pid):
    build_null_pool(pid, n=1500, seed=123)
    for rep in range(3):
        run_opt(pid, seed=9000 + rep, setting="medium")
    for rep in range(2):
        run_canon_sampler(pid, seed=9500 + rep)
    return pid


def do_real(_):
    build_null_pool("REAL", n=2500, seed=123)
    for setting in ("soft", "medium", "hard"):
        for rep in range(5):
            run_opt("REAL", seed=9000 + rep, setting=setting)
    for rep in range(5):
        run_canon_sampler("REAL", seed=9500 + rep)
    return "REAL"


if __name__ == "__main__":
    group = sys.argv[1]
    if group == "syn0":
        pids = [f"SYN0-{i:02d}" for i in range(12)]
        with Pool(6) as p:
            for r in p.imap_unordered(do_syn0, pids):
                print("done", r, flush=True)
    elif group == "syna":
        pids = [f"SYNA05-{i:02d}" for i in range(4)] + [f"SYNA10-{i:02d}" for i in range(4)]
        with Pool(6) as p:
            for r in p.imap_unordered(do_syna, pids):
                print("done", r, flush=True)
    elif group == "real":
        do_real(None)
        print("done REAL", flush=True)
