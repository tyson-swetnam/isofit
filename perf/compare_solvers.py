#!/usr/bin/env python3
"""
Phase 0/1: compare scipy least_squares tr_solver options ('lsmr' vs 'exact')
for the ISOFIT per-pixel inversion, on a real example spectrum.

ISOFIT's config default is tr_solver='lsmr' (inversion_config.py). For the
dense ~(n_meas+n_state) x n_state Jacobians ISOFIT produces, profiling shows
lsmr dominates runtime via tens of thousands of Python-level matvec calls.
This script measures both solvers and verifies they reach the same solution.

Usage:
    python perf/compare_solvers.py CONFIG.json [-n 5]
"""

import argparse
import logging
import time

import numpy as np


def run(iv, meas, geom, n):
    # warmup (numba, caches)
    states = iv.invert(meas, geom)
    t0 = time.time()
    for _ in range(n):
        states = iv.invert(meas, geom)
    dt = (time.time() - t0) / n
    return dt, states[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", help="ISOFIT run config JSON")
    ap.add_argument("-n", "--repeats", type=int, default=5)
    args = ap.parse_args()

    logging.basicConfig(level=logging.ERROR)

    from isofit.configs import configs
    from isofit.core.fileio import IO
    from isofit.core.forward import ForwardModel
    from isofit.inversion.inverse import Inversion

    config = configs.create_new_config(args.config)
    config.get_config_errors()
    fm = ForwardModel(config)
    iv = Inversion(config, fm)
    io = IO(config, fm)
    input_data = io.get_components_at_index(0, 0)
    meas, geom = input_data.meas, input_data.geom

    results = {}
    for solver in ["lsmr", "exact"]:
        iv.hashtable.clear()
        iv.least_squares_params["tr_solver"] = solver
        dt, x = run(iv, meas, geom, args.repeats)
        results[solver] = (dt, x)
        print(f"tr_solver={solver:6s}: {dt*1000:8.1f} ms/inversion")

    x_lsmr, x_exact = results["lsmr"][1], results["exact"][1]
    rfl = fm.idx_surf_rfl
    print(f"\nspeedup exact vs lsmr: {results['lsmr'][0]/results['exact'][0]:.2f}x")
    print(f"max |dx|      (full state): {np.max(np.abs(x_lsmr - x_exact)):.2e}")
    print(f"max |drfl|    (reflectance): {np.max(np.abs(x_lsmr[rfl] - x_exact[rfl])):.2e}")
    print(f"atm (lsmr) : {x_lsmr[fm.idx_RT]}")
    print(f"atm (exact): {x_exact[fm.idx_RT]}")


if __name__ == "__main__":
    main()
