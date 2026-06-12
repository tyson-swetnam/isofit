#!/usr/bin/env python3
"""
Phase 2 validation: batched torch inversion vs ISOFIT's scipy reference.

Builds a batch of synthetic pixels by perturbing a real example spectrum
(random per-pixel gain + noise), inverts them (a) one-at-a-time with
ISOFIT's Inversion.invert (scipy TRF) and (b) with the batched torch LM
prototype, then compares runtimes and solution quality.

Solution quality is judged with ISOFIT's *own* cost function evaluated at
each solver's solution (the OE cost surface has flat valleys, so comparing
states alone overstates differences; see perf/PROFILING.md).

Usage (from an example directory):
    python perf/validate_batched.py configs/modtran/ang20171108t184227_beckmanlawn.json \
        [-B 64] [--ref 16] [--dtype float64] [--device cpu]
"""

import argparse
import copy
import logging
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("-B", "--batch", type=int, default=64)
    ap.add_argument("--ref", type=int, default=16, help="pixels to run through scipy")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0,
                    help="torch CPU threads for the batched solver "
                         "(0 = all cores). ISOFIT pins BLAS to 1 thread at "
                         "import; the batched design uses one process per "
                         "node, so it should use every core.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.ERROR)

    import os

    import torch

    n_threads = args.threads or os.cpu_count()

    from isofit.configs import configs
    from isofit.core.common import eps
    from isofit.core.fileio import IO
    from isofit.core.forward import ForwardModel
    from isofit.inversion.inverse import Inversion
    from isofit.inversion.inverse_simple import invert_simple

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from batched_inversion import BatchedOE

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)

    config = configs.create_new_config(args.config)
    config.get_config_errors()
    fm = ForwardModel(config)
    iv = Inversion(config, fm)
    io = IO(config, fm)
    d = io.get_components_at_index(0, 0)
    meas0, geom0 = d.meas, d.geom

    # ---- synthetic pixel batch ----------------------------------------
    rng = np.random.default_rng(args.seed)
    B = args.batch
    gains = rng.uniform(0.85, 1.15, size=B)
    noise = rng.normal(0.0, 0.002, size=(B, len(meas0))) * meas0[None, :]
    meas_b = meas0[None, :] * gains[:, None] + noise

    # ---- per-pixel setup, mirroring Inversion.invert pre-solver steps -----
    print(f"setup: x0 (invert_simple), component, Seps per pixel for B={B}")
    t0 = time.time()
    x0_b = np.empty((B, fm.nstate))
    ci_b = np.empty(B, dtype=int)
    Seps_win_b = np.empty((B, len(iv.winidx), len(iv.winidx)))
    geoms = []
    for i in range(B):
        g = copy.deepcopy(geom0)
        x0 = invert_simple(fm, meas_b[i], g)
        if iv.config.priors_in_initial_guess:
            sub = np.arange(len(x0))[fm.idx_surf_rfl][iv.outside_ret_windows]
            x0[sub] = fm.surface.xa(x0, g)[sub]
        lo = x0 < fm.bounds[0]
        x0[lo] = fm.bounds[0][lo] + eps
        hi = x0 > fm.bounds[1]
        x0[hi] = fm.bounds[1][hi] - eps
        g.x_surf_init = x0[fm.idx_surface]
        g.x_RT_init = x0[fm.idx_RT]
        ci = fm.surface.component(x0[fm.idx_surface], g)
        Seps = fm.Seps(x0, meas_b[i], g)
        Seps_win = Seps[np.ix_(iv.winidx, iv.winidx)]
        x0_b[i], ci_b[i], Seps_win_b[i] = x0, ci, Seps_win
        geoms.append(g)
    t_setup = time.time() - t0
    print(f"  setup: {t_setup:.1f}s total, {t_setup/B*1000:.1f} ms/pixel")

    # ---- reference: scipy TRF one pixel at a time ----------------------
    nref = min(args.ref, B)
    print(f"reference: scipy Inversion.invert on {nref} pixels")
    iv.invert(meas_b[0], copy.deepcopy(geoms[0]))  # warmup
    x_ref = np.empty((nref, fm.nstate))
    t0 = time.time()
    for i in range(nref):
        states = iv.invert(meas_b[i], geoms[i])
        x_ref[i] = states[-1]
    t_ref = (time.time() - t0) / nref
    print(f"  scipy: {t_ref*1000:.1f} ms/pixel")

    # ---- batched torch -------------------------------------------------
    torch.set_num_threads(n_threads)
    print(f"batched: torch LM on {B} pixels (device={device}, "
          f"dtype={args.dtype}, threads={n_threads})")
    oe = BatchedOE(fm, iv, device=device, dtype=dtype)
    # warmup (compilation/allocator)
    oe.invert(meas_b[:2], geom0, x0_b[:2], ci_b[:2], Seps_win_b[:2], chunk=args.chunk)
    t0 = time.time()
    x_bat = oe.invert(meas_b, geom0, x0_b, ci_b, Seps_win_b, chunk=args.chunk)
    t_bat = (time.time() - t0) / B
    print(f"  batched: {t_bat*1000:.1f} ms/pixel  ({t_ref/t_bat:.1f}x vs scipy)")

    # ---- quality: ISOFIT's own cost at each solution -------------------
    win = iv.winidx
    rfl_idx = fm.idx_surf_rfl
    in_win = np.isin(np.arange(fm.nstate), rfl_idx[np.isin(rfl_idx, win)])

    def isofit_cost(x, i):
        Seps_inv, Seps_inv_sqrt = iv.calc_Seps(x0_b[i], meas_b[i], geoms[i])
        resid, _ = iv.loss_function(x, geoms[i], Seps_inv_sqrt, meas_b[i])
        return float(np.sum(resid**2))

    drfl, datm, costs = [], [], []
    for i in range(nref):
        c_ref = isofit_cost(x_ref[i], i)
        c_bat = isofit_cost(x_bat[i], i)
        costs.append((c_ref, c_bat))
        drfl.append(np.max(np.abs(x_ref[i][in_win] - x_bat[i][in_win])))
        datm.append(np.abs(x_ref[i][fm.idx_RT] - x_bat[i][fm.idx_RT]))
    drfl, datm = np.array(drfl), np.array(datm)
    costs = np.array(costs)

    print("\n--- quality vs scipy reference (windowed reflectance channels) ---")
    print(f"max|drfl| per pixel: median={np.median(drfl):.2e} max={drfl.max():.2e}")
    print(f"|dAOD| median={np.median(datm[:,0]):.2e} max={datm[:,0].max():.2e}")
    print(f"|dH2O| median={np.median(datm[:,1]):.2e} max={datm[:,1].max():.2e}")
    rel = (costs[:, 1] - costs[:, 0]) / costs[:, 0]
    print(f"ISOFIT cost (batched vs scipy): median rel diff={np.median(rel):+.2e}, "
          f"worst={rel.max():+.2e}, batched lower on {np.sum(rel<0)}/{nref}")
    print(f"\ntimings: setup {t_setup/B*1000:.1f} | scipy {t_ref*1000:.1f} | "
          f"batched {t_bat*1000:.1f} ms/pixel")


if __name__ == "__main__":
    main()
