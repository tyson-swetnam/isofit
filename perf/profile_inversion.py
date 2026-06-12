#!/usr/bin/env python3
"""
Phase 0 profiling harness: measure the per-pixel optimal-estimation inversion.

Loads an ISOFIT run config (e.g. the Pasadena examples), builds the forward
model / inversion / IO objects exactly as core.isofit.Worker does, then runs
the same spectrum through Inversion.invert repeatedly under cProfile to get a
stable per-function cost breakdown without Ray or multiprocessing noise.

Usage:
    python perf/profile_inversion.py CONFIG.json [-n 10] [-o out.pstats]
"""

import argparse
import cProfile
import io as _io
import logging
import pstats
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", help="ISOFIT run config JSON")
    ap.add_argument("-n", "--repeats", type=int, default=10)
    ap.add_argument("-o", "--output", default=None, help="pstats dump path")
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--col", type=int, default=0)
    ap.add_argument("--top", type=int, default=25, help="rows of stats to print")
    args = ap.parse_args()

    logging.basicConfig(level=logging.ERROR)

    from isofit.configs import configs
    from isofit.core.fileio import IO
    from isofit.core.forward import ForwardModel
    from isofit.inversion.inverse import Inversion

    t0 = time.time()
    config = configs.create_new_config(args.config)
    config.get_config_errors()
    fm = ForwardModel(config)
    iv = Inversion(config, fm)
    io = IO(config, fm)
    t_setup = time.time() - t0
    print(f"setup (config + ForwardModel/LUT + Inversion + IO): {t_setup:.2f}s")
    print(f"  n_meas={fm.n_meas}  nstate={fm.nstate}  statevec RT={fm.idx_RT}")

    input_data = io.get_components_at_index(args.row, args.col)
    meas, geom = input_data.meas, input_data.geom

    # Warmup: numba compilation, LUT interpolator caches, BLAS init
    t0 = time.time()
    states = iv.invert(meas, geom)
    print(f"warmup inversion: {time.time() - t0:.3f}s ({len(states)} trajectory pts)")

    # Timed, unprofiled (true wall clock)
    t0 = time.time()
    for _ in range(args.repeats):
        iv.invert(meas, geom)
    wall = (time.time() - t0) / args.repeats
    print(f"mean inversion wall time over {args.repeats} runs: {wall*1000:.1f} ms")

    # Profiled
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(args.repeats):
        states = iv.invert(meas, geom)
    pr.disable()

    # Include the output side once (posterior covariance products etc.)
    pr_out = cProfile.Profile()
    pr_out.enable()
    io.write_spectrum(args.row, args.col, states, fm, iv)
    pr_out.disable()

    for label, prof, sort in [
        ("INVERSION by cumulative", pr, "cumulative"),
        ("INVERSION by tottime", pr, "tottime"),
        ("WRITE_SPECTRUM by cumulative", pr_out, "cumulative"),
    ]:
        s = _io.StringIO()
        st = pstats.Stats(prof, stream=s).sort_stats(sort)
        st.print_stats(args.top)
        print(f"\n{'='*80}\n{label}\n{'='*80}")
        # strip the long path prefixes for readability
        text = s.getvalue().replace(str(Path.cwd()) + "/", "")
        print(text)

    if args.output:
        pstats.Stats(pr).dump_stats(args.output)
        print(f"pstats written to {args.output}")


if __name__ == "__main__":
    main()
