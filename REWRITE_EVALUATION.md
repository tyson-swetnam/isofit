# ISOFIT Performance Evaluation: Rewrite vs. Optimize

*Working document — evaluation of whether to port ISOFIT to a faster language
(Rust / CUDA / C++ / Fortran / Go) or pursue performance improvements within
Python. Based on a code-level review of ISOFIT 3.7.5 (~30k lines of Python,
104 files).*

---

## TL;DR

**A full rewrite in another language is not the high-leverage move here.**
ISOFIT is already a thin Python orchestration layer over compiled code: the
floating-point work runs in LAPACK/BLAS (via NumPy/SciPy), a Numba-JIT'd
interpolation kernel, a PyTorch neural emulator, and external Fortran/C
radiative-transfer executables. Rewriting 30k lines of mission-validated
science code to call the *same* LAPACK from Rust or C++ would cost
person-years and plausibly return only ~2–3× on the stages it touches.

The headroom that actually exists, in order of return on investment:

1. **Algorithmic acceleration** (10–100×) — already being built by the ISOFIT
   team itself, in Python: the `analytical_line` workflow, spatially
   constrained atmosphere estimation, and Gaussian-process emulators
   ("Accelerated Optimal Estimation", Susiluoto et al., *Remote Sensing*
   2025). Much of this is in this repo today.
2. **Batched GPU inversion** (est. 10–50× for the inversion stage) — the
   per-pixel solves are millions of independent, identically-shaped small
   nonlinear least-squares problems: ideal for `jax.vmap`/`torch.vmap`
   batching on GPU. This means rewriting ~2–4k lines of math in JAX/PyTorch,
   not 30k lines in Rust.
3. **Targeted Python/solver optimizations** (est. 2–5×) — exploit the special
   structure of the Jacobian, use Cholesky instead of eigendecomposition for
   SPD matrices, reduce per-iteration recomputation, share LUT memory across
   workers.
4. **Native kernel for the per-pixel solver** (Rust/PyO3 or C++/pybind11) —
   only if 1–3 prove insufficient, and only for `inverse.py` + the forward
   model hot path, keeping all I/O, configuration, and orchestration in
   Python.

**Recommended next step:** profile before deciding anything — run one of the
packaged examples under `py-spy`/`cProfile` and get a measured stage-by-stage
and function-by-function cost breakdown. Everything below about *where* time
goes is derived from code structure, not measurement, and should be confirmed.

---

## 1. What ISOFIT actually computes

ISOFIT performs **atmospheric correction by Bayesian optimal estimation**
(Rodgers 2000): given the radiance an imaging spectrometer measured at each
pixel, it inverts a physics-based forward model to recover surface
reflectance (~300–500 wavelength channels) plus atmospheric parameters
(aerosol optical depth, water vapor), with full posterior uncertainties.

```
state vector x  =  [ reflectance per channel (~300–500) | AOD, H2O (2–5) | instrument (0–few) ]

forward model F(x):  surface model → radiative transfer (LUT lookup) → instrument model
                     (forward.py: ForwardModel.calc_meas, ~250–400 wavelengths)

per-pixel inversion: minimize  || F(x) - measured ||²_Seps  +  || x - x_prior ||²_Sa
                     via scipy.optimize.least_squares (TRF), max 20 evaluations
                     (inversion/inverse.py:296-398)
```

A full flight line is processed by the `apply_oe` workflow in three stages:

1. **Segmentation** — SLIC superpixels over the scene
   (`utils/segment.py`, default segment size ~40 px).
2. **Full OE inversion on superpixel means only** — i.e. on roughly 1/40th
   of the pixels. This is the expensive Bayesian solve.
3. **Extrapolation to every pixel** — either the *empirical line* (per-band
   linear regression against k-nearest superpixel solutions,
   `utils/empirical_line.py`) or the *analytical line* (interpolate the
   atmospheric state spatially, then solve each pixel's reflectance in
   closed form, `utils/analytical_line.py`).

Before any of that, a **look-up table (LUT) of radiative-transfer quantities
is built once per scene**: thousands of runs of an external RT code (6SV —
Fortran; MODTRAN — proprietary; or the sRTMnet neural emulator, a PyTorch MLP
that runs batched on GPU/CPU), gridded over ~9 dimensions (AOD, H2O, solar &
view geometry, elevation, …) and stored as netCDF
(`radiative_transfer/radiative_transfer_engine.py`, `luts.py`).

## 2. Where the time and memory go (code-derived, to be confirmed by profiling)

| Stage | What runs | Bound by |
|---|---|---|
| LUT build | Thousands of subprocess calls to 6SV/MODTRAN (Ray-parallel), or sRTMnet PyTorch batches | **External binaries / GPU** — not Python |
| Superpixel OE inversion | Per-pixel `scipy.least_squares`: LUT interpolation (Numba kernel, `common.py:71`), dense matmuls, `eigh`-based covariance inversions (`common.py:421`), TRF's internal factorization of a ~(700×430) Jacobian per iteration | **LAPACK/BLAS flops + Python glue per iteration** |
| Empirical/analytical line | KD-tree queries + small per-band regressions over *every* pixel (Ray, row-chunked) | **Memory bandwidth + memmap I/O** |
| I/O | ENVI memory-mapped reads/writes, buffered per row (`core/fileio.py`) | Disk; row-granular |

Key structural facts found in the code:

- **The hot kernels are already compiled.** LUT interpolation is a Numba
  `@njit` multilinear kernel and is the *default* interpolator
  (`mlg_numba`, `configs/sections/radiative_transfer_config.py:409`). Matrix
  work is NumPy→LAPACK. The sRTMnet emulator is PyTorch with CUDA/MPS
  support (`engines/sRTMnet.py:116`). The interpreted-Python share of a
  pixel's runtime is the orchestration between these calls.
- **Parallelism is process-based via Ray**, one BLAS thread per worker
  (`MKL_NUM_THREADS=1`/`OMP_NUM_THREADS=1` pinned in `core/isofit.py`),
  work distributed as chunked pixel lists to an actor pool
  (`core/isofit.py:200-251`). Embarrassingly parallel across pixels; scaling
  is eventually limited by per-worker memory duplication (LUT ~0.5–2 GB per
  worker) and memmap contention, not arithmetic.
- **Per-pixel linear algebra is generic where it could be structured.** The
  Jacobian K (`forward.py:330-438`) is analytically assembled and is mostly
  a per-channel-diagonal surface block plus a few dense atmosphere columns —
  but it is handed to a dense TRF solver that factorizes it as a full
  (n_meas+n_state)×n_state matrix every iteration. Covariance inversions use
  `eigh` with an xxhash result cache (`common.py:421-481`); a Cholesky
  factorization would be ~3× cheaper where matrices are SPD.
- **Existing accelerations already trade algorithm for speed**: the
  superpixel + empirical/analytical-line design exists precisely because full
  OE on every pixel is too slow; `engines/kernel_flows.py` (Gaussian-process
  emulator) and the AOE work continue that trajectory upstream.

## 3. Why "rewrite it in X" has limited headroom

- **Amdahl's law, twice.** The LUT build is external Fortran/C executables or
  a GPU neural net — language of the wrapper is irrelevant. File I/O is
  disk-bound. Only the inversion and extrapolation stages would speed up.
- **The flops already run at native speed.** A Rust/C++ port would call the
  same BLAS/LAPACK. Realistic gain is removing per-iteration Python overhead
  and temporaries: ~1.5–3× on the stages touched, unless the port *also*
  changes the algorithm (structured solver, batching) — and those changes
  can be made in Python/JAX first, where they're cheaper to validate.
- **Validation burden dominates the cost.** ISOFIT is the operational
  atmospheric-correction code for NASA EMIT and AVIRIS-class missions. Any
  port must reproduce mission-validated products to tight numerical
  tolerance, across three RT engines, several surface models, and a large
  configuration space. That, not the coding, is the real person-year cost.
- **Fork divergence.** Upstream (`isofit/isofit`) is active (recent
  commits include LUT-interpolation speedups, memory tracking, lazy CLI).
  A rewrite freezes you out of ongoing science development like AOE and
  multisurface retrievals.

## 4. Language-by-language assessment

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| **Python + JAX/PyTorch (batched GPU)** | Per-pixel solves are identical-shape & independent → `vmap` to GPU; autodiff replaces hand Jacobians; XLA CPU fallback; only the math core (~2–4k LOC) is rewritten; stays in ecosystem | GPU memory management; iterative solvers under `vmap` need care (fixed iteration counts); numerics in float32 vs float64 need validation | **Best effort/return ratio** |
| **Rust (PyO3 kernel)** | Memory safety; true shared-memory threading (one LUT copy, no Ray on single node); incremental adoption via PyO3; great deployment | Thin scientific ecosystem (no scipy-equivalent TRF, rough netCDF/ENVI/GDAL bindings); still calls LAPACK for flops; team/community is Python-scientist heavy | Good for a *kernel*, poor for a *codebase* |
| **C++ (pybind11 kernel)** | Eigen/MKL mature; OpenMP; same incremental path as Rust | Memory safety, build complexity; same ROI ceiling as Rust | Same as Rust, with more footguns |
| **Hand-written CUDA** | The workload (millions of small independent solves + 9-D LUT interpolation, which maps to texture hardware) is genuinely GPU-shaped; highest raw ceiling | Most expensive to write/maintain; batched LM + reductions by hand; JAX/Torch capture most of the win for ~10% of the effort | Use CUDA *via* JAX/Torch/cuSOLVER, not by hand |
| **Fortran** | Native kinship with 6SV/MODTRAN; strong array numerics | Everything else: config/JSON/netCDF/ENVI tooling, orchestration, hiring, tests; the Fortran that matters (6SV) is already compiled and called | No |
| **Go** | Easy deployment, good for service orchestration | No serious numerics ecosystem (gonum is thin), GC pauses irrelevant but SIMD/BLAS story weak; cgo to BLAS negates simplicity | No — wrong tool for numerical kernels |
| **Julia** (not on your list, but the honest comparator) | Designed for exactly this (JIT'd numerics, CUDA.jl, strong autodiff/optimization libs); could express the structured solver natively | Same fork-divergence and revalidation burden as any rewrite; smaller operational footprint at JPL/NASA SDS | Strongest *full-rewrite* candidate, still not recommended |

## 5. Recommended phased plan

**Phase 0 — Measure (days).**
Run a packaged example end-to-end (`isofit download examples`, sRTMnet or
6S backend) under `py-spy record` / `cProfile`, plus the built-in
`debug/resource_tracker.py`. Produce: wall-clock per stage (LUT build,
superpixel OE, extrapolation, I/O), top-20 functions, spectra/s/core
(already logged at `core/isofit.py:259`), and per-worker memory. Every
estimate above gets confirmed or corrected here.

**Phase 1 — Cheap wins in place (weeks, est. 2–5× on inversion).**
- Cholesky (`scipy.linalg.cho_factor`) instead of `eigh` for SPD covariance
  inversions; keep `eigh` as fallback on failure.
- Exploit Jacobian structure: the surface block is per-channel diagonal +
  a handful of dense atmosphere columns. A custom Levenberg–Marquardt using
  block elimination / Woodbury drops per-iteration cost from O(m·n²) dense
  factorization to roughly O(n_bands · k²), k = # atmosphere params. This is
  pure NumPy work.
- Hoist iteration-invariant quantities (Seps Cholesky is already computed
  once per pixel; do the same for prior factors where state-independent).
- Share the LUT across workers via Ray's zero-copy object store / mmap
  instead of per-worker copies; batch pixels per task to amortize overhead.

**Phase 2 — Batched GPU inversion prototype (1–2 months, est. 10–50× on the
inversion stage).**
Reimplement `ForwardModel.calc_meas` + a fixed-iteration LM solver in JAX
(or PyTorch), `vmap` across superpixels, LUT interpolation as
`map_coordinates`-style gather. Keep ISOFIT's config, I/O, segmentation, and
empirical/analytical line untouched; validate against current outputs on the
examples (golden-file regression, per-band tolerance budget).

**Phase 3 — Only if needed: native kernel.**
Port `inversion/inverse.py` + forward-model hot path to Rust (PyO3) or C++
(pybind11) with OpenMP/rayon threading and a shared in-memory LUT. Scope:
a few thousand lines. Full-codebase port: **not recommended**.

**Throughout — adopt upstream algorithmic accelerations** (analytical line,
AOE / kernel-flows emulators) rather than re-deriving them: they are where
the 10–100× lives, and they're already in or arriving in this codebase.

## 6. When a full rewrite *would* be justified

- **Onboard/embedded processing** (e.g., reflectance products computed on the
  spacecraft or aircraft): flight software constraints genuinely demand
  C++/Rust, and you'd port a frozen, minimal subset (one RT emulator, one
  surface model, the analytical-line solve) — not the framework.
- **A from-scratch GPU-native operational pipeline** where compute cost at
  mission scale (constellations, daily global coverage) justifies
  person-years of engineering plus a formal revalidation campaign.
- **Python becomes prohibited** in the deployment environment (policy or
  licensing), which is rare.

If none of these apply, the data says: keep Python as the framework, make
the math batched and structured, and put the GPU to work through JAX or
PyTorch rather than hand-rolled kernels.

---

*References: Thompson et al., "Optimal estimation for imaging spectrometer
atmospheric correction," RSE 2018 (the ISOFIT algorithm paper); Susiluoto et
al., "Improved Atmospheric Correction for Remote Imaging Spectroscopy
Missions with Accelerated Optimal Estimation," Remote Sensing 17(22):3719,
2025; Brodrick et al., ISOFIT 3.0, AGU 2023.*
