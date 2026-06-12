# ISOFIT Performance: Phase 0–2 Measured Results

Companion to `../REWRITE_EVALUATION.md`. All numbers measured on a 4-vCPU /
15 GB container (no GPU), Python 3.11, ISOFIT 3.7.6, using the
`20171108_Pasadena` example (`ang20171108t184227_beckmanlawn` config:
AVIRIS-NG, 425 channels, state vector = 425 reflectances + AOT550 + H2OSTR
= 427, retrieval windows cover 349 channels, prebuilt MODTRAN LUT).

Reproduce with:

```bash
isofit download data && isofit download examples     # GitHub-hosted
python -c "from isofit.data.build_examples import Examples, build; \
           build(Examples['Pasadena'], validate=False)"
cd <examples>/20171108_Pasadena
python -c "from isofit.utils import surface_model; \
           surface_model('configs/ang20171108t184227_surface.json')"
python $REPO/perf/profile_inversion.py configs/modtran/ang20171108t184227_beckmanlawn.json
python $REPO/perf/compare_solvers.py   configs/modtran/ang20171108t184227_beckmanlawn.json
python $REPO/perf/validate_batched.py  configs/modtran/ang20171108t184227_beckmanlawn.json -B 256 --ref 16
```

---

## Phase 0 — Where the time actually goes

Per-pixel optimal-estimation inversion, measured with `profile_inversion.py`
(10 repeated inversions of a real spectrum, serial, no Ray):

| Quantity | Measured |
|---|---|
| Mean wall time per inversion | **543 ms** (~5–8 solver evaluations) |
| scipy TRF + LSMR internals | **~85%** of inversion time |
| ISOFIT physics (LUT interpolation, forward model, priors, Jacobian assembly) | ~15% |
| `calc_Seps` + covariance `eigh`, cold cache (once per fresh pixel) | 25 ms |
| End-to-end single-spectrum `isofit run` (incl. Ray + LUT load) | 21.6 s wall, 5.35 s inversion stage |

Findings that killed several "obvious" fixes:

- **`tr_solver='lsmr'` is ISOFIT's config default** (`inversion_config.py:159`).
  Each TRF iteration runs ~430 LSMR iterations = tens of thousands of
  Python-wrapped matvecs on the dense 777×427 Jacobian.
- Switching to `tr_solver='exact'` is **0.87x (slower)** — the dense SVD per
  iteration costs as much as LSMR's iterations. Both converge to equivalent
  optima (AOD differs by ~0.03 in a flat cost valley).
- Capping LSMR inner iterations (`tr_options={'maxiter': 100}`) gives up to
  2.2x but drifts reflectance by ~4e-2 — not safe as a default; usable as a
  per-campaign tunable.
- `eigh` vs Cholesky on the 349² covariance: **11.7 vs 11.6 ms — a wash**
  (OpenBLAS `eigh` is fine at this size; the "Cholesky is 3x cheaper"
  intuition does not hold here).
- The windowed-covariance Python row loop vs `np.ix_`: 1.0 → 0.5 ms.

Conclusion: the per-pixel cost is structural — a generic dense trust-region
solver applied to a Jacobian that is actually (diagonal + 2 columns) plus a
prior block. No in-place micro-optimization can touch the 85%.

## Phase 1 — Safe in-place fixes (merged into the package)

Exact-equivalence changes (final states bit-identical on the example;
`pytest -m unmarked` 32 passed and `-m slow` suites pass):

1. `inversion/inverse.py`: windowed Seps via `np.ix_` instead of a Python
   row loop.
2. `surface/surface_multicomp.py`: cache the constant `dlamb_dsurface` /
   `dLs_dsurface` matrices (a fresh ~425×425 identity + zeros were
   allocated every iteration); copies returned to preserve mutability.
3. `radiative_transfer/radiative_transfer.py`: `concat_rt_outputs()` fast
   path for the single-engine case (~1,600 `np.hstack` calls per inversion
   were pure copy overhead); still returns fresh arrays because
   `get_L_coupled` mutates results in place.

Measured effect: **noise-level on wall time** (533.9 vs 532.5 ms) — kept for
allocation hygiene, honestly reported as not the win. This is the empirical
refutation of "Phase 1 ≈ 2–5x" hoped for in the evaluation doc: the solver
dominates everything.

## Phase 2 — Batched torch inversion (`perf/batched_inversion.py`)

Replaces scipy's per-pixel TRF with a **batched bounded Levenberg-Marquardt**
across pixels, built by introspecting the existing `ForwardModel`/`Inversion`
(same LUT arrays, windows, priors, noise covariances; torch is already an
ISOFIT dependency).

Key structure exploited — per iteration and pixel, the Gauss-Newton system

```
H = K^T Seps_inv K + Sa_inv        g = K^T Seps_inv dmeas + Sa_inv (x - xa)
```

is assembled in O(n_window²) using K = (per-channel diagonal | 2 atmosphere
columns), cached per-component `Sa_inv` (= PᵀP exactly), and the per-pixel
`Seps_inv` fixed at x0 (as ISOFIT does). A batched 427³ Cholesky solves all
pixels at once.

**Exactness**: cost, gradient, and Hessian verified against ISOFIT's own
`loss_function`/`jacobian` to 5e-16 / 3e-15 / 5e-13 relative.

**The hard-won part — bounds.** scipy's `trf` is trust-region *reflective*;
plain clamped LM pins pixels at bounds (every pixel drove AOT550 to its lower
bound and stalled ~1–2% above scipy's cost; small initial damping made it
catastrophically worse, +650%). Two fixes were required:

- Nielsen/Madsen gain-ratio damping (λ adapted by predicted-vs-actual
  reduction), replacing naive multiplicative schedules;
- **active-set refinement**: when a raw step crosses a bound, move those
  coordinates exactly to the bound and re-solve the reduced damped system
  for the rest; coordinates release automatically when the gradient pulls
  inward.

### Results (B = 256 synthetic pixels from the real spectrum: ±15% gain, 0.2% noise)

| Configuration | ms/pixel | vs scipy/core |
|---|---|---|
| scipy `Inversion.invert` (1 core, production setup) | 543–572 | 1.0x |
| batched f64, 1 thread | **209** | **2.7x** |
| batched f64, 4 threads (one process) | **112** | 4.8x vs 1-core scipy; ~1.3x node-level vs 4 Ray workers |
| per-pixel setup (`invert_simple` + Seps, serial numpy) | 11 | — |

f32 on CPU is **not** beneficial (455 ms/px): the Hessian's conditioning
makes f32 Cholesky fail/retry. Use f64.

### Solution quality (16 reference pixels)

| Metric | Value |
|---|---|
| ISOFIT cost at solution, batched vs scipy | **lower on 16/16** (median −0.5%) |
| max abs reflectance difference (windowed channels) | 5.1e-3 (median 4.8e-3) |
| ΔAOT550 | ~0.039 — flat-valley difference: batched settles at the (slightly deeper) bound minimum, TRF is interior-biased. Same magnitude as scipy's own `lsmr` vs `exact` spread (~0.03). |
| ΔH2OSTR | ≤1.7e-2 |

### Deployment guidance

- **CPU**: run one batched process per core (exactly like today's Ray
  workers, each worker batching its pixel block) → **~2.7x node-level
  throughput**. Single-process multithreading scales at only ~47% for these
  batched ops; don't rely on it.
- **GPU**: untested here (no GPU in this container) — the workload is
  batched 427² Cholesky factorizations, batched small matmuls, and LUT
  gathers at B in the thousands, which is squarely what GPUs accelerate.
  The code runs on CUDA via `--device cuda` unchanged; measuring it is the
  next step on GPU hardware.
- At ≥2.7x solver speedup, the serial numpy setup path (11 ms/px:
  `invert_simple`, component selection, `Seps`) becomes ~5% and will
  dominate any GPU run — it is all elementwise math and should be batched
  next.

### Path to production (not done in the prototype)

- Generalize: glint/thermal surfaces, multiple RT engines, per-pixel
  geometry vectors (scalars now), instrument statevectors, `Seps`/`x0`
  batching.
- Integrate as an alternative `Worker.run_set_of_spectra` batch mode behind
  a config flag; golden-file regression versus the scipy path per release.
- Revisit convergence criteria against mission tolerance budgets (current:
  cost-equal-or-better vs scipy at max_iter=20, converges ~8 iterations).
