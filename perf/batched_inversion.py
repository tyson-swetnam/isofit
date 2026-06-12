#!/usr/bin/env python3
"""
Phase 2 prototype: batched optimal-estimation inversion with PyTorch.

ISOFIT solves one pixel at a time with scipy.optimize.least_squares; profiling
(see perf/PROFILING.md) shows ~85% of per-pixel time is spent inside scipy's
trust-region machinery on a dense ~(777 x 427) Jacobian. The per-pixel
problems are independent and identically shaped, so they can be solved as one
batched Levenberg-Marquardt iteration stream on CPU or GPU.

The key structural identity exploited here: with the multicomponent surface,
the measurement Jacobian K is a per-channel diagonal plus a few dense
atmosphere columns, the prior whitener satisfies P^T P = Sa_inv (cached per
component), and Seps_inv is fixed per pixel. The Gauss-Newton system

    H = K^T Seps_inv K + Sa_inv          g = K^T Seps_inv dmeas + Sa_inv dx_a

can therefore be assembled in O(n_window^2) per pixel - no dense (777 x 427)
products at all - leaving the batched Cholesky solve as the dominant cost.

This module builds the batched model directly from an existing ISOFIT
``ForwardModel``/``Inversion`` pair, reusing the same LUT arrays, retrieval
windows, surface prior components, and noise covariances. torch is already an
ISOFIT dependency (sRTMnet), so this adds nothing new.

Scope (asserted at construction; out-of-scope configs raise):
  - single RT engine, 'mlg_numba' interpolators, multipart (6c) or 1c modes
  - MultiComponentSurface (Lambertian), no thermal/glint, no instrument states
  - fixed instrument calibration (sample() is a pass-through)
  - no integration grid, no model discrepancy file, no background reflectance

The solver is bounded LM with per-pixel damping: not bit-identical to scipy
TRF, so solutions are validated against ISOFIT's own cost function
(perf/validate_batched.py).
"""

from types import SimpleNamespace

import numpy as np
import torch


# ---------------------------------------------------------------------------
# LUT interpolation
# ---------------------------------------------------------------------------
class TorchLUT:
    """Batched multilinear interpolation over the RT engine's LUT grid.

    Mirrors isofit.core.common.VectorInterpolator (mlg_numba): clamped
    multilinear interpolation, channels in the trailing axis. All LUT keys
    are stacked into one tensor so a batch of points gathers every quantity
    in one fully-vectorized pass (no Python loop over corners).
    """

    def __init__(self, engine, device, dtype):
        self.device = device
        self.dtype = dtype

        interps = {
            k: v
            for k, v in engine.luts.items()
            if getattr(v, "method", None) == 3  # mlg_numba style
        }
        if not interps:
            raise NotImplementedError("expected mlg_numba interpolators")
        first = next(iter(interps.values()))

        self.keys = list(interps)
        self.grids = [
            torch.tensor(g, device=device, dtype=dtype) for g in first.grid_tuples
        ]
        self.n_dim = len(self.grids)
        self.n_chan = first.num_channels

        # (n_keys, n_grid_rows, n_chan)
        data = np.stack(
            [interps[k].gridarrays.reshape(-1, self.n_chan) for k in self.keys]
        )
        self.data = torch.tensor(data, device=device, dtype=dtype)

        shape = first.gridarrays.shape[:-1]
        strides = []
        s = 1
        for d in reversed(shape):
            strides.append(s)
            s *= d
        strides = list(reversed(strides))

        # Precompute corner offsets/sign masks: (2^D, D)
        D = self.n_dim
        bits = ((torch.arange(1 << D)[:, None] >> torch.arange(D)[None, :]) & 1).to(
            device
        )
        self.bits = bits.to(dtype)  # (C, D) 0/1
        sizes = torch.tensor(shape, device=device, dtype=torch.long)
        self.strides_t = torch.tensor(strides, device=device, dtype=torch.long)
        self.sizes = sizes

        # constant values (single-valued LUT keys, method == -1)
        self.const = {
            k: float(v.value)
            for k, v in engine.luts.items()
            if getattr(v, "method", None) == -1
        }

    def __call__(self, points):
        """points: (B, n_dim) -> dict key -> (B, n_chan)"""
        B = points.shape[0]
        idx = torch.empty((B, self.n_dim), device=self.device, dtype=torch.long)
        delta = torch.empty((B, self.n_dim), device=self.device, dtype=self.dtype)
        for d, g in enumerate(self.grids):
            i = torch.searchsorted(g, points[:, d].contiguous(), right=False) - 1
            i = i.clamp(0, max(len(g) - 2, 0))
            idx[:, d] = i
            if len(g) > 1:
                t = (points[:, d] - g[i]) / (g[i + 1] - g[i])
            else:
                t = torch.zeros(B, device=self.device, dtype=self.dtype)
            delta[:, d] = t.clamp(0.0, 1.0)

        # corner weights: (C, B) = prod_d bit ? delta : 1-delta
        w = torch.where(
            self.bits[:, None, :].bool(), delta[None, :, :], 1.0 - delta[None, :, :]
        ).prod(dim=2)
        # corner rows: (C, B)
        corner_idx = torch.minimum(
            idx[None, :, :] + self.bits.long()[:, None, :],
            (self.sizes - 1)[None, None, :],
        )
        rows = (corner_idx * self.strides_t[None, None, :]).sum(dim=2)

        # gather: (K, C*B, nch) -> weighted sum over corners
        g = self.data[:, rows.reshape(-1), :].reshape(
            len(self.keys), w.shape[0], B, self.n_chan
        )
        out = (g * w[None, :, :, None]).sum(dim=1)

        res = {k: out[i] for i, k in enumerate(self.keys)}
        for k, v in self.const.items():
            res[k] = torch.full((B, 1), v, device=self.device, dtype=self.dtype)
        return res


# ---------------------------------------------------------------------------
# Batched forward model + inversion
# ---------------------------------------------------------------------------
class BatchedOE:
    """Batched ISOFIT optimal estimation, mirroring Inversion.invert."""

    def __init__(self, fm, iv, device="cpu", dtype=torch.float64):
        self.device = torch.device(device)
        self.dtype = dtype
        self.fm = fm
        self.iv = iv

        # ---- scope checks ------------------------------------------------
        RT = fm.RT
        if len(RT.rt_engines) != 1:
            raise NotImplementedError("prototype supports a single RT engine")
        self.engine = RT.rt_engines[0]
        if len(fm.idx_instrument):
            raise NotImplementedError("instrument statevector not supported")
        if len(iv.integration_grid):
            raise NotImplementedError("integration grid not supported")
        if fm.model_discrepancy is not None:
            raise NotImplementedError("model discrepancy not supported")
        if not fm.instrument.calibration_fixed or len(fm.instrument.wl_init) != len(
            RT.wl
        ):
            raise NotImplementedError("instrument resampling not supported")
        surf = fm.surface
        if type(surf).__name__ not in ("MultiComponentSurface",):
            raise NotImplementedError(f"surface {type(surf).__name__} not supported")

        t = lambda a: torch.tensor(np.asarray(a), device=self.device, dtype=dtype)

        # ---- LUT + point assembly ----------------------------------------
        self.lut = TorchLUT(self.engine, self.device, dtype)
        self.n_point = self.engine.n_point
        self.x_RT_inds = list(np.atleast_1d(self.engine.indices.x_RT))
        self.geom_inds = dict(self.engine.indices.geom or {})
        czi = self.engine.indices.convert_observer_zenith
        self.convert_obs_zenith = list(np.atleast_1d(czi)) if czi is not None else []

        self.rt_mode = self.engine.rt_mode
        self.multipart = RT.multipart_transmittance
        self.coupling_terms = list(getattr(self.engine, "coupling_terms", []) or [])
        self.solar_irr = t(RT.solar_irr)

        # ---- state vector layout -------------------------------------------
        self.nb = len(surf.idx_lamb)
        self.n_rt = len(fm.idx_RT)
        self.n_state = fm.nstate
        self.lb = t(fm.bounds[0])
        self.ub = t(fm.bounds[1])
        self.eps = 1e-5  # matches isofit.core.common.eps (FD step)

        # ---- retrieval windows ---------------------------------------------
        self.winidx = t(iv.winidx).long()
        self.nw = len(iv.winidx)
        # active columns of the GN system's measurement part:
        # windowed rfl channels first, then the RT params
        self.act = torch.cat(
            [self.winidx, torch.arange(self.n_rt, device=self.device) + self.nb]
        )

        # ---- surface prior components ----------------------------------------
        self.comp_means = t(np.array([m for m in surf.component_means]))
        self.idx_ref = t(surf.idx_ref).long()
        self.normalize = surf.normalize
        Sa_inv, Sa_diagmean = [], []
        for ci in range(surf.n_comp):
            Sa_inv.append(surf.Sa_inv_normalized[ci])
            Sa_diagmean.append(np.mean(np.diag(surf.component_covs[ci])))
        self.comp_Sa_inv = t(np.array(Sa_inv))
        self.comp_Sa_diagmean = t(np.array(Sa_diagmean))

        # RT prior block (state-independent)
        self.rt_prior_mean = t(RT.xa())
        Sa_RT = RT.Sa()
        self.rt_Sa_inv_norm = t(RT.Sa_inv_normalized)
        self.rt_scale = float(np.sqrt(np.mean(np.diag(Sa_RT))))

    # -- pieces ------------------------------------------------------------
    def _geom_consts(self, geom):
        g = SimpleNamespace(
            coszen=float(geom.coszen),
            cos_i=float(geom.cos_i) if geom.cos_i is not None else float(geom.coszen),
            skyview_factor=float(getattr(geom, "skyview_factor", 1.0) or 1.0),
            geom=geom,
        )
        if getattr(geom, "bg_rfl", None) is not None:
            raise NotImplementedError("background reflectance not supported")
        return g

    def _points(self, x_rt, geom):
        B = x_rt.shape[0]
        pt = torch.zeros((B, self.n_point), device=self.device, dtype=self.dtype)
        pt[:, self.x_RT_inds] = x_rt
        for i, key in self.geom_inds.items():
            pt[:, i] = float(getattr(geom, key))
        for i in self.convert_obs_zenith:
            pt[:, i] = 180.0 - pt[:, i]
        return pt

    def _to_rdn(self, v, g):
        if self.rt_mode == "rdn":
            return v
        return v * (self.solar_irr[None, :] * g.coszen / np.pi)

    def calc_rdn(self, x_rt, rho, g):
        """Batched RadiativeTransfer.calc_rdn (Lambertian multicomp case).

        rho: (B, nb). Returns (rdn, parts); parts carries what the analytic
        surface derivative needs.
        """
        r = self.lut(self._points(x_rt, g.geom))
        L_atm = self._to_rdn(r["rhoatm"], g)
        s_alb = r["sphalb"]

        if self.multipart:
            Lc = [self._to_rdn(r[k], g) for k in self.coupling_terms]
            L_dir_dir = Lc[0] / g.coszen * g.cos_i
            L_dif_dir = Lc[1]
            L_dir_dif = Lc[2]  # (/coszen * cos_i_bg) with cos_i_bg == coszen
            L_dif_dif = Lc[3]
            t_down_dir = r["transm_down_dir"]
            L_dif_dir = L_dif_dir * (
                t_down_dir * (g.cos_i / g.coszen)
                + (1.0 - t_down_dir) * g.skyview_factor
            )
            # background topo correction multiplier reduces to 1 (cos_i_bg==coszen)
            eq11 = 1.0 - s_alb * rho
            L_tot = L_dir_dir + L_dif_dir + L_dir_dif + L_dif_dif
            L_dif_dir = L_dif_dir / eq11
            L_dif_dif = L_dif_dif / eq11
            rdn = (
                L_atm
                + (L_dir_dir + L_dif_dir + L_dir_dif + L_dif_dif) * rho
                + (L_tot * (s_alb * rho) * rho) / (1.0 - s_alb * rho)
            )
        else:
            L_tot = self._to_rdn(r["transm_down_dif"], g)
            rdn = L_atm + (L_tot * rho) / (1.0 - s_alb * rho)

        parts = SimpleNamespace(s_alb=s_alb, L_tot=L_tot)
        return rdn, parts

    def forward_fd(self, x, g):
        """One stacked forward pass returning rdn, d(rdn)/d(rfl) diagonal and
        d(rdn)/d(x_RT) columns (FD, mirroring RadiativeTransfer.drdn_dRT)."""
        B = x.shape[0]
        rho = x[:, : self.nb]
        x_rt = x[:, self.nb :]

        # stack [base, +eps e_1, ..., +eps e_nrt] into one batch
        xs = [x_rt]
        for j in range(self.n_rt):
            xp = x_rt.clone()
            xp[:, j] += self.eps
            xs.append(xp)
        rho_rep = rho.repeat(self.n_rt + 1, 1)
        rdn_all, parts = self.calc_rdn(torch.cat(xs, 0), rho_rep, g)
        rdn = rdn_all[:B]
        d_rt = torch.stack(
            [(rdn_all[(j + 1) * B : (j + 2) * B] - rdn) / self.eps
             for j in range(self.n_rt)],
            dim=-1,
        )  # (B, nb, n_rt)

        s_alb = parts.s_alb[:B]
        L_tot = parts.L_tot[:B] if torch.is_tensor(parts.L_tot) else parts.L_tot
        d_rfl = L_tot / (1.0 - s_alb * rho) ** 2  # (B, nb)
        return rdn, d_rfl, d_rt

    # -- prior ---------------------------------------------------------------
    def _norm(self, x):
        ref = x[:, : self.nb][:, self.idx_ref]
        if self.normalize == "Euclidean":
            return torch.linalg.norm(ref, dim=1)
        if self.normalize == "RMS":
            return torch.sqrt(torch.mean(ref**2, dim=1))
        return torch.ones(x.shape[0], device=self.device, dtype=self.dtype)

    def prior_mean(self, x, ci, norm):
        xa = torch.zeros((x.shape[0], self.n_state), device=self.device,
                         dtype=self.dtype)
        xa[:, : self.nb] = self.comp_means[ci] * norm[:, None]
        xa[:, self.nb :] = self.rt_prior_mean[None, :]
        return xa

    def _per_component(self, ci):
        """Yield (component, bool mask) groups; components are few."""
        for c in torch.unique(ci):
            yield int(c), ci == c

    def prior_apply(self, dxa, ci, norm):
        """Sa_inv @ dxa without materializing (B, n, n), via per-component
        matmuls. Returns (B, n)."""
        out = torch.empty_like(dxa)
        scale2 = self.comp_Sa_diagmean[ci] * norm**2
        for c, m in self._per_component(ci):
            out[m, : self.nb] = dxa[m, : self.nb] @ self.comp_Sa_inv[c]
        out[:, : self.nb] /= scale2[:, None]
        out[:, self.nb :] = (dxa[:, self.nb :] @ self.rt_Sa_inv_norm) / self.rt_scale**2
        return out

    # -- cost / GN system ------------------------------------------------------
    def cost(self, x, meas, Seps_inv, ci, g):
        """ISOFIT loss: dmeas^T Seps_inv dmeas + dxa^T Sa_inv dxa (windowed)."""
        rho = x[:, : self.nb]
        rdn, _ = self.calc_rdn(x[:, self.nb :], rho, g)
        dmeas = (rdn - meas)[:, self.winidx]
        norm = self._norm(x)
        dxa = x - self.prior_mean(x, ci, norm)
        c_meas = torch.einsum("bi,bij,bj->b", dmeas, Seps_inv, dmeas)
        c_prior = (dxa * self.prior_apply(dxa, ci, norm)).sum(1)
        return c_meas + c_prior

    def gn_system(self, x, meas, Seps_inv, ci, g, H_buf=None):
        """Assemble H = K^T Seps_inv K + Sa_inv and g = K^T Seps_inv dmeas
        + Sa_inv (x - xa) using the diag+columns structure of K."""
        B = x.shape[0]
        rdn, d_rfl, d_rt = self.forward_fd(x, g)
        dmeas = (rdn - meas)[:, self.winidx]  # (B, nw)
        dw = d_rfl[:, self.winidx]  # (B, nw)
        Aw = d_rt[:, self.winidx]  # (B, nw, n_rt)

        # M = Seps_inv @ K_active, K_active = [diag(dw) | Aw]
        SiD = Seps_inv * dw[:, None, :]  # cols scaled: (B, nw, nw)
        SiA = torch.einsum("bij,bjk->bik", Seps_inv, Aw)
        H11 = SiD * dw[:, :, None]
        H12 = SiA * dw[:, :, None]
        H22 = torch.einsum("bji,bjk->bik", Aw, SiA)

        norm = self._norm(x)
        dxa = x - self.prior_mean(x, ci, norm)
        scale2 = self.comp_Sa_diagmean[ci] * norm**2

        # H = Sa_inv (block diagonal) + scattered measurement block
        if H_buf is None or H_buf.shape[0] != B:
            H_buf = torch.zeros(
                (B, self.n_state, self.n_state), device=self.device, dtype=self.dtype
            )
        H = H_buf
        H.zero_()
        for c, m in self._per_component(ci):
            H[m, : self.nb, : self.nb] = self.comp_Sa_inv[c][None]
        H[:, : self.nb, : self.nb] /= scale2[:, None, None]
        H[:, self.nb :, self.nb :] = self.rt_Sa_inv_norm[None] / self.rt_scale**2

        act = self.act
        H[:, act[: self.nw, None], act[None, : self.nw]] += H11
        H[:, act[: self.nw, None], act[None, self.nw :]] += H12
        H[:, act[self.nw :, None], act[None, : self.nw]] += H12.transpose(1, 2)
        H[:, act[self.nw :, None], act[None, self.nw :]] += H22

        gvec = self.prior_apply(dxa, ci, norm)
        c_prior = (dxa * gvec).sum(1)
        Sid = torch.einsum("bij,bj->bi", Seps_inv, dmeas)  # (B, nw)
        gvec[:, self.winidx] += dw * Sid
        gvec[:, self.nb :] += torch.einsum("bji,bj->bi", Aw, Sid)

        cost = torch.einsum("bi,bi->b", dmeas, Sid) + c_prior
        return H, gvec, cost

    # -- solver -----------------------------------------------------------------
    def invert(self, meas_np, geom, x0_np, ci_np, Seps_win_np, max_iter=20,
               xtol=1e-8, ftol=1e-8, chunk=64, lam0=1e-5, callback=None):
        """Batched bounded Levenberg-Marquardt.

        meas_np: (B, nb); x0_np: (B, n_state); ci_np: (B,) surface component
        per pixel (as ISOFIT selects from the heuristic init); Seps_win_np:
        (B, nw, nw) windowed observation covariance. Returns (B, n_state).
        """
        out = np.empty_like(x0_np)
        for lo in range(0, len(meas_np), chunk):
            hi = min(lo + chunk, len(meas_np))
            out[lo:hi] = self._invert_chunk(
                meas_np[lo:hi], geom, x0_np[lo:hi], ci_np[lo:hi],
                Seps_win_np[lo:hi], max_iter, xtol, ftol, lam0, callback,
            )
        return out

    def _invert_chunk(self, meas_np, geom, x0_np, ci_np, Seps_win_np,
                      max_iter, xtol, ftol, lam0, callback):
        t = lambda a: torch.tensor(np.asarray(a), device=self.device, dtype=self.dtype)
        meas, x = t(meas_np), t(x0_np)
        ci = torch.tensor(np.asarray(ci_np), device=self.device, dtype=torch.long)
        B = x.shape[0]
        g = self._geom_consts(geom)

        # Seps_inv exactly as common.svd_inv (eigh-based), batched
        Seps = t(Seps_win_np)
        D, P = torch.linalg.eigh(Seps)
        D = torch.clamp(D, min=1e-12)
        Seps_inv = (P / D[:, None, :]) @ P.transpose(1, 2)

        x = torch.clamp(x, self.lb + self.eps, self.ub - self.eps)
        lam = torch.full((B,), lam0, device=self.device, dtype=self.dtype)
        active = torch.ones(B, dtype=torch.bool, device=self.device)

        H, gvec, cost = self.gn_system(x, meas, Seps_inv, ci, g)
        Ht = torch.empty_like(H)

        nu = torch.full((B,), 2.0, device=self.device, dtype=self.dtype)

        for it in range(max_iter):
            accepted = torch.zeros(B, dtype=torch.bool, device=self.device)
            # full-batch damping rounds with masked updates: cheaper than
            # subsetting (B, n, n) tensors with fancy indexing.
            # Damping follows the Nielsen/Madsen gain-ratio schedule, which
            # adapts lambda by predicted-vs-actual cost reduction (the same
            # principle as scipy TRF's trust radius update).
            for _retry in range(8):
                trial = active & ~accepted
                if not trial.any():
                    break
                Ht.copy_(H)
                diagH = torch.diagonal(Ht, dim1=1, dim2=2)
                damp = lam[:, None] * diagH
                diagH += damp
                L, info = torch.linalg.cholesky_ex(Ht)
                ok = (info == 0) & trial
                dx = -torch.cholesky_solve(gvec[..., None], L)[..., 0]
                lo, hi = self.lb + self.eps, self.ub - self.eps
                xt = torch.clamp(x + dx, lo, hi)

                # Active-set refinement: a raw step that crosses a bound gets
                # clamped, which silently corrupts the other coordinates
                # (scipy's TRF avoids this with reflective scaling). Move the
                # offending coordinates exactly to their bound (dx_z) and
                # re-solve the reduced damped system for the free ones:
                #   (H+lam*D)_ff dx_f = -(g + (H+lam*D) dx_z)_f
                frozen = (((x + dx) < lo) & (dx < 0)) | (((x + dx) > hi) & (dx > 0))
                if frozen.any():
                    dx_z = torch.zeros_like(dx)
                    dx_z = torch.where(frozen & (dx < 0), lo - x, dx_z)
                    dx_z = torch.where(frozen & (dx > 0), hi - x, dx_z)
                    g_eff = gvec + torch.einsum("bij,bj->bi", Ht, dx_z)
                    free = (~frozen).to(self.dtype)
                    Ht.mul_(free[:, :, None] * free[:, None, :])
                    torch.diagonal(Ht, dim1=1, dim2=2).add_(1.0 - free)
                    L2, info2 = torch.linalg.cholesky_ex(Ht)
                    ok = ok & (info2 == 0)
                    dx2 = -torch.cholesky_solve((g_eff * free)[..., None], L2)[..., 0]
                    xt2 = torch.clamp(x + dx2 * free + dx_z, lo, hi)
                    has_frozen = frozen.any(dim=1)
                    xt = torch.where(has_frozen[:, None], xt2, xt)

                dxc = xt - x  # actual (possibly clamped) step
                ct = self.cost(xt, meas, Seps_inv, ci, g)
                # predicted reduction of the damped quadratic model for
                # cost = r^T r:  -2 g.dx - dx.H.dx = dx.(lam*D)dx - g.dx
                pred = (dxc * damp * dxc).sum(1) - (dxc * gvec).sum(1)
                rho = (cost - ct) / torch.clamp(pred, min=1e-300)
                better = ok & (ct < cost) & (pred > 0)

                if better.any():
                    step = dxc.abs().max(dim=1).values
                    rel = (cost - ct) / torch.clamp(cost, min=1e-300)
                    x = torch.where(better[:, None], xt, x)
                    cost = torch.where(better, ct, cost)
                    accepted |= better
                    done = better & ((step < xtol) | (rel < ftol))
                    active &= ~done
                # Nielsen schedule
                fac = torch.clamp(1.0 - (2.0 * rho - 1.0) ** 3, min=1.0 / 3.0)
                lam = torch.where(better, torch.clamp(lam * fac, min=1e-12), lam)
                nu = torch.where(better, torch.full_like(nu, 2.0), nu)
                rejected = trial & ~better
                lam = torch.where(rejected, lam * nu, lam)
                nu = torch.where(rejected, nu * 2.0, nu)

            if callback:
                callback(it, int(active.sum()))
            # pixels that never accepted a step are converged/stuck
            active &= accepted
            if not active.any():
                break
            H, gvec, cost2 = self.gn_system(x, meas, Seps_inv, ci, g, H_buf=H)
            cost = torch.where(active, cost2, cost)

        return x.cpu().numpy()
