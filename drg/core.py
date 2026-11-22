"""Dual-Response Guidance (DRG): steering a frozen diffusion dose predictor towards structure-level clinical targets.

At each steering step (Algorithm 1), DRG corrects the clean estimate x0 so that every structure's metric q_k (OAR mean dose,
PTV D95) moves to its target y_k. For every structure k it measures two responses of the predictor f,
    N_k  noise response:   (f(x_t + h 1_k, c) - f(x_t - h 1_k, c)) / 2h, 1_k = the structure (OAR) or its coldest 10 % (PTV),
    C_k  context response: M_k * (f(x_t, c) - f(x_t, c without structure k)), M_k a linear feather (1 on k, 0 beyond d),
and adds Delta = sum_k alpha_k N_k + beta_k C_k, with the strengths from the priced joint solve (drg_solve).

This file is model-agnostic. A `spec` object wraps one predictor and its sampler (see specs_openkbp.py):
    spec.n_steps                       number of DDIM steps; i = n_steps - 1, ..., 0 (i = 0 is the last step)
    spec.x_T()                         initial noise (Z, 1, H, W)
    spec.eps_all(x, i)                 predicted noise for all slices
    spec.eps_slices(x, i, zs)          predicted noise for the slices zs (x holds only those slices)
    spec.x0_from_eps(x, eps, i)        unclipped x0
    spec.x0_drop(x, i, name)           unclipped x0 with structure `name` removed from the condition
    spec.next_x(x, x0, i)              one deterministic DDIM step from a (clipped, possibly corrected) x0
    spec.lo, spec.hi                   clipping range of x0
    spec.to_gy(x0), spec.gy_scale      model units -> Gy (linear)
    spec.final_dose(x)                 dose in Gy after the last step (clipped, times the possible-dose mask)
and either spec.d (a guided-diffusion SpacedDiffusion) or spec.alphas_bar / spec._ts (MD-Dose) for the noise level.
"""
import math

import numpy as np
import torch
from scipy import ndimage

OARS = ['Brainstem', 'Esophagus', 'Larynx', 'LeftParotid', 'Mandible', 'RightParotid', 'SpinalCord']
PTVS = ['PTV70', 'PTV63', 'PTV56']

# DRG hyperparameters. Shared: B = 3, h = 0.02 (model units), the 10 % PTV cold set, the factor 1e-3 in mu, the feather
# distance d = 20 mm and the steering steps S = {3, 2, 1, 0} of the 8 DDIM steps. Per model, chosen on validation cases:
# lam, the weight of the shape term (0: none), and sigma_beta = 0.03 * sqrt(gamma).
DRG_STEPS = (3, 2, 1, 0)
DRG_OPENKBP = {
    'dosediff': dict(lam=0.1, sigma_beta=0.03 * math.sqrt(3)),       # gamma = 3
    'mddose': dict(lam=0.03, sigma_beta=0.03 * math.sqrt(0.1)),      # gamma = 0.1
    'dosedit': dict(lam=0.1, sigma_beta=0.03 * math.sqrt(0.003)),    # gamma = 0.003
}
DRG_GDP = dict(lam=0.0, sigma_beta=0.03 * math.sqrt(0.1))            # DoseDiff on GDP-HMM: gamma = 0.1


# --------------------------------------------------------------------------- metrics (numpy, original grid, Gy)
def struct_metric(dose, mask, kind):
    """OAR: mean dose in the mask; PTV: D95 (5th percentile)."""
    roi = dose[mask]
    if roi.size == 0:
        return float('nan')
    return float(roi.mean()) if kind == 'OAR' else float(np.percentile(roi, 5))


def dvh_metrics(dose, P):
    """OpenKBP DVH points: PTV D1 / D95 / D99, OAR D_0.1cc / mean."""
    out = {}
    vox = float(np.prod(P.spacing_zyx))
    for s in P.names:
        roi = dose[P.mask[s]]
        if roi.size == 0:
            continue
        if s in PTVS:
            out[(s, 'D1')] = float(np.percentile(roi, 99))
            out[(s, 'D95')] = float(np.percentile(roi, 5))
            out[(s, 'D99')] = float(np.percentile(roi, 1))
        else:
            frac = 100 - np.maximum(1, np.round(100 / vox)) / len(roi) * 100
            out[(s, 'D_0.1_cc')] = float(np.percentile(roi, frac))
            out[(s, 'mean')] = float(roi.mean())
    return out


def evaluate(dose, P, gt_dvh=None):
    """official = OpenKBP dose score (voxel MAE in the possible-dose mask); dvh = mean |DVH point error|;
    target = mean |structure metric - target|; within / outside = voxel MAE inside / outside the union of the structures."""
    gt_dvh = dvh_metrics(P.gt, P) if gt_dvh is None else gt_dvh
    body, union = P.body, P.union
    pd_ = dvh_metrics(dose, P)
    keys = [k for k in gt_dvh if k in pd_]
    return {'official': float(np.abs(dose - P.gt)[body].mean()),
            'dvh': float(np.mean([abs(pd_[k] - gt_dvh[k]) for k in keys])) if keys else float('nan'),
            'target': float(np.mean([abs(struct_metric(dose, P.mask[s], P.kind[s]) - P.target[s]) for s in P.names])),
            'within': float(np.abs(dose - P.gt)[union & body].mean()),
            'outside': float(np.abs(dose - P.gt)[body & ~union].mean())}


class PatientStructs:
    """The structures of one case on the model grid (Z, H, W). masks: {name: bool array}; gt: reference dose (Gy), for the
    targets (OAR mean, PTV D95) and the scores; body: possible-dose mask; spacing_zyx in mm. Empty structures are dropped."""

    def __init__(self, pid, masks, gt, body, spacing_zyx, device='cuda'):
        self.pid, self.gt, self.body, self.spacing_zyx, self.dev = pid, gt, body, tuple(spacing_zyx), device
        self.names = [s for s in OARS + PTVS if s in masks and masks[s].any()]
        self.mask = {s: masks[s] for s in self.names}
        self.kind = {s: ('OAR' if s in OARS else 'PTV') for s in self.names}
        self.target = {s: struct_metric(gt, self.mask[s], self.kind[s]) for s in self.names}
        self.union = np.zeros_like(body)
        for s in self.names:
            self.union |= self.mask[s]
        self.K = len(self.names)
        self.E = [torch.from_numpy(self.mask[s]).to(device) for s in self.names]
        self.layers = [torch.nonzero(e.flatten(1).any(1)).flatten() for e in self.E]
        self._feather = {}

    def feather_t(self, k, d_mm):
        """Feathered mask M_k = clip(1 - dist / d, 0, 1), dist = distance (mm) to structure k inside the body."""
        key = (k, float(d_mm))
        if key not in self._feather:
            m = self.mask[self.names[k]] & self.body
            dist = ndimage.distance_transform_edt(~m, sampling=self.spacing_zyx)
            self._feather[key] = torch.from_numpy(np.clip(1.0 - dist / float(d_mm), 0.0, 1.0).astype(np.float32)).to(self.dev)
        return self._feather[key]


# --------------------------------------------------------------------------- solve
def _abar(spec, i):
    """alpha_bar of step i (i = 0 is the last step)."""
    if hasattr(spec, 'alphas_bar'):                                # MD-Dose
        t, _ = spec._ts(i)
        return float(spec.alphas_bar[t])
    return float(spec.d.alphas_cumprod[i])


def _shape_gram(cols, masks, gy):
    """M with theta^T M theta = V(theta) = sum_k Var_{Omega_k}(sum_j theta_j col_j) (Gy^2): the shape term."""
    n = len(cols)
    M = torch.zeros(n, n, device=cols[0].device, dtype=torch.float64)
    for m in masks:
        if not bool(m.any()):
            continue
        X = torch.stack([c[m].double() for c in cols])
        X = X - X.mean(1, keepdim=True)
        M = M + (X @ X.T) / float(X.shape[1])
        del X
    return M * float(gy) ** 2


def drg_solve(GN, GC, r, v, sigma_beta, lam=0.0, M=None, B=3.0, mu_factor=1e-3):
    """The solve of DRG: the minimizer over |beta_k| <= B of
        Phi(theta) = ||G theta - r||^2 + lam V(theta) + mu P(theta),   theta = (alpha, beta),  G = [GN GC],
        P(theta)   = sum_k ||GN[:, k] alpha_k||^2 / v_k + beta_k^2 / sigma_beta^2,
        mu         = mu_factor / (2K) (sum_k v_k + sigma_beta^2 sum_k ||GC[:, k]||^2),
    with V(theta) = theta^T M theta (lam = 0: no shape term). GN, GC: G^N, G^C, the metric changes per unit strength (K x K,
    Gy); r: residuals (Gy); v: v_k (Gy^2). P is a Gaussian prior with variances v_k / ||GN[:, k]||^2 for alpha_k and
    sigma_beta^2 for beta_k; v_k <= 0 is set to 0, and the variances of the alpha_k are then increased by 1e-6 times their
    mean (at least by 1e-36). Solved by bounded-variable least squares in the prior-whitened strengths. Returns theta."""
    from scipy.optimize import lsq_linear
    GN, GC, r = np.asarray(GN, np.float64), np.asarray(GC, np.float64), np.asarray(r, np.float64)
    K = len(r)
    v = np.maximum(np.asarray(v, np.float64), 0.0)
    cn = (GN ** 2).sum(0)
    var_a = np.divide(v, cn, out=np.zeros(K), where=cn > 0)
    var_a = var_a + max(1e-6 * var_a.mean(), 1e-36)
    s = np.sqrt(np.concatenate([var_a, np.full(K, float(sigma_beta) ** 2)]))   # prior standard deviations
    mu = mu_factor / (2 * K) * (v.sum() + float(sigma_beta) ** 2 * (GC ** 2).sum())
    A = [np.concatenate([GN, GC], 1) * s[None, :], np.sqrt(mu) * np.eye(2 * K)]
    if lam > 0:
        Ms = np.asarray(M, np.float64) * s[:, None] * s[None, :]
        Ms = 0.5 * (Ms + Ms.T) + 1e-9 * (np.trace(Ms) / (2 * K) + 1e-30) * np.eye(2 * K)
        A.append(np.sqrt(lam) * np.linalg.cholesky(Ms).T)
    A = np.vstack(A)
    b = np.concatenate([r, np.zeros(A.shape[0] - K)])
    ub = np.concatenate([np.full(K, np.inf), float(B) / s[K:]])
    return s * lsq_linear(A, b, bounds=(-ub, ub), method='bvls').x


# --------------------------------------------------------------------------- one steering step
def drg_step(spec, P, x, i, x0_raw, x0, lam, sigma_beta, h=0.02, B=3.0, cold=0.1, d_mm=20.0, mu_factor=1e-3, chunk=128):
    """The corrected, clipped x0 at steering step i (Algorithm 1). x: x_t; x0_raw: the estimate f(x_t, c); x0: x0_raw clipped."""
    gy = spec.to_gy(x0[:, 0])
    ones, r = [], []
    for k, s in enumerate(P.names):                                # 1_k and the residual r_k = y_k - q_k(x0)
        e = P.E[k]
        if P.kind[s] == 'OAR':
            ones.append(e)
            r.append(float(P.target[s] - gy[e].mean()))
        else:                                                      # PTV: its coldest 10 % at the current x0; q_k = D95
            dose_k = gy[e]
            ones.append(e & (gy <= torch.quantile(dose_k, cold)))
            r.append(float(P.target[s] - torch.quantile(dose_k, 0.05)))

    # noise response N_k = (f(x_t + h 1_k, c) - f(x_t - h 1_k, c)) / 2h. The predictors work slice by slice, so the slices of
    # structure k are pushed and evaluated one by one, and the other slices of N_k are zero.
    N = [torch.zeros_like(x0[:, 0]) for _ in range(P.K)]
    jobs = [(k, z) for k in range(P.K) for z in P.layers[k].tolist()]
    for a in range(0, len(jobs), chunk):
        part = jobs[a:a + chunk]
        zs = torch.tensor([z for _, z in part], device=x.device)
        push = torch.stack([h * ones[k][z].float() for k, z in part])[:, None]
        xp, xm = x[zs] + push, x[zs] - push
        fp = spec.x0_from_eps(xp, spec.eps_slices(xp, i, zs), i)[:, 0]
        fm = spec.x0_from_eps(xm, spec.eps_slices(xm, i, zs), i)[:, 0]
        for n_, (k, z) in enumerate(part):
            N[k][z] += (fp[n_] - fm[n_]) / (2.0 * h)

    # context response C_k = M_k * (f(x_t, c) - f(x_t, c without structure k))
    C = [(x0_raw - spec.x0_drop(x, i, s))[:, 0] * P.feather_t(k, d_mm) for k, s in enumerate(P.names)]

    def metric_change(cols):                                       # G_jk = m_j(col_k) in Gy, m_j the mean over 1_j
        return np.array([[float(col[o].sum()) / float(o.sum()) * spec.gy_scale for col in cols] for o in ones])

    GN, GC = metric_change(N), metric_change(C)
    ab = _abar(spec, i)
    kappa_t = (1.0 - ab) / math.sqrt(ab)
    v = kappa_t * spec.gy_scale * np.diag(GN) / np.array([float(o.sum()) for o in ones])   # v_k = kappa_t G^N_kk / |1_k|
    M = _shape_gram(N + C, P.E, spec.gy_scale).cpu().numpy() if lam > 0 else None
    theta = drg_solve(GN, GC, r, v, sigma_beta, lam, M, B, mu_factor)
    delta = sum(float(t) * col for t, col in zip(theta, N + C))
    return torch.clamp(x0[:, 0] + delta, spec.lo, spec.hi)[:, None]


# --------------------------------------------------------------------------- sampler
def sample(spec, P, steps=(), drg_kw=None):
    """Deterministic DDIM from spec.x_T(); DRG corrects x0 at the steps in `steps`. steps = () gives the unsteered prediction.
    Returns the dose in Gy on the original grid."""
    drg_kw = drg_kw or {}
    x = spec.x_T()
    with torch.no_grad():
        for i in range(spec.n_steps - 1, -1, -1):
            eps = spec.eps_all(x, i)
            x0_raw = spec.x0_from_eps(x, eps, i)
            x0 = x0_raw.clamp(spec.lo, spec.hi)
            if i in steps:
                x0 = drg_step(spec, P, x, i, x0_raw, x0, **drg_kw)
            x = spec.next_x(x, x0, i)
        return spec.final_dose(x)
