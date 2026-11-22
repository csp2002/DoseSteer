"""DoseDiff on GDP-HMM as a DRG spec (the model interface of drg/core.py), with the model builder, the plan loader and the scores.

The model works on a (96, 128, 160) crop centred on the plan's isocenter, slice by slice, with 47 condition channels
(preprocessing/gdp_hmm/gdp_data.py); x0 in [-1, 1], Gy = 40 (x + 1). Only the crop slices that contain body are sampled (at most 96, one batch),
and the prediction is pasted back to the original grid.
Structures: the plan's PTV levels (in the condition: OPT shell + clinical PTV and the PTV-count prompt; for the metrics: the
clinical PTV) and the OARs used in its optimisation. Targets and structure metrics are computed on the original grid inside
body ∩ crop coverage.
"""
import json
import os
import sys
import zlib

import numpy as np
import torch
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (os.path.join(REPO, 'models', 'dosediff'), os.path.join(REPO, 'models'), os.path.join(REPO, 'preprocessing', 'gdp_hmm')):
    if _p not in sys.path:                                            # DoseDiff's guided_diffusion; evaluate_gdp; gdp_data
        sys.path.insert(0, _p)
from gdp_data import (GDPPlan, assemble, crop_pad, paste_back, read_meta, IN_SIZE, IMG, DIS_CHANNELS,  # noqa: E402
                      DOSE_MAX_GY, HaN_OAR_DICT, Lung_OAR_DICT, B_OAR, B_OPT, B_PTV)
from evaluate_gdp import gdp_metric1                                  # noqa: E402

HALF = DOSE_MAX_GY / 2.0                     # x = Gy / 40 - 1
CH_OPT, CH_PTV, CH_OAR, CH_NPTV = 0, 3, 6, 42
PTV_LEVELS = ('PTV_High', 'PTV_Mid', 'PTV_Low')
SEED = 0
CHUNK = 96                                   # slices per forward pass


def build_diffusion(T=1000, ddim_n=8):
    from guided_diffusion import gaussian_diffusion as gd
    from guided_diffusion.respace import SpacedDiffusion, space_timesteps
    return SpacedDiffusion(use_timesteps=space_timesteps(T, 'ddim%d' % ddim_n), betas=gd.get_named_beta_schedule('linear', T),
                           model_mean_type=gd.ModelMeanType.EPSILON, model_var_type=gd.ModelVarType.FIXED_LARGE,
                           loss_type=gd.LossType.MSE, rescale_timesteps=False)


def build_model(ckpt, device='cuda', model_channels=128):
    """DoseDiff (UNetModel_MS_Former) for GDP-HMM, frozen."""
    from guided_diffusion.unet import UNetModel_MS_Former
    net = UNetModel_MS_Former(image_size=IMG, in_channels=1, ct_channels=1, dis_channels=DIS_CHANNELS, model_channels=model_channels,
                              out_channels=1, num_res_blocks=2, attention_resolutions=(16, 32), dropout=0, channel_mult=(1, 1, 2, 3, 4),
                              conv_resample=True, dims=2, num_classes=None, use_checkpoint=False, use_fp16=False, num_heads=4,
                              num_head_channels=-1, num_heads_upsample=-1, use_scale_shift_norm=True, resblock_updown=False,
                              use_new_attention_order=False).to(device).eval()
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)
    net.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()})
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def load_dicts(gdp_root):
    return (json.load(open(os.path.join(gdp_root, 'meta_files', 'PTV_DICT.json'))),
            json.load(open(os.path.join(gdp_root, 'meta_files', 'Pat_Obj_DICT.json'))))


def plan_rows(gdp_root, split='test'):
    return read_meta(os.path.join(gdp_root, 'meta_files', 'meta_data.csv'), split)


# --------------------------------------------------------------------------- one plan
def struct_metric(dose, s):
    """OAR: mean dose; PTV: D95 (5th percentile); inside body ∩ crop coverage."""
    roi = dose[s['eval']]
    if roi.size == 0:
        return float('nan')
    return float(roi.mean()) if s['kind'] == 'OAR' else float(np.percentile(roi, 5))


class GDPCase:
    """One plan: the network inputs on the crop, the structures (name, kind, condition channels, masks) and the targets."""

    def __init__(self, row, ptv_dict, obj_dict, gdp_root):
        self.plan = p = GDPPlan(row['npz_path'], row, ptv_dict, obj_dict, data_root=gdp_root)
        self.id, self.site, self.pid = p.id, p.site, p.pid
        self.spacing = p.spacing_zyx
        self.ct, self.dis, _ = assemble(p.packed())                    # (Z, 1, H, W), (Z, 47, H, W) on the crop
        self.valid_z = p.valid_z
        self.gt = p.dose_gy                                            # reference dose, original grid, Gy
        self.body = p.body_full > 0
        self.coverage = paste_back(np.ones(IN_SIZE, np.float32), p.isocenter, IN_SIZE, p.ori_shape) > 0
        self.roi = self.body & self.coverage
        d = np.load(os.path.join(gdp_root, row['npz_path']), allow_pickle=True)['arr_0'].item()
        pinfo = ptv_dict[p.pid]
        bits = p.bits
        self.structures = []
        lvls = [(lvl, float(pinfo[lvl]['PDose'])) for lvl in PTV_LEVELS if lvl in pinfo]
        lvls.sort(key=lambda t: -t[1])                                  # channel order: prescription descending
        for k, (lvl, pdose) in enumerate(lvls):
            sname = pinfo[lvl]['OPTName'] if lvl == 'PTV_High' else pinfo[lvl].get('StructName')
            sname = sname if sname in d else pinfo[lvl]['OPTName']
            m_opt = ((bits >> np.uint64(B_OPT + k)) & np.uint64(1)).astype(bool)
            m_ptv = ((bits >> np.uint64(B_PTV + k)) & np.uint64(1)).astype(bool)
            # dropping a PTV level removes both its channels (OPT shell, clinical PTV) and lowers the PTV-count prompt;
            # the mask of the context response is their union, the metric (D95) uses the clinical PTV
            self.structures.append(dict(name='%s(%s)' % (lvl, sname), kind='PTV', chans=[CH_OPT + k, CH_PTV + k], nptv=True,
                                        mask_crop=(m_opt | m_ptv), mask_ori=(d[sname] > 0)))
        oar_dict = HaN_OAR_DICT if p.site < 1.5 else Lung_OAR_DICT
        need = obj_dict.get(p.pid, list(oar_dict))
        for name, idx in oar_dict.items():
            if name not in need or name not in d:
                continue
            self.structures.append(dict(name=name, kind='OAR', chans=[CH_OAR + idx - 1], nptv=False,
                                        mask_crop=((bits >> np.uint64(B_OAR + idx - 1)) & np.uint64(1)).astype(bool),
                                        mask_ori=(d[name] > 0)))
        # keep the structures the model can act on (voxels in the crop and in body ∩ coverage)
        self.structures = [s for s in self.structures if s['mask_crop'].any() and (s['mask_ori'] & self.roi).any()]
        for s in self.structures:
            s['eval'] = s['mask_ori'] & self.roi
            s['target'] = struct_metric(self.gt, s)
        self.struct = {s['name']: s for s in self.structures}
        self.union = np.zeros_like(self.roi)
        for s in self.structures:
            self.union |= s['eval']
        del d

    def to_gy(self, pred_norm):
        return self.plan.to_original_grid(pred_norm)


class GDPStructs:
    """The structures of a GDPCase on the model grid (the crop slices that contain body): the `P` of drg/core.py."""

    def __init__(self, case, device='cuda'):
        iso = case.plan.isocenter
        self.case, self.dev = case, device
        self.vz = np.asarray(case.valid_z)
        self.spacing_zyx = tuple(case.spacing)
        self.names = [s['name'] for s in case.structures]
        self.kind = {s['name']: s['kind'] for s in case.structures}
        self.target = {s['name']: float(s['target']) for s in case.structures}
        self.E = []
        for s in case.structures:
            full = crop_pad(s['eval'].astype(np.float32), iso, IN_SIZE) > 0
            e = full[self.vz]
            assert full.sum() == s['eval'].sum() and e.sum() == full.sum(), s['name']   # all metric voxels are on sampled slices
            self.E.append(torch.from_numpy(e).to(device))
        self.K = len(self.E)
        self.layers = [torch.nonzero(e.flatten(1).any(1)).flatten() for e in self.E]
        self._feather = {}

    def feather_t(self, k, d_mm):
        """M_k = clip(1 - dist / d, 0, 1) on the crop, dist = distance (mm) to structure k (its voxels in body ∩ crop)."""
        key = (k, float(d_mm))
        if key not in self._feather:
            m = crop_pad(self.case.structures[k]['eval'].astype(np.float32), self.case.plan.isocenter, IN_SIZE) > 0
            dist = ndimage.distance_transform_edt(~m, sampling=self.spacing_zyx).astype(np.float32)
            self._feather[key] = torch.from_numpy(np.clip(1.0 - dist / float(d_mm), 0.0, 1.0)[self.vz]).to(self.dev)
        return self._feather[key]


# --------------------------------------------------------------------------- scores (original grid, Gy)
def dvh_metrics(dose, case):
    """PTV D1 / D95 / D99, OAR D_0.1cc / mean (OpenKBP's DVH points; D_0.1cc from each plan's own voxel size)."""
    n_tenth_cc = max(1, int(round(100.0 / float(np.prod(case.spacing)))))
    out = {}
    for s in case.structures:
        roi = dose[s['eval']]
        if roi.size == 0:
            continue
        if s['kind'] == 'PTV':
            out[(s['name'], 'D_99')] = float(np.percentile(roi, 1))
            out[(s['name'], 'D_95')] = float(np.percentile(roi, 5))
            out[(s['name'], 'D_1')] = float(np.percentile(roi, 99))
        else:
            out[(s['name'], 'D_0.1_cc')] = float(np.percentile(roi, max(0.0, 100.0 - n_tenth_cc / len(roi) * 100.0)))
            out[(s['name'], 'mean')] = float(roi.mean())
    return out


def evaluate(dose, case, gt_dvh=None):
    """official = GDP-HMM challenge score (sum |error| over the 5 Gy region of the reference or prediction, divided by the
    reference's 5 Gy volume, in body); dvh = mean |DVH point error|; target = mean |structure metric - target|;
    within = voxel MAE in the union of the structures; outside = voxel MAE outside it, in body, in the 5 Gy region."""
    gt_dvh = dvh_metrics(case.gt, case) if gt_dvh is None else gt_dvh
    pd_ = dvh_metrics(dose, case)
    keys = [k for k in gt_dvh if k in pd_]
    err = np.abs(dose - case.gt)
    out5 = ((case.gt > 5) | (dose > 5)) & case.body & ~case.union
    return {'official': gdp_metric1(dose, case.gt, case.body),
            'dvh': float(np.mean([abs(pd_[k] - gt_dvh[k]) for k in keys])) if keys else float('nan'),
            'target': float(np.mean([abs(struct_metric(dose, s) - s['target']) for s in case.structures])),
            'within': float(err[case.union].mean()) if case.union.any() else float('nan'),
            'outside': float(err[out5].mean()) if out5.any() else float('nan')}


# --------------------------------------------------------------------------- sampler interface
class GDPSpec:
    """DoseDiff on one GDP-HMM plan. x0_drop uses the condition without structure `name` (its channels set to zero; for a PTV level
    the PTV-count prompt is lowered by one, as for a plan with one level fewer)."""
    lo, hi, gy_scale = -1.0, 1.0, HALF

    def __init__(self, net, diffusion, case, device='cuda', seed=SEED, chunk=CHUNK):
        self.net, self.d, self.case, self.dev, self.seed, self.chunk = net, diffusion, case, device, seed, chunk
        self.tmap = torch.tensor(diffusion.timestep_map, device=device, dtype=torch.long)
        self.vz = np.asarray(case.valid_z)
        self.B = len(self.vz)
        assert self.B <= 96
        self.ct = torch.from_numpy(case.ct[self.vz]).to(device)
        self.dis = torch.from_numpy(case.dis[self.vz]).to(device)
        self.n_steps = diffusion.num_timesteps

    def x_T(self):
        g = torch.Generator(device='cpu').manual_seed(self.seed + zlib.crc32(self.case.id.encode()) % 100000)
        return torch.randn(self.B, 1, IMG[0], IMG[1], generator=g).to(self.dev)

    def _t(self, n, i):
        return torch.full((n,), i, device=self.dev, dtype=torch.long)

    def _eps(self, x, i, ct, dis):
        t = self.tmap[self._t(x.shape[0], i)]
        out = []
        for a in range(0, x.shape[0], self.chunk):
            out.append(self.net(x[a:a + self.chunk], t[a:a + self.chunk], ct[a:a + self.chunk], dis[a:a + self.chunk]))
        return torch.cat(out, 0)

    def eps_all(self, x, i):
        return self._eps(x, i, self.ct, self.dis)

    def eps_slices(self, x, i, zs):
        return self._eps(x, i, self.ct[zs], self.dis[zs])

    def x0_drop(self, x, i, name):
        s = self.case.struct[name]
        d2 = self.dis.clone()
        d2[:, s['chans']] = 0.0
        if s['nptv']:
            d2[:, CH_NPTV] = torch.clamp(d2[:, CH_NPTV] - 1.0, min=0.0)
        return self.x0_from_eps(x, self._eps(x, i, self.ct, d2), i)

    def x0_from_eps(self, x, eps, i):
        return self.d._predict_xstart_from_eps(x, self._t(x.shape[0], i), eps)

    def next_x(self, x, x0, i):
        from guided_diffusion.gaussian_diffusion import _extract_into_tensor
        t = self._t(x.shape[0], i)
        e = self.d._predict_eps_from_xstart(x, t, x0)
        ab_prev = _extract_into_tensor(self.d.alphas_cumprod_prev, t, x.shape)
        return x0 * torch.sqrt(ab_prev) + torch.sqrt(1 - ab_prev) * e

    def to_gy(self, x0):
        return (x0 + 1.0) * HALF

    def final_dose(self, x):
        pred = np.full((self.case.dis.shape[0],) + tuple(IMG), -1.0, np.float32)
        pred[self.vz] = x[:, 0].cpu().numpy()
        return self.case.to_gy(pred)
