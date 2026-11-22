"""The three OpenKBP predictors as DRG specs (the model interface of drg/core.py), with their model builders and case loaders.

  dosediff  DoseDiff (UNetModel_MS_Former), guided-diffusion SpacedDiffusion with 8 DDIM steps, x0 in [-1, 1], Gy = 40 (x + 1),
            forward in batches of 32 slices; one shared initial noise (seed 600000) for every case.
  dosedit   DoseDiT: the same diffusion, inputs and sampler as DoseDiff, with the diffusion-transformer denoiser.
  mddose    MD-Dose (MambaUnet), its own linear-beta DDIM, x0 in [0, 1], Gy = 80 x, all 128 slices in one batch;
            initial noise seeded by crc32(case id).
All three read the OpenKBP test cases as preprocessed by preprocessing/openkbp (one folder per case with ct.nii.gz, dose.nii.gz,
Mask_<structure>.nii.gz and, for DoseDiff / DoseDiT, PSDM_<structure>.nii.gz).
"""
import os
import sys
import zlib

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
MODELS = os.path.join(REPO, 'models')


def _add_path(p):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)


# --------------------------------------------------------------------------- DoseDiff / DoseDiT (shared inputs and sampler)
NEEDED_MASKS = ['Brainstem', 'Esophagus', 'Larynx', 'LeftParotid', 'Mandible', 'possible_dose_mask', 'RightParotid', 'SpinalCord',
                'PTV56', 'PTV63', 'PTV70']
SEED = 600000
BS_Z = 32


def build_diffusion(T=1000, ddim_n=8):
    _add_path(os.path.join(MODELS, 'dosediff'))
    from guided_diffusion import gaussian_diffusion as gd
    from guided_diffusion.respace import SpacedDiffusion, space_timesteps
    return SpacedDiffusion(use_timesteps=space_timesteps(T, 'ddim%d' % ddim_n), betas=gd.get_named_beta_schedule('linear', T),
                           model_mean_type=gd.ModelMeanType.EPSILON, model_var_type=gd.ModelVarType.FIXED_LARGE,
                           loss_type=gd.LossType.MSE, rescale_timesteps=False)


def build_dosediff(ckpt, device='cuda'):
    """DoseDiff with frozen weights. The UNet's internal gradient checkpointing is turned into a plain call (inference only)."""
    _add_path(os.path.join(MODELS, 'dosediff'))
    import guided_diffusion.unet as gdunet
    gdunet.checkpoint = lambda func, inputs, params, flag: func(*inputs)
    net = gdunet.UNetModel_MS_Former(
        image_size=(128, 128), in_channels=1, ct_channels=1, dis_channels=20, model_channels=128, out_channels=1, num_res_blocks=2,
        attention_resolutions=(16, 32), dropout=0, channel_mult=(1, 1, 2, 3, 4), conv_resample=True, dims=2, num_classes=None,
        use_checkpoint=False, use_fp16=False, num_heads=4, num_head_channels=-1, num_heads_upsample=-1, use_scale_shift_norm=True,
        resblock_updown=False, use_new_attention_order=False)
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)
    net.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()})
    net.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


DOSEDIT_KW = dict(image_size=(128, 128), in_channels=1, ct_channels=1, dis_channels=20, grid=32, dim=384, heads=6, dose_depth=4,
                  dit_depth=12, taps=(2, 5, 8, 11), sdm_depth=12, ms_fusion=True, uvit_skip=True, out_conv=True)


def build_dosedit(ckpt, device='cuda'):
    """DoseDiT with the EMA weights of ckpt (an ema_*.pth state dict, or a training checkpoint holding 'ema'), frozen."""
    _add_path(os.path.join(MODELS, 'dosediff'))                       # guided_diffusion.vit
    _add_path(os.path.join(REPO, 'third_party', 'dinov3'))          # the DINOv3 ViT implementation used by the encoders
    _add_path(os.path.join(MODELS, 'dosedit'))
    import dosedit_model
    net = dosedit_model.DoseDiT(**DOSEDIT_KW)
    obj = torch.load(ckpt, map_location='cpu', weights_only=False)
    state = obj['ema'] if isinstance(obj, dict) and 'ema' in obj else obj
    net.load_state_dict({k.replace('module.', ''): v for k, v in state.items()}, strict=True)
    net.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def _read(path):
    """(array (z, y, x), spacing (z, y, x) as float32 values), as DoseDiff's Nii_utils.NiiDataRead."""
    import SimpleITK as sitk
    img = sitk.ReadImage(path)
    sp = img.GetSpacing()
    return sitk.GetArrayFromImage(img), np.array([sp[2], sp[1], sp[0]]).astype(np.float32)


def load_case(case_dir):
    """One preprocessed OpenKBP case (DoseDiff / DoseDiT inputs): CT, masks, PSDMs, reference dose, spacing (z, y, x) in mm."""
    ct, spacing = _read(os.path.join(case_dir, 'ct.nii.gz'))
    masks = {n: _read(os.path.join(case_dir, 'Mask_%s.nii.gz' % n))[0].astype(np.float32) for n in NEEDED_MASKS}
    psdms = {n: _read(os.path.join(case_dir, 'PSDM_%s.nii.gz' % n))[0].astype(np.float32) for n in NEEDED_MASKS}
    gt = _read(os.path.join(case_dir, 'dose.nii.gz'))[0].astype(np.float32)
    return {'ct': ct.astype(np.float32), 'masks': masks, 'psdms': psdms, 'gt_dose': gt, 'spacing_zyx': tuple(float(v) for v in spacing)}


def dis_volume(masks, psdms):
    """The 20 condition channels of DoseDiff: fused PTV (70/70, 63/70, 56/70), 8 masks, 11 signed distance maps."""
    ptvs = (70.0 / 70.0) * masks['PTV70'] + (63.0 / 70.0) * masks['PTV63'] + (56.0 / 70.0) * masks['PTV56']
    mask_order = ['Brainstem', 'Esophagus', 'Larynx', 'LeftParotid', 'Mandible', 'possible_dose_mask', 'RightParotid', 'SpinalCord']
    psdm_order = ['Brainstem', 'Esophagus', 'Larynx', 'LeftParotid', 'Mandible', 'possible_dose_mask', 'PTV56', 'PTV63', 'PTV70',
                  'RightParotid', 'SpinalCord']
    chans = [ptvs[np.newaxis]] + [masks[n][np.newaxis] for n in mask_order] + [psdms[n][np.newaxis] for n in psdm_order]
    return np.concatenate(chans, axis=0).astype(np.float32)


def dis_tensor(masks, psdms, device):
    return torch.from_numpy(dis_volume(masks, psdms)).float().permute(1, 0, 2, 3).to(device)


def _extract(arr, t, shape):
    _add_path(os.path.join(MODELS, 'dosediff'))
    from guided_diffusion.gaussian_diffusion import _extract_into_tensor
    return _extract_into_tensor(arr, t, shape)


class DoseDiffSpec:
    """DoseDiff / DoseDiT on one case. x0_drop uses the condition without structure `name` (its mask and PSDM set to zero)."""
    lo, hi, gy_scale = -1.0, 1.0, 40.0

    def __init__(self, net, diffusion, case, device='cuda', seed=SEED, bs_z=BS_Z):
        self.d, self.seed, self.bs_z, self.dev = diffusion, seed, bs_z, device
        self.masks, self.psdms = case['masks'], case['psdms']
        self.ct = torch.from_numpy(np.clip(case['ct'], 0, 2500) / 1250.0 - 1.0).unsqueeze(1).float().to(device)
        self.dis = dis_tensor(self.masks, self.psdms, device)
        self.pdm = torch.from_numpy(self.masks['possible_dose_mask']).float().to(device)
        self.wrapped = diffusion._wrap_model(net)
        self.n_steps = diffusion.num_timesteps
        self.Z, _, self.H, self.W = self.ct.shape
        self._drop = {}

    def x_T(self):
        g = torch.Generator(device=self.dev).manual_seed(int(self.seed))
        return torch.randn((self.Z, 1, self.H, self.W), generator=g, device=self.dev, dtype=torch.float32)

    def _t(self, n, i):
        return torch.full((n,), i, device=self.dev, dtype=torch.long)

    def _eps(self, x, i, dis):
        out = []
        for zs in range(0, self.Z, self.bs_z):
            ze = zs + self.bs_z
            out.append(self.wrapped(x[zs:ze], self._t(ze - zs, i), ct=self.ct[zs:ze], dis=dis[zs:ze]))
        return torch.cat(out, 0)

    def eps_all(self, x, i):
        return self._eps(x, i, self.dis)

    def eps_slices(self, x, i, zs):
        out = []
        for a in range(0, len(zs), self.bs_z):
            z = zs[a:a + self.bs_z]
            out.append(self.wrapped(x[a:a + self.bs_z], self._t(len(z), i), ct=self.ct[z], dis=self.dis[z]))
        return torch.cat(out, 0)

    def x0_drop(self, x, i, name):
        if name not in self._drop:
            m2, p2 = dict(self.masks), dict(self.psdms)
            m2[name] = np.zeros_like(self.masks[name])
            p2[name] = np.zeros_like(self.psdms[name])
            self._drop[name] = dis_tensor(m2, p2, self.dev)
        return self.x0_from_eps(x, self._eps(x, i, self._drop[name]), i)

    def x0_from_eps(self, x, eps, i):
        t = self._t(x.shape[0], i)
        return _extract(self.d.sqrt_recip_alphas_cumprod, t, x.shape) * x - _extract(self.d.sqrt_recipm1_alphas_cumprod, t, x.shape) * eps

    def next_x(self, x, x0, i):
        t = self._t(x.shape[0], i)
        srp = _extract(self.d.sqrt_recip_alphas_cumprod, t, x.shape)
        srm1 = _extract(self.d.sqrt_recipm1_alphas_cumprod, t, x.shape)
        eps_c = (srp * x - x0) / srm1
        ab_prev = _extract(self.d.alphas_cumprod_prev, t, x.shape)
        return x0 * torch.sqrt(ab_prev) + torch.sqrt(1 - ab_prev) * eps_c

    def to_gy(self, x0):
        return (x0 + 1.0) * 40.0

    def final_dose(self, x):
        return (((x[:, 0] + 1.0) * 40.0).clamp(0.0, 80.0) * self.pdm).cpu().numpy()


# --------------------------------------------------------------------------- MD-Dose
OARS_MD = ['Brainstem', 'Esophagus', 'Larynx', 'LeftParotid', 'Mandible', 'RightParotid', 'SpinalCord']
PTVS_MD = ['PTV70', 'PTV63', 'PTV56']
PTV_W = {'PTV70': 70.0 / 70.0, 'PTV63': 63.0 / 70.0, 'PTV56': 56.0 / 70.0}


def build_mddose(ckpt, device='cuda'):
    """MD-Dose (MambaUnet with the structure encoder, 10 condition channels) with the 'net_model' weights of ckpt, frozen."""
    _add_path(os.path.join(MODELS, 'mddose'))
    from vim.mamba_struc_unet import MambaUnet
    ck = torch.load(ckpt, map_location='cpu')
    net = MambaUnet(inp_channels=1, inp_struc_channels=10)
    net.load_state_dict(ck['net_model'])
    net.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


class MDCase:
    """One preprocessed OpenKBP case for MD-Dose: CT in [0, 1], 10 condition channels [CT, fused PTV, 7 OARs + body]."""

    def __init__(self, case_dir):
        import SimpleITK as sitk

        def read(name):
            img = sitk.ReadImage(os.path.join(case_dir, name))
            return sitk.GetArrayFromImage(img).astype(np.float32), img
        self.pid = os.path.basename(os.path.normpath(case_dir))
        ct_raw, ref = read('ct.nii.gz')
        self.ct = np.clip(ct_raw, 0, 2500) / 2500.0
        sp = ref.GetSpacing()
        self.spacing_zyx = (sp[2], sp[1], sp[0])
        self.oar = {s: read('Mask_%s.nii.gz' % s)[0] for s in OARS_MD}
        self.ptv = {s: read('Mask_%s.nii.gz' % s)[0] for s in PTVS_MD}
        self.body = read('Mask_possible_dose_mask.nii.gz')[0]
        self.gt = read('dose.nii.gz')[0]
        self.present = [s for s in OARS_MD + PTVS_MD if (self.oar.get(s) if s in OARS_MD else self.ptv[s]).sum() > 0]

    def mask(self, s):
        return self.oar[s] if s in OARS_MD else self.ptv[s]

    def condition(self, drop=None):
        """(slices, 10, 128, 128); drop = structure name (an OAR channel set to zero, or a PTV left out of the fused PTV) or None."""
        fused = np.zeros_like(self.ct)
        for p in PTVS_MD:
            if p != (drop if drop in PTVS_MD else None):
                fused = fused + PTV_W[p] * self.ptv[p]
        chans = [self.ct, fused]
        for s in ['Brainstem', 'Esophagus', 'Larynx', 'LeftParotid', 'Mandible']:
            chans.append(np.zeros_like(self.ct) if drop == s else self.oar[s])
        chans.append(self.body)
        for s in ['RightParotid', 'SpinalCord']:
            chans.append(np.zeros_like(self.ct) if drop == s else self.oar[s])
        return torch.from_numpy(np.stack(chans, axis=0)).float().permute(1, 0, 2, 3)

    def x_T(self):
        g = torch.Generator().manual_seed(zlib.crc32(self.pid.encode()) % (2 ** 31))
        return torch.randn(self.ct.shape[0], 1, 128, 128, generator=g)


class MDDoseSpec:
    """MD-Dose's DDIM: linear beta 1e-4 ... 0.02 (T = 1000), steps sorted({round(i T / N)}), x0 clipped to [0, 1]."""
    lo, hi, gy_scale = 0.0, 1.0, 80.0

    def __init__(self, model, case, ddim_steps=8, T=1000, beta_1=1e-4, beta_T=0.02):
        self.model, self.case = model, case
        betas = torch.linspace(beta_1, beta_T, T).double()
        self.alphas_bar = torch.cumprod(1. - betas, dim=0).cuda()
        self.seq = sorted({int(round(i * T / ddim_steps)) for i in range(ddim_steps)})
        self.n_steps = ddim_steps
        self.cond = case.condition().cuda()
        self._drop = {}

    def x_T(self):
        return self.case.x_T().cuda()

    def _ts(self, i):
        return self.seq[i], (self.seq[i - 1] if i > 0 else -1)

    def eps_all(self, x, i):
        t = x.new_full((x.size(0),), self._ts(i)[0], dtype=torch.long)
        return self.model(x, self.cond, t)

    def eps_slices(self, x, i, zs):
        t = x.new_full((x.size(0),), self._ts(i)[0], dtype=torch.long)
        return self.model(x, self.cond[zs], t)

    def x0_drop(self, x, i, name):
        if name not in self._drop:
            self._drop[name] = self.case.condition(drop=name).cuda()
        t = x.new_full((x.size(0),), self._ts(i)[0], dtype=torch.long)
        return self.x0_from_eps(x, self.model(x, self._drop[name], t), i)

    def x0_from_eps(self, x, eps, i):
        a_t = self.alphas_bar[self._ts(i)[0]].float()
        return (x - (1. - a_t).sqrt() * eps) / a_t.sqrt()

    def next_x(self, x, x0, i):
        t_cur, t_prev = self._ts(i)
        a_t = self.alphas_bar[t_cur].float()
        a_prev = self.alphas_bar[t_prev].float() if t_prev >= 0 else x.new_tensor(1.0)
        eps_c = (x - a_t.sqrt() * x0) / (1. - a_t).sqrt()
        return a_prev.sqrt() * x0 + (1. - a_prev).sqrt() * eps_c

    def to_gy(self, x0):
        return x0 * 80.0

    def final_dose(self, x):
        return (np.clip(x[:, 0].cpu().numpy() * 80.0, 0, 80) * self.case.body).astype(np.float32)
