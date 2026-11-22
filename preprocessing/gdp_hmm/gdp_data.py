"""GDP-HMM (AAPM 2025) data layer for DoseDiff-style 2D slice diffusion.

Input channels follow the organisers' data_loader.py (CatStructures=True) plus
the two beam-geometry PSDMs of the DoseDiff authors' challenge entry. Per axial slice:

  ct  (1)   CT clipped to [-1000, 1000] HU / 500
  dis (47)
     0-2   cat_optptv : OPT structures x PDose/10, sorted by PDose desc
     3-5   cat_ptv    : clinical PTV structures x PDose/10, same order
     6-37  cat_oar    : 32 binary slots, HaN names -> slots 0-29, Lung -> 9-18,
                       only structures listed in Pat_Obj_DICT
     38    Body
     39    beam_plate (raw values)
     40    angle_plate, the 2D gantry-line plate tiled on every slice
     41-44 prompt planes [isVMAT, n_PTV, site, cohort]
     45    PSDM of a radius-3-voxel sphere at the isocenter
     46    PSDM of the angle plate (2D, tiled)
  dose (1)  reference dose in Gy after the D97 -> prescription rescale, clip [0,80],
            /40 - 1; zero outside Body.

Geometry: isocenter-centred crop of (96, 128, 160) voxels at native spacing
(2.0 x 2.5 x 2.5 mm), symmetric zero padding, no resizing.

Cache: one directory per plan holding bit-packed binary channels (uint64 per
voxel), fp16 continuous channels and a small meta file.
"""

import argparse
import csv
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

# organisers' lists; 'Chiasim' typo fixed to the NPZ key 'Chiasm'
HaN_OAR_LIST = ['Cochlea_L', 'Cochlea_R', 'Eyes', 'Lens_L', 'Lens_R', 'OpticNerve_L', 'OpticNerve_R',
                'Chiasm', 'LacrimalGlands', 'BrachialPlexus', 'Brain', 'BrainStem_03', 'Esophagus',
                'Lips', 'Lungs', 'Trachea', 'Posterior_Neck', 'Shoulders', 'Larynx-PTV', 'Mandible-PTV',
                'OCavity-PTV', 'ParotidCon-PTV', 'ParotidIps-PTV', 'Parotids-PTV', 'PharConst-PTV',
                'Submand-PTV', 'SubmandL-PTV', 'SubmandR-PTV', 'Thyroid-PTV', 'SpinalCord_05']
HaN_OAR_DICT = {n: i + 1 for i, n in enumerate(HaN_OAR_LIST)}
Lung_OAR_LIST = ['PTV_Ring.3-2', 'Total Lung-GTV', 'SpinalCord', 'Heart', 'LAD', 'Esophagus',
                 'BrachialPlexus', 'GreatVessels', 'Trachea', 'Body_Ring0-3']
Lung_OAR_DICT = {n: i + 10 for i, n in enumerate(Lung_OAR_LIST)}

N_OAR_SLOTS = 32
DIS_CHANNELS = 47
IN_SIZE = (96, 128, 160)
IMG = (128, 160)
DOSE_MAX_GY = 80.0
DOSE_DIV = 10.0          # dose_div_factor of the PTV maps
CT_LO, CT_HI, CT_DIV = -1000.0, 1000.0, 500.0
# bit layout of the packed binary channels
B_OAR, B_OPT, B_PTV, B_BODY, N_BITS = 0, 32, 35, 38, 39


# ----------------------------------------------------------------------------- meta
EXCLUDED = {'0522c0009+9Ag+MOS_29166'}   # corrupt in the first data release; the other 2727 training plans are used


def read_meta(csv_path, dev_split=None, site=None, min_presc=40.0):
    """Rows of meta_data.csv (optionally one dev_split and site) prescribed at least `min_presc` Gy, without EXCLUDED."""
    rows = list(csv.DictReader(open(csv_path)))
    if dev_split is not None:
        rows = [r for r in rows if r['dev_split'] == dev_split]
    if site is not None:
        rows = [r for r in rows if int(r['site']) == site]
    rows = [r for r in rows if plan_id(r) not in EXCLUDED]
    return [r for r in rows if float(r['HighDose']) >= min_presc]


def plan_id(row):
    return os.path.basename(row['npz_path']).replace('.npz', '')


# ------------------------------------------------------------------------ geometry
def _crop_slices(center, size, shape):
    src, dst = [], []
    for c, s, n in zip(center, size, shape):
        start = max(int(round(c)) - s // 2, 0)
        end = start + s
        avail = min(end, n) - start
        pad_front = (s - avail) // 2 if end > n else 0
        src.append(slice(start, start + avail))
        dst.append(slice(pad_front, pad_front + avail))
    return tuple(src), tuple(dst)


def crop_pad(vol, center, size):
    """Isocenter-centred crop to `size` with symmetric zero padding (any #dims)."""
    src, dst = _crop_slices(center, size, vol.shape)
    out = np.zeros(size, dtype=vol.dtype)
    out[dst] = vol[src]
    return out


def paste_back(crop, center, size, ori_shape):
    """Puts a crop back on the original grid (zeros outside the crop)."""
    src, dst = _crop_slices(center, size, ori_shape)
    out = np.zeros(ori_shape, dtype=crop.dtype)
    out[src] = crop[dst]
    return out


def signed_distance(mask, spacing):
    """DoseDiff PSDM: (edt(inside) - edt(outside)) / 100 with physical spacing (mm)."""
    m = mask > 0
    if not m.any():
        return np.zeros(mask.shape, np.float32)
    if m.all():
        return (ndimage.distance_transform_edt(m, sampling=spacing) / 100.0).astype(np.float32)
    inside = ndimage.distance_transform_edt(m, sampling=spacing)
    outside = ndimage.distance_transform_edt(~m, sampling=spacing)
    return ((inside - outside) / 100.0).astype(np.float32)


# ---------------------------------------------------------------------- one plan
class GDPPlan:
    """Loads one NPZ plan and produces the packed representation (see module doc)."""

    def __init__(self, npz_path, row, ptv_dict, pat_obj_dict, data_root='', in_size=IN_SIZE):
        self.id = plan_id(row)
        self.pid = self.id.split('+')[0]
        if len(self.pid) < 3:
            self.pid = self.pid.zfill(3)
        self.in_size = tuple(in_size)
        d = np.load(os.path.join(data_root, npz_path), allow_pickle=True)['arr_0'].item()
        self.ori_shape = tuple(d['img'].shape)
        self.isocenter = [float(v) for v in d['isocenter']]
        sp = d['spacing']                                   # SimpleITK (x, y, z)
        self.spacing_zyx = (float(sp[2]), float(sp[1]), float(sp[0]))
        self.site, self.cohort = int(row['site']), int(row['cohort'])
        self.is_vmat = bool(d['isVMAT'])
        pinfo = ptv_dict[self.pid]
        self.presc_high = float(pinfo['PTV_High']['PDose'])
        cp = lambda v: crop_pad(v, self.isocenter, self.in_size)

        # ---- reference dose (Gy): dose_scale, then D97(PTV-high OPT) -> prescription
        self.has_dose = 'dose' in d
        body_full = (d['Body'] > 0)
        self.body_full = body_full.astype(np.uint8)
        if self.has_dose:
            dose = d['dose'].astype(np.float32) * float(d['dose_scale'])
            ptv_opt = d[pinfo['PTV_High']['OPTName']].astype(bool)
            dose = dose * (self.presc_high / (np.percentile(dose[ptv_opt], 3) + 1e-5))
            self.dose_gy = (dose * body_full).astype(np.float32)

        # ---- PTV maps: (mask, PDose) per level, sorted by PDose descending
        opt_maps, struct_maps = [], []
        for lvl in ('PTV_High', 'PTV_Mid', 'PTV_Low'):
            if lvl not in pinfo:
                continue
            pd = float(pinfo[lvl]['PDose'])
            opt_maps.append((d[pinfo[lvl]['OPTName']] > 0, pd))
            sname = pinfo[lvl]['OPTName'] if lvl == 'PTV_High' else pinfo[lvl].get('StructName')
            sname = sname if sname in d else pinfo[lvl]['OPTName']
            struct_maps.append((d[sname] > 0, pd))
        opt_maps.sort(key=lambda t: -t[1])
        struct_maps.sort(key=lambda t: -t[1])
        self.n_ptv = len(opt_maps)
        self.pd_opt = np.array([t[1] for t in opt_maps] + [0.0] * (3 - len(opt_maps)), np.float32)
        self.pd_struct = np.array([t[1] for t in struct_maps] + [0.0] * (3 - len(struct_maps)), np.float32)

        # ---- OAR slots: only structures used in optimisation
        oar_dict = HaN_OAR_DICT if self.site < 1.5 else Lung_OAR_DICT
        need = pat_obj_dict.get(self.id.split('+')[0], list(oar_dict))
        bits = np.zeros(self.in_size, np.uint64)
        for name, idx in oar_dict.items():
            if name in need and name in d:
                bits |= (cp(d[name] > 0).astype(np.uint64) << np.uint64(B_OAR + idx - 1))
        for k, (m, _) in enumerate(opt_maps):
            bits |= (cp(m).astype(np.uint64) << np.uint64(B_OPT + k))
        for k, (m, _) in enumerate(struct_maps):
            bits |= (cp(m).astype(np.uint64) << np.uint64(B_PTV + k))
        bits |= (cp(body_full).astype(np.uint64) << np.uint64(B_BODY))
        self.bits = bits

        # ---- continuous channels: ct, beam_plate, isocenter PSDM, dose (Gy)
        ct = np.clip(d['img'], CT_LO, CT_HI).astype(np.float32) / CT_DIV
        beam = d['beam_plate'].astype(np.float32)
        zz, yy, xx = np.indices(self.ori_shape)
        iz, iy, ix = [int(round(c)) for c in self.isocenter]
        sphere = ((zz - iz) ** 2 + (yy - iy) ** 2 + (xx - ix) ** 2) <= 9   # radius 3 voxels
        iso_psdm = signed_distance(cp(sphere), self.spacing_zyx)
        dose_c = cp(self.dose_gy) if self.has_dose else np.zeros(self.in_size, np.float32)
        self.cont = np.stack([cp(ct), cp(beam), iso_psdm, dose_c], 0).astype(np.float16)

        # ---- 2D angle plate (+ its PSDM), same for every slice
        ap = d['angle_plate'].astype(np.float32)
        if ap.shape != self.ori_shape[1:]:
            ap = ndimage.zoom(ap, (self.ori_shape[1] / ap.shape[0], self.ori_shape[2] / ap.shape[1]), order=0)
        ap2 = crop_pad((ap > 0).astype(np.uint8), self.isocenter[1:], self.in_size[1:])
        self.angle2d = ap2
        self.angle_psdm2d = signed_distance(ap2, self.spacing_zyx[1:]).astype(np.float16)
        self.prompt = np.array([float(self.is_vmat), float(self.n_ptv), float(self.site), float(self.cohort)],
                               np.float32)
        body_c = (bits >> np.uint64(B_BODY)) & np.uint64(1)
        self.valid_z = np.nonzero(body_c.reshape(self.in_size[0], -1).max(1))[0].astype(np.int32)

    def packed(self):
        return {'bits': self.bits, 'cont': self.cont, 'angle2d': self.angle2d,
                'angle_psdm2d': self.angle_psdm2d, 'prompt': self.prompt, 'pd_opt': self.pd_opt,
                'pd_struct': self.pd_struct, 'valid_z': self.valid_z, 'has_dose': self.has_dose}

    def save(self, cache_dir):
        out = os.path.join(cache_dir, self.id)
        os.makedirs(out, exist_ok=True)
        np.save(os.path.join(out, 'bits.npy'), self.bits)
        np.save(os.path.join(out, 'cont.npy'), self.cont)
        np.savez(os.path.join(out, 'meta.npz'), angle2d=self.angle2d, angle_psdm2d=self.angle_psdm2d,
                 prompt=self.prompt, pd_opt=self.pd_opt, pd_struct=self.pd_struct, valid_z=self.valid_z,
                 has_dose=np.array(self.has_dose), site=np.array(self.site))
        open(os.path.join(out, 'DONE'), 'w').close()

    def to_original_grid(self, pred_norm):
        """model output (Z,Y,X) in [-1,1] -> Gy on the original grid, Body-masked."""
        gy = (np.asarray(pred_norm, np.float32) + 1) * (DOSE_MAX_GY / 2)
        gy = np.clip(gy, 0, DOSE_MAX_GY)
        return paste_back(gy, self.isocenter, self.in_size, self.ori_shape) * self.body_full


# ------------------------------------------------------- packed -> network channels
def _bit_planes(bits):
    """uint64 bit-packed voxels (...) -> float32 (N_BITS, ...), one contiguous plane per bit."""
    bits = np.asarray(bits)
    planes = np.empty((N_BITS,) + bits.shape, np.float32)
    for k in range(N_BITS):
        planes[k] = (bits >> np.uint64(k)) & np.uint64(1)
    return planes


def assemble(p, z=None):
    """Build (ct, dis, dose_norm) float32 arrays from a packed dict: one slice z gives
    (C, H, W) arrays, the whole volume (z=None) C-contiguous (Z, C, H, W) arrays."""
    vol = z is None
    b = _bit_planes(p['bits'] if vol else p['bits'][z])                      # (39, [Z,] H, W)
    cont = np.asarray(p['cont'] if vol else p['cont'][:, z], np.float32)   # (4, [Z,] H, W)
    sh = b.shape[1:]
    sc_opt, sc_ptv = p['pd_opt'] / DOSE_DIV, p['pd_struct'] / DOSE_DIV
    planes = ([b[B_OPT + k] * sc_opt[k] for k in range(3)]
              + [b[B_PTV + k] * sc_ptv[k] for k in range(3)]
              + [b[B_OAR + k] for k in range(N_OAR_SLOTS)]
              + [b[B_BODY], cont[1], np.broadcast_to(np.asarray(p['angle2d'], np.float32), sh)]
              + [np.full(sh, float(v), np.float32) for v in p['prompt']]
              + [cont[2], np.broadcast_to(np.asarray(p['angle_psdm2d'], np.float32), sh)])
    dose = np.clip(cont[3], 0, DOSE_MAX_GY) / (DOSE_MAX_GY / 2) - 1.0
    if not vol:
        return cont[0:1], np.stack(planes, 0), dose[None]
    dis = np.empty((sh[0], len(planes)) + sh[1:], np.float32)
    for c, plane in enumerate(planes):
        dis[:, c] = plane
    return cont[0][:, None], dis, dose[:, None]


def load_packed(cache_dir, pid, mmap=True):
    out = os.path.join(cache_dir, pid)
    mode = 'r' if mmap else None
    m = np.load(os.path.join(out, 'meta.npz'))
    return {'bits': np.load(os.path.join(out, 'bits.npy'), mmap_mode=mode),
            'cont': np.load(os.path.join(out, 'cont.npy'), mmap_mode=mode),
            'angle2d': m['angle2d'], 'angle_psdm2d': m['angle_psdm2d'], 'prompt': m['prompt'],
            'pd_opt': m['pd_opt'], 'pd_struct': m['pd_struct'], 'valid_z': m['valid_z'],
            'has_dose': bool(m['has_dose']), 'site': int(m['site'])}


# ----------------------------------------------------------------------- augment
_NEAREST = list(range(0, 39)) + [40] + list(range(41, 45))   # masks, PTV maps, angle plate, prompt
_BILINEAR = [39, 45, 46]                                       # beam plate, PSDMs


def augment(ct, dis, dose, rng):
    """Per-slice random LR/AP flips and a small in-plane rotation. Tensors (C,H,W)."""
    if rng.random() < 0.5:
        ct, dis, dose = ct.flip(-1), dis.flip(-1), dose.flip(-1)
    if rng.random() < 0.3:
        ct, dis, dose = ct.flip(-2), dis.flip(-2), dose.flip(-2)
    if rng.random() < 0.3:
        ang = float(rng.uniform(-15, 15)) * np.pi / 180
        theta = torch.tensor([[np.cos(ang), -np.sin(ang), 0.], [np.sin(ang), np.cos(ang), 0.]], dtype=torch.float32)
        grid = F.affine_grid(theta[None], (1, 1) + tuple(ct.shape[-2:]), align_corners=False)
        rot = lambda t, mode: F.grid_sample(t[None], grid, mode=mode, padding_mode='zeros', align_corners=False)[0]
        ct, dose = rot(ct, 'bilinear'), rot(dose, 'bilinear')
        d2 = dis.clone()
        d2[_NEAREST] = rot(dis[_NEAREST], 'nearest')
        d2[_BILINEAR] = rot(dis[_BILINEAR], 'bilinear')
        dis = d2
    return ct, dis, dose


# ---------------------------------------------------------------------- datasets
class GDPSliceDataset(torch.utils.data.Dataset):
    """Uniform sampling over (plan, axial slice) pairs from the packed cache."""

    def __init__(self, csv_path, dev_split, cache_dir, train=True, site=None, max_plans=None):
        rows = read_meta(csv_path, dev_split, site)
        if max_plans:
            rows = rows[:max_plans]
        self.cache_dir, self.train = cache_dir, train
        self.pids, self.index = [], []
        for r in rows:
            pid = plan_id(r)
            done = os.path.join(cache_dir, pid, 'DONE')
            if not os.path.exists(done):
                continue
            vz = np.load(os.path.join(cache_dir, pid, 'meta.npz'))['valid_z']
            self.pids.append(pid)
            self.index += [(pid, int(z)) for z in vz]
        if len(self.pids) < len(rows):
            print('[GDPSliceDataset] %d/%d plans have no cache yet (run preprocessing/gdp_hmm/gdp_data.py)' % (len(rows) - len(self.pids), len(rows)))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        pid, z = self.index[i]
        ct, dis, dose = (torch.from_numpy(a) for a in assemble(load_packed(self.cache_dir, pid), z))
        if self.train:
            ct, dis, dose = augment(ct, dis, dose, np.random.default_rng())
        return ct, dis, dose


def plan_tensors(cache_dir, pid):
    """All valid slices of one cached plan: (ct, dis, dose) tensors of shape (Z_valid, C, H, W)."""
    p = load_packed(cache_dir, pid, mmap=False)
    ct, dis, dose = assemble(p)
    vz = p['valid_z']
    return torch.from_numpy(ct[vz]), torch.from_numpy(dis[vz]), torch.from_numpy(dose[vz]), p


# ------------------------------------------------------------------- cache build
def _build_one(args):
    row, ptv_json, obj_json, data_root, cache_dir = args
    try:
        if os.path.exists(os.path.join(cache_dir, plan_id(row), 'DONE')):
            return None
        ptv, obj = json.load(open(ptv_json)), json.load(open(obj_json))
        GDPPlan(row['npz_path'], row, ptv, obj, data_root).save(cache_dir)
        return None
    except Exception as e:  # keep the pool alive; report at the end
        return '%s\t%r' % (row['npz_path'], e)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Build the packed slice cache.')
    ap.add_argument('--gdp_root', default='data/gdp_hmm', help='GDP-HMM challenge folder with meta_files/ and data/')
    ap.add_argument('--cache_dir', default='data/gdp_hmm/cache')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--dev_split', default=None, help="train, valid or test (default: all splits)")
    a = ap.parse_args()
    from multiprocessing import Pool
    meta = os.path.join(a.gdp_root, 'meta_files/meta_data.csv')
    rows = read_meta(meta, a.dev_split)
    os.makedirs(a.cache_dir, exist_ok=True)
    jobs = [(r, os.path.join(a.gdp_root, 'meta_files/PTV_DICT.json'),
             os.path.join(a.gdp_root, 'meta_files/Pat_Obj_DICT.json'), a.gdp_root, a.cache_dir) for r in rows]
    fails = []
    with Pool(a.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_build_one, jobs, chunksize=4)):
            if res:
                fails.append(res)
            if (i + 1) % 200 == 0:
                print('%d/%d done, %d failed' % (i + 1, len(jobs), len(fails)), flush=True)
    print('CACHE BUILD COMPLETE: %d plans, %d failed' % (len(jobs), len(fails)))
    for f in fails:
        print('FAILED', f)
