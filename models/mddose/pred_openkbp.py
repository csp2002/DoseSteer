"""Test MD-Dose on OpenKBP: predict every case in --data_dir slice by slice, write the doses
(<out_root>/<run>/predictions/<case>/dose.nii.gz) and score them with models/evaluate_openKBP.py.

  python models/mddose/pred_openkbp.py
"""

import argparse
import os
import sys
from datetime import datetime

# pin the GPU before torch is imported (triton probes the driver at import)
for _i, _a in enumerate(sys.argv):
    if _a == '--gpu' and _i + 1 < len(sys.argv):
        os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[_i + 1]
    elif _a.startswith('--gpu='):
        os.environ['CUDA_VISIBLE_DEVICES'] = _a.split('=', 1)[1]

import numpy as np
import SimpleITK as sitk
import torch
from tqdm import tqdm

sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))  # evaluate_openKBP
from evaluate_openKBP import get_Dose_score_and_DVH_score

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='single GPU id')
parser.add_argument('--model_dir', type=str,
                    default='runs/mddose/Save/ddpm-unet/hxdose-struc-mamba-unet',
                    help='checkpoint directory written by train_struc.py')
parser.add_argument('--ckpt', type=str, default='model_best_mae.pkl',
                    help='checkpoint filename inside --model_dir')
parser.add_argument('--data_dir', type=str, default='data/openkbp/preprocessed/test-pats_preprocess')
parser.add_argument('--gt_dir', type=str, default='data/openkbp/preprocessed/test-pats')
parser.add_argument('--out_root', type=str, default='runs/mddose/results')
parser.add_argument('--use_ema', type=int, default=0, help='0 = net_model weights (the paper), 1 = ema_model')
parser.add_argument('--bs', type=int, default=128, help='slices per sampling batch')
parser.add_argument('--T', type=int, default=1000)
parser.add_argument('--beta_1', type=float, default=1e-4)
parser.add_argument('--beta_T', type=float, default=0.02)
parser.add_argument('--ddim', type=int, default=8,
                    help='N>0 = deterministic N-step DDIM (the paper: 8); 0 = original 1000-step DDPM sampler')
parser.add_argument('--init', type=str, default='randn', choices=['minmax', 'randn'],
                    help="x_T init: 'randn' = N(0, 1) (the paper); 'minmax' = min-max-scaled N(0, 1), as in train_struc.py")
args = parser.parse_args()

TEST_COND_DIR = args.data_dir
TEST_GT_DIR = args.gt_dir

# same layout as Dataset/datasets_openkbp.py: CT, fused PTV, then these masks
NON_PTV_MASKS = ['Mask_Brainstem', 'Mask_Esophagus', 'Mask_Larynx',
                 'Mask_LeftParotid', 'Mask_Mandible', 'Mask_possible_dose_mask',
                 'Mask_RightParotid', 'Mask_SpinalCord']
COND_CHANNELS = 10
IMG_SIZE = 128


def nii_read(path, as_type=np.float32):
    nii = sitk.ReadImage(path)
    volume = sitk.GetArrayFromImage(nii)  # (z, y, x)
    return volume.astype(as_type), nii


def nii_write_like(save_path, volume, ref_nii):
    out = sitk.GetImageFromArray(volume.astype(np.float32))
    out.SetSpacing(ref_nii.GetSpacing())
    out.SetOrigin(ref_nii.GetOrigin())
    out.SetDirection(ref_nii.GetDirection())
    sitk.WriteImage(out, save_path)


from diffusion_struc import GaussianDiffusionSampler
from vim.mamba_struc_unet import MambaUnet
net = MambaUnet(inp_channels=1, inp_struc_channels=COND_CHANNELS)

checkpoint = torch.load(os.path.join(args.model_dir, args.ckpt), map_location='cpu')
weights_key = 'ema_model' if args.use_ema else 'net_model'
net.load_state_dict(checkpoint[weights_key])
print('loaded {} ({}) from epoch {}'.format(
    os.path.join(args.model_dir, args.ckpt), weights_key, checkpoint.get('epoch')))
net.cuda().eval()

if args.ddim > 0:
    from ddim_struc import DDIMSampler
    sampler = DDIMSampler(net, args.beta_1, args.beta_T, args.T,
                          ddim_steps=args.ddim, struc=True).cuda()
    sampler_tag = 'ddim{}_{}'.format(args.ddim, args.init)
else:
    sampler = GaussianDiffusionSampler(
        net, args.beta_1, args.beta_T, args.T, img_size=IMG_SIZE,
        mean_type='epsilon', var_type='fixedlarge').cuda()
    sampler_tag = 'ddpm{}_{}'.format(args.T, args.init)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
run_tag = '{}_{}_{}_{}'.format(os.path.basename(os.path.normpath(args.model_dir)),
                               weights_key, sampler_tag, timestamp)
out_dir = os.path.join(args.out_root, run_tag)
os.makedirs(os.path.join(out_dir, 'predictions'), exist_ok=True)
print(out_dir)

with torch.no_grad():
    for ID in tqdm(sorted(os.listdir(TEST_COND_DIR))):
        pat_dir = os.path.join(TEST_COND_DIR, ID)
        ct_raw, ref_nii = nii_read(os.path.join(pat_dir, 'ct.nii.gz'))
        ct = np.clip(ct_raw, 0, 2500) / 2500.0  # [0,1], same source range as training

        masks = {m: nii_read(os.path.join(pat_dir, m + '.nii.gz'))[0] for m in NON_PTV_MASKS}
        ptv70 = nii_read(os.path.join(pat_dir, 'Mask_PTV70.nii.gz'))[0]
        ptv63 = nii_read(os.path.join(pat_dir, 'Mask_PTV63.nii.gz'))[0]
        ptv56 = nii_read(os.path.join(pat_dir, 'Mask_PTV56.nii.gz'))[0]
        ptvs = (70.0 / 70.0) * ptv70 + (63.0 / 70.0) * ptv63 + (56.0 / 70.0) * ptv56

        condition_vol = np.stack([ct, ptvs] + [masks[m] for m in NON_PTV_MASKS], axis=0)
        n_slices = condition_vol.shape[1]
        pred_vol = np.zeros((n_slices, IMG_SIZE, IMG_SIZE), np.float32)

        for start in range(0, n_slices, args.bs):
            stop = min(start + args.bs, n_slices)
            cond = torch.from_numpy(condition_vol[:, start:stop]).float().permute(1, 0, 2, 3).cuda()

            x_T = torch.randn(cond.size(0), 1, IMG_SIZE, IMG_SIZE)
            if args.init == 'minmax':
                x_T = (x_T - x_T.min()) / (x_T.max() - x_T.min())  # as in train_struc.py
            x_T = x_T.cuda()

            out = sampler(x_T, cond)  # DDPM: snapshot list; DDIM: final x_0 tensor
            final = out[-1] if isinstance(out, list) else out
            pred_vol[start:stop] = final[:, 0].cpu().numpy()

        pred_dose = np.clip(pred_vol * 80.0, 0, 80)  # [0,1] -> Gy
        pred_dose = pred_dose * masks['Mask_possible_dose_mask']

        os.makedirs(os.path.join(out_dir, 'predictions', ID), exist_ok=True)
        nii_write_like(os.path.join(out_dir, 'predictions', ID, 'dose.nii.gz'),
                       pred_dose, ref_nii)

dose_score, dose_std, dvh_score, dvh_std = get_Dose_score_and_DVH_score(
    prediction_dir=os.path.join(out_dir, 'predictions'), gt_dir=TEST_GT_DIR)
print('Dose_score: {} ({})'.format(dose_score, dose_std))
print('DVH_score: {} ({})'.format(dvh_score, dvh_std))
with open(os.path.join(out_dir, 'score.txt'), 'w') as f:
    f.write('Dose_score: {} {}\nDVH_score: {} {}'.format(dose_score, dose_std, dvh_score, dvh_std))
