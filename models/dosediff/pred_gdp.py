"""Test DoseDiff on GDP-HMM and score Metric 1.

  python models/dosediff/pred_gdp.py --ckpt runs/gdp_hmm/<run>/model_best_mae.pth

Writes <ID>_pred.npy (Gy, original grid, Body-masked), per_plan.csv and summary.json
(cohort means, overall and per site).
"""

import argparse
import csv
import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path[:0] = [_HERE, os.path.join(_ROOT, 'models'), os.path.join(_ROOT, 'preprocessing', 'gdp_hmm')]  # guided_diffusion; evaluate_gdp; gdp_data
for _i, _a in enumerate(sys.argv):
    if _a == '--gpu' and _i + 1 < len(sys.argv):
        os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[_i + 1]


def _pin_to_freest_numa_node():
    """Run on the CPUs of the NUMA node with the most free memory."""
    try:
        nodes = []
        for nd in glob.glob('/sys/devices/system/node/node[0-9]*'):
            free_kb = next(int(l.split()[-2]) for l in open(nd + '/meminfo') if 'MemFree' in l)
            cpus = set()
            for part in open(nd + '/cpulist').read().strip().split(','):
                lo, _, hi = part.partition('-')
                cpus.update(range(int(lo), int(hi or lo) + 1))
            nodes.append((free_kb, os.path.basename(nd), cpus))
        if len(nodes) > 1:
            free_kb, name, cpus = max(nodes, key=lambda t: t[0])
            os.sched_setaffinity(0, cpus)
            print('pinned to %s (%.1f GB free, %d CPUs)' % (name, free_kb / 1048576, len(cpus)), flush=True)
    except (OSError, StopIteration, ValueError):
        pass


_pin_to_freest_numa_node()

import numpy as np
import torch
from datetime import datetime

from guided_diffusion import gaussian_diffusion as gd
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.unet import UNetModel_MS_Former
from gdp_data import DIS_CHANNELS, IMG, GDPPlan, assemble, read_meta
from evaluate_gdp import body_mae, gdp_mae, gdp_metric1, ptv_stats, summarize

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--ckpt', type=str, required=True)
parser.add_argument('--dev_split', type=str, default='test', help="'valid' (148, model selection) or 'test' (356, held-out)")
parser.add_argument('--site', type=int, default=None)
parser.add_argument('--ddim', type=int, default=8)
parser.add_argument('--T', type=int, default=1000)
parser.add_argument('--model_channels', type=int, default=128)
parser.add_argument('--gdp_root', type=str, default='data/gdp_hmm')
parser.add_argument('--out_root', type=str, default='runs/gdp_hmm/results')
parser.add_argument('--max_plans', type=int, default=None)
parser.add_argument('--stride', type=int, default=1, help='every k-th plan')
parser.add_argument('--offset', type=int, default=0, help='start index for --stride (shard k of n = --offset k --stride n)')
parser.add_argument('--bs', type=int, default=96, help='slices per sampling batch')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

out_dir = os.path.join(args.out_root, '{}__{}_ddim{}_{}'.format(os.path.basename(os.path.dirname(args.ckpt)), args.dev_split,
                                                               args.ddim, datetime.now().strftime('%Y%m%d_%H%M%S')))
if args.offset or args.stride > 1:                      # one output folder per shard
    out_dir += '_s%dof%d' % (args.offset, args.stride)
os.makedirs(os.path.join(out_dir, 'predictions'), exist_ok=True)
print(out_dir)

diffusion = SpacedDiffusion(use_timesteps=space_timesteps(args.T, 'ddim%d' % args.ddim),
                            betas=gd.get_named_beta_schedule('linear', args.T),
                            model_mean_type=gd.ModelMeanType.EPSILON, model_var_type=gd.ModelVarType.FIXED_LARGE,
                            loss_type=gd.LossType.MSE, rescale_timesteps=False)
net = UNetModel_MS_Former(image_size=IMG, in_channels=1, ct_channels=1, dis_channels=DIS_CHANNELS,
                          model_channels=args.model_channels, out_channels=1, num_res_blocks=2,
                          attention_resolutions=(16, 32), dropout=0, channel_mult=(1, 1, 2, 3, 4), conv_resample=True,
                          dims=2, num_classes=None, use_checkpoint=False, use_fp16=False, num_heads=4,
                          num_head_channels=-1, num_heads_upsample=-1, use_scale_shift_norm=True,
                          resblock_updown=False, use_new_attention_order=False).cuda().eval()
sd = torch.load(args.ckpt, map_location='cpu')
net.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()})

rows = read_meta(os.path.join(args.gdp_root, 'meta_files/meta_data.csv'), args.dev_split, args.site)
rows = rows[args.offset::args.stride]
if args.max_plans:
    rows = rows[:args.max_plans]
ptv_dict = json.load(open(os.path.join(args.gdp_root, 'meta_files/PTV_DICT.json')))
obj_dict = json.load(open(os.path.join(args.gdp_root, 'meta_files/Pat_Obj_DICT.json')))

records = []
with torch.no_grad():
    for i, row in enumerate(rows):
        plan = GDPPlan(row['npz_path'], row, ptv_dict, obj_dict, data_root=args.gdp_root)
        ct, dis, _ = assemble(plan.packed())                     # (Z, C, H, W) float32
        pred = np.full(ct.shape[0:1] + IMG, -1.0, np.float32)
        vz = plan.valid_z
        torch.manual_seed(args.seed)
        for s in range(0, len(vz), args.bs):
            sel = vz[s:s + args.bs]
            c, d_ = torch.from_numpy(ct[sel]).cuda(), torch.from_numpy(dis[sel]).cuda()
            out = diffusion.ddim_sample_loop(net, (c.size(0), 1, IMG[0], IMG[1]), clip_denoised=True, eta=0.0,
                                             model_kwargs={'ct': c, 'dis': d_}, progress=False)
            pred[sel] = out[:, 0].cpu().numpy()
        dose_gy = plan.to_original_grid(pred)
        np.save(os.path.join(out_dir, 'predictions', plan.id + '_pred.npy'), dose_gy.astype(np.float32))
        rec = {'id': plan.id, 'site': plan.site, 'isVMAT': plan.is_vmat, 'presc': plan.presc_high}
        if plan.has_dose:
            rec['metric1'] = gdp_metric1(dose_gy, plan.dose_gy, plan.body_full)   # leaderboard formula
            rec['mae5'] = gdp_mae(dose_gy, plan.dose_gy, plan.body_full)          # plain masked mean
            rec['mae_body'] = body_mae(dose_gy, plan.dose_gy, plan.body_full)
            d = np.load(os.path.join(args.gdp_root, row['npz_path']), allow_pickle=True)['arr_0'].item()
            rec.update({'pred_' + k: v for k, v in ptv_stats(dose_gy, d[ptv_dict[plan.pid]['PTV_High']['OPTName']]).items()})
        records.append(rec)
        print('%d/%d %s metric1=%.3f mae5=%.3f' % (i + 1, len(rows), plan.id, rec.get('metric1', float('nan')),
                                                   rec.get('mae5', float('nan'))), flush=True)

keys = sorted({k for r in records for k in r})
with open(os.path.join(out_dir, 'per_plan.csv'), 'w') as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(records)
summary = summarize(records)
print('SUMMARY', summary)
json.dump(summary, open(os.path.join(out_dir, 'summary.json'), 'w'), indent=1)
