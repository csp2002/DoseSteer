"""Test on OpenKBP: the unsteered prediction and DRG steering of DoseDiff, DoseDiT or MD-Dose.

For every test case the targets are the structure metrics of the reference dose (OAR mean dose, PTV D95); the unsteered prediction
is the model's deterministic 8-step DDIM sample, and DRG corrects the last four steps (i = 3, 2, 1, 0) from the same initial noise.
Both are scored with the OpenKBP dose score (`official`), the DVH score, the target error and the voxel errors inside / outside
the structures. Results go to <out>/metrics.json (per case); --save_dose also writes the doses (float16 .npy).

    python drg/steer_openkbp.py --model dosediff --ckpt runs/dosediff/<run>/model_best_mae.pth
    python drg/steer_openkbp.py --model mddose        # default --ckpt: the checkpoint models/mddose/train_struc.py writes
    python drg/steer_openkbp.py --model dosedit       # default --ckpt: the checkpoint models/dosedit/train.py writes
"""
import argparse
import json
import os
import sys
import time

for _i, _a in enumerate(sys.argv):                                  # pin the GPU before torch is imported
    if _a == '--gpu' and _i + 1 < len(sys.argv):
        os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[_i + 1]
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the repository root
from drg import core                                                # noqa: E402
from drg import specs_openkbp as specs                              # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True, choices=('dosediff', 'dosedit', 'mddose'))
ap.add_argument('--ckpt', default=None, help='required for dosediff; mddose / dosedit default to their training output')
ap.add_argument('--data_dir', default='data/openkbp/preprocessed/test-pats_preprocess', help='one folder per test case')
ap.add_argument('--method', default='both', choices=('unsteered', 'drg', 'both'))
ap.add_argument('--cases', default='all', help="'all' or comma-separated case ids (e.g. pt_241)")
ap.add_argument('--out', default=None, help='default: results/openkbp_<model>')
ap.add_argument('--save_dose', action='store_true')
ap.add_argument('--gpu', default='0')
args = ap.parse_args()
DEFAULT_CKPT = {'mddose': 'runs/mddose/Save/ddpm-unet/hxdose-struc-mamba-unet/model_best_mae.pkl',
                'dosedit': 'runs/dosedit/DoseDiT_S4_T1000_bs24_epoch1600/ema_best_mae.pth'}
if args.ckpt is None:
    if args.model not in DEFAULT_CKPT:
        ap.error('--ckpt is required for --model dosediff (e.g. runs/dosediff/<run>/model_best_mae.pth)')
    args.ckpt = DEFAULT_CKPT[args.model]
args.out = args.out or 'results/openkbp_' + args.model
torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))

device = 'cuda'
if args.model == 'mddose':
    net = specs.build_mddose(args.ckpt, device)
else:
    net = specs.build_dosediff(args.ckpt, device) if args.model == 'dosediff' else specs.build_dosedit(args.ckpt, device)
    diffusion = specs.build_diffusion(1000, 8)
methods = ['unsteered', 'drg'] if args.method == 'both' else [args.method]
drg_kw = core.DRG_OPENKBP[args.model]
cases = sorted(os.listdir(args.data_dir)) if args.cases == 'all' else args.cases.split(',')
os.makedirs(args.out, exist_ok=True)
res_f = os.path.join(args.out, 'metrics.json')
res = json.load(open(res_f)) if os.path.exists(res_f) else {}

for n, pid in enumerate(cases):
    if all(m in res.get(pid, {}) for m in methods):
        continue
    t0 = time.time()
    if args.model == 'mddose':
        case = specs.MDCase(os.path.join(args.data_dir, pid))
        spec = specs.MDDoseSpec(net, case)
        P = core.PatientStructs(pid, {s: case.mask(s) > 0 for s in case.present}, case.gt.astype(np.float32), case.body > 0,
                                case.spacing_zyx, device)
    else:
        case = specs.load_case(os.path.join(args.data_dir, pid))
        spec = specs.DoseDiffSpec(net, diffusion, case, device)
        masks = case['masks']
        P = core.PatientStructs(pid, {s: masks[s] > 0 for s in core.OARS + core.PTVS if s in masks}, case['gt_dose'].astype(np.float32),
                                masks['possible_dose_mask'] > 0, case['spacing_zyx'], device)
    gt_dvh = core.dvh_metrics(P.gt, P)
    rec = res.setdefault(pid, {})
    rec['targets'] = {s: P.target[s] for s in P.names}
    for m in methods:
        dose = core.sample(spec, P) if m == 'unsteered' else core.sample(spec, P, core.DRG_STEPS, drg_kw)
        rec[m] = core.evaluate(dose, P, gt_dvh)
        if args.save_dose:
            os.makedirs(os.path.join(args.out, 'doses'), exist_ok=True)
            np.save(os.path.join(args.out, 'doses', '%s__%s.npy' % (pid, m)), dose.astype(np.float16))
    json.dump(res, open(res_f + '.tmp', 'w'), indent=1)
    os.replace(res_f + '.tmp', res_f)
    print('%d/%d %s %.0f s  %s' % (n + 1, len(cases), pid, time.time() - t0,
                                   '  '.join('%s official %.3f target %.3f' % (m, rec[m]['official'], rec[m]['target']) for m in methods)),
          flush=True)

done = [c for c in cases if all(m in res.get(c, {}) for m in methods)]
if done:
    print('cohort mean over %d cases:' % len(done))
    for m in methods:
        print('  %-9s %s' % (m, '  '.join('%s %.4f' % (k, np.mean([res[c][m][k] for c in done])) for k in
                                          ('official', 'dvh', 'target', 'within', 'outside'))))
    if len(methods) == 2:
        o = np.mean([res[c]['unsteered']['official'] for c in done])
        d = np.mean([res[c]['drg']['official'] for c in done])
        print('  official improvement of DRG: %.2f %%' % (100 * (1 - d / o)))
