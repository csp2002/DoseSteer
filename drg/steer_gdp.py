"""Test on GDP-HMM: the unsteered prediction and DRG steering of DoseDiff.

For every test plan the targets are the structure metrics of the reference dose (OAR mean dose, PTV D95, inside body ∩ crop);
the unsteered prediction is the model's deterministic 8-step DDIM sample, and DRG corrects the last four steps (i = 3, 2, 1, 0)
from the same initial noise. Both are scored with the GDP-HMM challenge score (`official`), the DVH score, the target error and
the voxel errors inside / outside the structures. Results go to <out>/metrics*.json (per plan); --save_dose also writes the doses
(float16 .npy). Plans can be split over GPUs with --shard / --nshards; the cohort summary reads every metrics file in <out>.

    python drg/steer_gdp.py --ckpt runs/gdp_hmm/<run>/model_best_mae.pth
    python drg/steer_gdp.py --ckpt ... --gpu 1 --shard 1 --nshards 4        # one of four shards
"""
import argparse
import glob
import json
import os
import sys
import time

for _i, _a in enumerate(sys.argv):                                  # pin the GPU before torch is imported
    if _a == '--gpu' and _i + 1 < len(sys.argv):
        os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[_i + 1]
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the repository root
from drg import core                                                # noqa: E402
from drg import specs_gdp as specs                                  # noqa: E402
from gdp_data import plan_id                                        # noqa: E402  (preprocessing/gdp_hmm, put on the path by specs_gdp)

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--gdp_root', default='data/gdp_hmm', help='GDP-HMM folder with meta_files/ and data/')
ap.add_argument('--split', default='test', help="dev_split of meta_data.csv")
ap.add_argument('--method', default='both', choices=('unsteered', 'drg', 'both'))
ap.add_argument('--plans', default='all', help="'all' or comma-separated plan ids (npz names, e.g. 0522c0002+15Ag+MOS_21176)")
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--nshards', type=int, default=1)
ap.add_argument('--out', default='results/gdp_hmm')
ap.add_argument('--save_dose', action='store_true')
ap.add_argument('--gpu', default='0')
args = ap.parse_args()
torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))

device = 'cuda'
net = specs.build_model(args.ckpt, device)
diffusion = specs.build_diffusion(1000, 8)
methods = ['unsteered', 'drg'] if args.method == 'both' else [args.method]
drg_kw = dict(core.DRG_GDP, chunk=specs.CHUNK)
ptv_dict, obj_dict = specs.load_dicts(args.gdp_root)
rows = specs.plan_rows(args.gdp_root, args.split)
if args.plans != 'all':
    want = set(args.plans.split(','))
    rows = [r for r in rows if plan_id(r) in want]
rows = rows[args.shard::args.nshards]
os.makedirs(args.out, exist_ok=True)
res_f = os.path.join(args.out, 'metrics.json' if args.nshards == 1 else 'metrics_s%dof%d.json' % (args.shard, args.nshards))
res = json.load(open(res_f)) if os.path.exists(res_f) else {}

for n, row in enumerate(rows):
    pid = plan_id(row)
    if all(m in res.get(pid, {}) for m in methods):
        continue
    t0 = time.time()
    case = specs.GDPCase(row, ptv_dict, obj_dict, args.gdp_root)
    spec = specs.GDPSpec(net, diffusion, case, device)
    P = specs.GDPStructs(case, device)
    gt_dvh = specs.dvh_metrics(case.gt, case)
    rec = res.setdefault(pid, {})
    rec['targets'] = {s['name']: s['target'] for s in case.structures}
    for m in methods:
        dose = core.sample(spec, P) if m == 'unsteered' else core.sample(spec, P, core.DRG_STEPS, drg_kw)
        rec[m] = specs.evaluate(dose, case, gt_dvh)
        if args.save_dose:
            os.makedirs(os.path.join(args.out, 'doses'), exist_ok=True)
            np.save(os.path.join(args.out, 'doses', '%s__%s.npy' % (pid, m)), dose.astype(np.float16))
    json.dump(res, open(res_f + '.tmp', 'w'), indent=1)
    os.replace(res_f + '.tmp', res_f)
    print('%d/%d %s %.0f s  %s' % (n + 1, len(rows), pid, time.time() - t0,
                                   '  '.join('%s official %.3f target %.3f' % (m, rec[m]['official'], rec[m]['target']) for m in methods)),
          flush=True)

allres = {}
for f in sorted(glob.glob(os.path.join(args.out, 'metrics*.json'))):
    allres.update(json.load(open(f)))
done = [c for c in allres if all(m in allres[c] for m in methods)]
if done:
    print('cohort mean over %d plans (all metrics files in %s):' % (len(done), args.out))
    for m in methods:
        print('  %-9s %s' % (m, '  '.join('%s %.4f' % (k, np.nanmean([allres[c][m][k] for c in done])) for k in
                                          ('official', 'dvh', 'target', 'within', 'outside'))))
    if len(methods) == 2:
        o = np.array([allres[c]['unsteered']['official'] for c in done], float)
        d = np.array([allres[c]['drg']['official'] for c in done], float)
        ok = np.isfinite(o) & np.isfinite(d)
        print('  official improvement of DRG: %.2f %%' % (100 * (1 - d[ok].mean() / o[ok].mean())))
