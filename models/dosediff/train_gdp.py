"""Train DoseDiff (UNetModel_MS_Former, 2D axial slices) on GDP-HMM.

Build the cache first:  python preprocessing/gdp_hmm/gdp_data.py
Launch:  torchrun --nproc_per_node=1 models/dosediff/train_gdp.py

One "epoch" = `slices_per_epoch` slices drawn uniformly at random over all
(plan, slice) pairs of dev_split == 'train'. Every `val_every` epochs, the dev-valid
plans are sampled with a 4-step DDIM and scored on the crop grid; each validation
saves model_epoch<N>.pth, and model_best_mae.pth keeps the best Metric 1.
"""

import argparse
import csv
import os
import sys
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path[:0] = [_HERE, os.path.join(_ROOT, 'models'), os.path.join(_ROOT, 'preprocessing', 'gdp_hmm')]  # guided_diffusion; evaluate_gdp; gdp_data

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Sampler

from guided_diffusion import gaussian_diffusion as gd
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.unet import UNetModel_MS_Former
from gdp_data import (B_BODY, DIS_CHANNELS, DOSE_MAX_GY, IMG, GDPSliceDataset, plan_id, plan_tensors,
                      read_meta)
from evaluate_gdp import gdp_mae, gdp_metric1

BODY_CH = 38  # index of the Body channel inside `dis`

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--bs', type=int, default=32, help='slices per batch per GPU')
parser.add_argument('--slices_per_epoch', type=int, default=21824)
parser.add_argument('--T', type=int, default=1000)
parser.add_argument('--epoch', type=int, default=300)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--model_channels', type=int, default=128)
parser.add_argument('--site', type=int, default=None, help='1 = HaN, 2 = Lung (default: both)')
parser.add_argument('--gdp_root', type=str, default='data/gdp_hmm')
parser.add_argument('--cache_dir', type=str, default='data/gdp_hmm/cache')
parser.add_argument('--save_root', type=str, default='runs/gdp_hmm')
parser.add_argument('--val_plans', type=int, default=None, help='use only the first N dev-valid plans (default: all)')
parser.add_argument('--val_every', type=int, default=10)
parser.add_argument('--num_workers', type=int, default=6)
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--local_rank', '--local-rank', default=-1, type=int)
args = parser.parse_args()
if 'LOCAL_RANK' in os.environ:  # set by torchrun
    args.local_rank = int(os.environ['LOCAL_RANK'])
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

torch.cuda.set_device(args.local_rank)
dist.init_process_group(backend='nccl', init_method='env://')
rank, world = dist.get_rank(), dist.get_world_size()
rank0 = rank == 0


class RankRandomSampler(Sampler):
    """`num_samples` indices per rank, drawn with replacement; reseeded every epoch."""

    def __init__(self, n, num_samples, seed):
        self.n, self.num_samples, self.seed, self.epoch = n, num_samples, seed, 0

    def set_epoch(self, e):
        self.epoch = e

    def __iter__(self):
        g = np.random.default_rng(self.seed * 1000003 + self.epoch * 7919 + rank)
        return iter(g.integers(0, self.n, self.num_samples).tolist())

    def __len__(self):
        return self.num_samples


meta = os.path.join(args.gdp_root, 'meta_files/meta_data.csv')
train_data = GDPSliceDataset(meta, 'train', args.cache_dir, train=True, site=args.site)
sampler = RankRandomSampler(len(train_data), args.slices_per_epoch // world, args.seed)
train_loader = DataLoader(train_data, batch_size=args.bs, sampler=sampler, num_workers=args.num_workers,
                          pin_memory=True, persistent_workers=True, drop_last=True)
val_pids = [plan_id(r) for r in read_meta(meta, 'valid', args.site)][:args.val_plans]
val_pids = [p for p in val_pids if os.path.exists(os.path.join(args.cache_dir, p, 'DONE'))]
my_val = val_pids[rank::world]

model = UNetModel_MS_Former(image_size=IMG, in_channels=1, ct_channels=1, dis_channels=DIS_CHANNELS,
                            model_channels=args.model_channels, out_channels=1, num_res_blocks=2,
                            attention_resolutions=(16, 32), dropout=0, channel_mult=(1, 1, 2, 3, 4),
                            conv_resample=True, dims=2, num_classes=None, use_checkpoint=False, use_fp16=False,
                            num_heads=4, num_head_channels=-1, num_heads_upsample=-1, use_scale_shift_norm=True,
                            resblock_updown=False, use_new_attention_order=False).cuda()
if args.resume:
    sd = torch.load(args.resume, map_location='cpu')
    model.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()})
model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False,
            bucket_cap_mb=128, find_unused_parameters=False)

mk = lambda steps: SpacedDiffusion(use_timesteps=space_timesteps(args.T, steps),
                                   betas=gd.get_named_beta_schedule('linear', args.T),
                                   model_mean_type=gd.ModelMeanType.EPSILON, model_var_type=gd.ModelVarType.FIXED_LARGE,
                                   loss_type=gd.LossType.MSE, rescale_timesteps=False)
diffusion, diffusion_val = mk([args.T]), mk('ddim4')
schedule_sampler = create_named_schedule_sampler('uniform', diffusion)
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
lr_scheduler = MultiStepLR(optimizer, milestones=[int(0.7 * args.epoch)], gamma=0.1)

if rank0:
    tag = 'GDP_T{}_bs{}_e{}_{}'.format(args.T, args.bs * world, args.epoch, datetime.now().strftime('%Y%m%d_%H%M%S'))
    out_dir = os.path.join(args.save_root, tag)
    os.makedirs(out_dir, exist_ok=True)
    log_f = open(os.path.join(out_dir, 'log.csv'), 'a')
    log_w = csv.writer(log_f)
    log_w.writerow(['epoch', 'lr', 'train_loss', 'val_metric1', 'val_mae5'])
    print(tag, '| train slices %d (%d plans) | val plans %d' % (len(train_data), len(train_data.pids), len(val_pids)), flush=True)


@torch.no_grad()
def validate():
    m1s, mae5s = [], []
    for pid in my_val:
        ct, dis, dose, _ = plan_tensors(args.cache_dir, pid)
        pred = []
        for s in range(0, ct.shape[0], 96):
            c, d_ = ct[s:s + 96].cuda(), dis[s:s + 96].cuda()
            pred.append(diffusion_val.ddim_sample_loop(model=model, shape=(c.size(0), 1, IMG[0], IMG[1]),
                                                       clip_denoised=True, model_kwargs={'ct': c, 'dis': d_},
                                                       eta=0.0, progress=False).cpu())
        p = (torch.cat(pred)[:, 0].numpy() + 1) * (DOSE_MAX_GY / 2)
        g = (dose[:, 0].numpy() + 1) * (DOSE_MAX_GY / 2)
        body = dis[:, BODY_CH].numpy()
        m1s.append(gdp_metric1(p, g, body))
        mae5s.append(gdp_mae(p, g, body))
    t = torch.tensor([float(np.nansum(m1s)), float(np.nansum(mae5s)), float(len(m1s))], device='cuda')
    dist.all_reduce(t)
    return (t[0] / t[2]).item(), (t[1] / t[2]).item()


best = 1e9
for epoch in range(args.epoch):
    sampler.set_epoch(epoch)
    model.train()
    losses = []
    for i, (ct, dis, dose) in enumerate(train_loader):
        ct, dis, dose = ct.cuda(non_blocking=True), dis.cuda(non_blocking=True), dose.cuda(non_blocking=True)
        t, w = schedule_sampler.sample(dose.shape[0], dose.device)
        loss = (diffusion.training_losses(model=model, x_start=dose, t=t,
                                          model_kwargs={'ct': ct, 'dis': dis})['loss'] * w).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
        if rank0 and i % 100 == 0:
            print('[%d/%d %d/%d] loss %.4f' % (epoch + 1, args.epoch, i + 1, len(train_loader), loss.item()), flush=True)
    lr_scheduler.step()

    val_m1 = val_mae5 = float('nan')
    if epoch == 0 or (epoch + 1) % args.val_every == 0:
        model.eval()
        val_m1, val_mae5 = validate()
        if rank0:
            torch.save(model.state_dict(), os.path.join(out_dir, 'model_epoch%d.pth' % (epoch + 1)))
            if val_m1 < best:
                best = val_m1
                torch.save(model.state_dict(), os.path.join(out_dir, 'model_best_mae.pth'))
    if rank0:
        log_w.writerow([epoch + 1, optimizer.param_groups[0]['lr'], float(np.mean(losses)), val_m1, val_mae5])
        log_f.flush()
        print('== epoch %d train_loss %.4f val_metric1 %.3f val_mae5 %.3f (best metric1 %.3f)'
              % (epoch + 1, np.mean(losses), val_m1, val_mae5, best), flush=True)
if rank0:
    log_f.close()
    print('saved:', out_dir)
