"""Train DoseDiT-S/4 on OpenKBP."""
from __future__ import annotations

import argparse
import copy
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
import torch.utils.data as Data

sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dosediff")))  # dataset.py, guided_diffusion
from dataset import Dataset_PSDM_train, Dataset_PSDM_val
from dosedit_model import DoseDiT, count_parameters
from guided_diffusion import gaussian_diffusion as gd
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from tensorboardX import SummaryWriter
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--bs", type=int, default=24, help="per-GPU batch size")
parser.add_argument("--T", type=int, default=1000)
parser.add_argument("--epoch", type=int, default=1600)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--encoder_lr_scale", type=float, default=0.1)
parser.add_argument("--wd", type=float, default=1e-4)
parser.add_argument("--ema", type=float, default=0.9999)
parser.add_argument("--val_every", type=int, default=10)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--dinov3_ckpt", type=str, default="", help="optional DINOv3-S/16 .pth")
parser.add_argument("--data_root_train", type=str, default="data/openkbp/preprocessed/NPY/train")
parser.add_argument("--data_root_val", type=str, default="data/openkbp/preprocessed/NPY/validation")
parser.add_argument("--save_root", type=str, default="runs/dosedit")
parser.add_argument("--resume", type=str, default="")
parser.add_argument("--max_steps", type=int, default=0, help="stop after this many steps (0 = off)")
parser.add_argument("--skip_val", type=int, default=0)
parser.add_argument("--local_rank", default=-1, type=int)
parser.add_argument("--local-rank", dest="local_rank", type=int)  # torch.distributed.launch
args = parser.parse_args()
if "LOCAL_RANK" in os.environ:
    args.local_rank = int(os.environ["LOCAL_RANK"])
if "WORLD_SIZE" in os.environ:
    world_size_env = int(os.environ["WORLD_SIZE"])
else:
    world_size_env = max(1, len(str(args.gpu).split(",")))

# --gpu applies only when CUDA_VISIBLE_DEVICES is unset
if "CUDA_VISIBLE_DEVICES" not in os.environ or os.environ["CUDA_VISIBLE_DEVICES"] == "":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

torch.cuda.set_device(args.local_rank)
dist.init_process_group(backend="nccl", init_method="env://")
device = torch.device("cuda")

train_bs = args.bs
val_bs = 2 * args.bs
lr_max = args.lr
img_size = (128, 128)
all_epochs = args.epoch
true_bs = dist.get_world_size() * args.bs
save_name = "DoseDiT_S4_T{}_bs{}_epoch{}".format(args.T, true_bs, args.epoch)
save_dir = os.path.join(args.save_root, save_name)

if dist.get_rank() == 0:
    os.makedirs(save_dir, exist_ok=True)
    print("save_name:", save_name)
    print("world_size:", dist.get_world_size(), "per_gpu_bs:", train_bs, "true_bs:", true_bs)
    train_writer = SummaryWriter(os.path.join(save_dir, "log/train"), flush_secs=2)

train_data = Dataset_PSDM_train(data_root=args.data_root_train)
train_samper = Data.distributed.DistributedSampler(train_data)
val_data = Dataset_PSDM_val(data_root=args.data_root_val)
val_samper = Data.distributed.DistributedSampler(val_data)
train_dataloader = DataLoader(
    dataset=train_data, batch_size=train_bs, sampler=train_samper, shuffle=False,
    num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0,
    prefetch_factor=4 if args.num_workers > 0 else None,
)
val_dataloader = DataLoader(
    dataset=val_data, batch_size=val_bs, sampler=val_samper, shuffle=False,
    num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0,
    prefetch_factor=4 if args.num_workers > 0 else None,
)

if dist.get_rank() == 0:
    print("train_length: %i   val_length: %i" % (train_data.len, val_data.len))

dis_channels = 20
model = DoseDiT(
    image_size=img_size, in_channels=1, ct_channels=1, dis_channels=dis_channels,
    grid=32, dim=384, heads=6, dose_depth=4, dit_depth=12, taps=(2, 5, 8, 11),
    sdm_depth=12, ms_fusion=True, uvit_skip=True, out_conv=True,
)
if dist.get_rank() == 0:
    nparam = count_parameters(model)
    print(f"DoseDiT params: {nparam:,} ({nparam/1e6:.2f}M)")
    assert nparam < 150_000_000

if args.dinov3_ckpt and os.path.isfile(args.dinov3_ckpt):
    model.load_dinov3_weights(args.dinov3_ckpt)
elif dist.get_rank() == 0:
    print("CT/SDM encoders: random initialization")

diffusion = SpacedDiffusion(
    use_timesteps=space_timesteps(args.T, [args.T]),
    betas=gd.get_named_beta_schedule("linear", args.T),
    model_mean_type=gd.ModelMeanType.EPSILON,
    model_var_type=gd.ModelVarType.FIXED_LARGE,
    loss_type=gd.LossType.MSE, rescale_timesteps=False,
)
diffusion_test = SpacedDiffusion(
    use_timesteps=space_timesteps(args.T, "ddim4"),
    betas=gd.get_named_beta_schedule("linear", args.T),
    model_mean_type=gd.ModelMeanType.EPSILON,
    model_var_type=gd.ModelVarType.FIXED_LARGE,
    loss_type=gd.LossType.MSE, rescale_timesteps=False,
)

model.cuda()
enc_ids = model.encoder_param_ids()
param_groups = [
    {"params": [p for p in model.parameters() if id(p) in enc_ids], "lr": lr_max * args.encoder_lr_scale},
    {"params": [p for p in model.parameters() if id(p) not in enc_ids], "lr": lr_max},
]
ema_model = copy.deepcopy(model)
for p in ema_model.parameters():
    p.requires_grad_(False)
ema_model.eval()

model = DDP(
    model, device_ids=[args.local_rank], output_device=args.local_rank,
    broadcast_buffers=False, bucket_cap_mb=128, find_unused_parameters=False,
)

schedule_sampler = create_named_schedule_sampler("uniform", diffusion)
optimizer = optim.AdamW(param_groups, lr=lr_max, weight_decay=args.wd, betas=(0.9, 0.999))
# bf16 autocast (fp16 where bf16 is unsupported), no GradScaler
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
if dist.get_rank() == 0:
    print(f"AMP dtype: {amp_dtype}")
best_MAE = 1000.0
start_epoch = 0
global_step = 0

if args.resume and os.path.isfile(args.resume):
    ckpt = torch.load(args.resume, map_location="cpu")
    model.module.load_state_dict(ckpt["model"])
    if "ema" in ckpt:
        ema_model.load_state_dict(ckpt["ema"])
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch = int(ckpt.get("epoch", 0))
    best_MAE = float(ckpt.get("best_MAE", best_MAE))
    if dist.get_rank() == 0:
        print(f"resumed from {args.resume} epoch={start_epoch} best_MAE={best_MAE}")


def update_ema(ema, online, decay):
    with torch.no_grad():
        ema_p = dict(ema.named_parameters())
        for name, p in online.named_parameters():
            ema_p[name].mul_(decay).add_(p.data, alpha=1.0 - decay)


for epoch in range(start_epoch, all_epochs):
    train_samper.set_epoch(epoch)
    lr = optimizer.param_groups[1]["lr"]
    model.train()
    train_epoch_loss = []
    t0 = time.time()
    for i, (ct, dis, rtdose) in enumerate(train_dataloader):
        ct = ct.cuda(non_blocking=True).float()
        dis = dis.cuda(non_blocking=True).float()
        rtdose = rtdose.cuda(non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)
        t, weights = schedule_sampler.sample(rtdose.shape[0], rtdose.device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            losses = diffusion.training_losses(
                model=model, x_start=rtdose, t=t,
                model_kwargs={"ct": ct, "dis": dis}, noise=None,
            )
            loss = (losses["loss"] * weights).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        update_ema(ema_model, model.module, args.ema)

        dist.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
        loss = loss / dist.get_world_size()
        train_epoch_loss.append(loss.item())
        global_step += 1
        if dist.get_rank() == 0 and ((i + 1) % 20 == 0 or i == 0):
            print("[%d/%d, %d/%d] train_loss: %.4f" %
                  (epoch + 1, all_epochs, i + 1, len(train_dataloader), loss.item()))
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    if dist.get_rank() == 0:
        train_epoch_loss = float(np.mean(train_epoch_loss)) if train_epoch_loss else 0.0
        sps = len(train_dataloader) * train_bs * dist.get_world_size() / max(time.time() - t0, 1e-6)
        train_writer.add_scalar("lr", lr, epoch + 1)
        train_writer.add_scalar("train_loss", train_epoch_loss, epoch + 1)
        train_writer.add_scalar("samples_per_sec", sps, epoch + 1)
        print(f"epoch {epoch+1} done loss={train_epoch_loss:.4f} samples/s≈{sps:.1f}")

    do_val = (not args.skip_val) and ((epoch == 0) or (((epoch + 1) % args.val_every) == 0))
    if do_val:
        # validate with the EMA weights
        was_training = model.training
        model.eval()
        val_epoch_MAE = []
        image_CT, ture_rtdose, pred_rtdose = [], [], []
        # temporarily swap DDP module weights with EMA for sampling
        online_sd = {k: v.detach().clone() for k, v in model.module.state_dict().items()}
        model.module.load_state_dict(ema_model.state_dict(), strict=True)
        with torch.no_grad():
            for i, (ct, dis, rtdose) in enumerate(val_dataloader):
                ct = ct.cuda(non_blocking=True).float()
                dis = dis.cuda(non_blocking=True).float()
                rtdose = rtdose.cuda(non_blocking=True).float()
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    pred = diffusion_test.ddim_sample_loop(
                        model=model, shape=(ct.size(0), 1, img_size[0], img_size[1]),
                        noise=None, clip_denoised=True, denoised_fn=None, cond_fn=None,
                        model_kwargs={"ct": ct, "dis": dis}, device=None, progress=False, eta=0.0,
                    )
                rtdose_gy = (rtdose + 1) * 40
                pred_gy = (pred.float() + 1) * 40
                body_mask = dis[:, 6:7]
                MAE = (torch.abs(rtdose_gy - pred_gy) * body_mask).sum() / body_mask.sum().clamp_min(1.0)
                dist.all_reduce(MAE, op=torch.distributed.ReduceOp.SUM)
                MAE = MAE / dist.get_world_size()
                val_epoch_MAE.append(MAE.item())
                if i in [0, 2, 4, 8]:
                    image_CT.append(ct[0:1].cpu())
                    ture_rtdose.append(rtdose_gy[0:1].cpu())
                    pred_rtdose.append(pred_gy[0:1].cpu())
        model.module.load_state_dict(online_sd, strict=True)
        if was_training:
            model.train()

        if dist.get_rank() == 0:
            val_epoch_MAE = float(np.mean(val_epoch_MAE))
            train_writer.add_scalar("val_MAE", val_epoch_MAE, epoch + 1)
            print(f"val_MAE={val_epoch_MAE:.4f} best={best_MAE:.4f}")
            if image_CT:
                train_writer.add_image("image_CT", make_grid(torch.cat(image_CT, 0), 2, normalize=True), epoch + 1)
                train_writer.add_image("ture_rtdose", make_grid(torch.cat(ture_rtdose, 0), 2, normalize=True), epoch + 1)
                train_writer.add_image("pred_rtdose", make_grid(torch.cat(pred_rtdose, 0), 2, normalize=True), epoch + 1)
            ckpt = {
                "model": model.module.state_dict(),
                "ema": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "best_MAE": best_MAE,
                "args": vars(args),
            }
            torch.save(ckpt, os.path.join(save_dir, f"model_epoch{epoch + 1}.pth"))
            # EMA weights alone, for pred.py and drg/
            torch.save(ema_model.state_dict(), os.path.join(save_dir, f"ema_epoch{epoch + 1}.pth"))
            if val_epoch_MAE < best_MAE:
                best_MAE = val_epoch_MAE
                ckpt["best_MAE"] = best_MAE
                torch.save(ckpt, os.path.join(save_dir, "model_best_mae.pth"))
                torch.save(ema_model.state_dict(), os.path.join(save_dir, "ema_best_mae.pth"))
                print("saved new best:", best_MAE)

    if args.max_steps > 0 and global_step >= args.max_steps:
        if dist.get_rank() == 0:
            print(f"reached max_steps={args.max_steps}, stopping")
            torch.save(
                {"model": model.module.state_dict(), "ema": ema_model.state_dict(),
                 "epoch": epoch + 1, "best_MAE": best_MAE},
                os.path.join(save_dir, "model_smoke.pth"),
            )
        break

if dist.get_rank() == 0:
    train_writer.close()
    print("saved_model_name:", save_name)
dist.barrier()
