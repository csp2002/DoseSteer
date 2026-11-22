"""DoseDiT-S/4, called like DoseDiff's UNetModel_MS_Former:
    forward(x_t [B,1,128,128], timesteps, ct [B,1,128,128], dis [B,20,128,128]) -> eps [B,1,128,128]
"""
from __future__ import annotations

import math
import os
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# guided_diffusion is shared with DoseDiff (models/dosediff)
_DOSEDIFF = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dosediff"))
if _DOSEDIFF not in sys.path:
    sys.path.insert(0, _DOSEDIFF)
from guided_diffusion.vit import Transformer as DoseDiffFusionTransformer

# Prefer an importable dinov3 package, else the vendored subset in third_party/dinov3
try:
    from dinov3.hub.backbones import _make_dinov3_vit
except ImportError:
    _root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "third_party", "dinov3"))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from dinov3.hub.backbones import _make_dinov3_vit

IMG_SIZE = (128, 128)
IN_CHANNELS = 1
CT_CHANNELS = 1
DIS_CHANNELS = 20
DIS_CHANNEL_ORDER = (
    "PTVs_mask",
    "Mask_Brainstem", "Mask_Esophagus", "Mask_Larynx", "Mask_LeftParotid", "Mask_Mandible",
    "Mask_possible_dose_mask", "Mask_RightParotid", "Mask_SpinalCord",
    "PSDM_Brainstem", "PSDM_Esophagus", "PSDM_Larynx", "PSDM_LeftParotid", "PSDM_Mandible",
    "PSDM_possible_dose_mask", "PSDM_PTV56", "PSDM_PTV63", "PSDM_PTV70",
    "PSDM_RightParotid", "PSDM_SpinalCord",
)
assert len(DIS_CHANNEL_ORDER) == DIS_CHANNELS

DINOV3_VITS16_KWARGS = dict(
    pos_embed_rope_base=100, pos_embed_rope_normalize_coords="separate", pos_embed_rope_rescale_coords=2,
    pos_embed_rope_dtype="fp32", embed_dim=384, depth=12, num_heads=6, ffn_ratio=4, qkv_bias=True,
    drop_path_rate=0.0, layerscale_init=1.0e-05, norm_layer="layernormbf16", ffn_layer="mlp", ffn_bias=True,
    proj_bias=True, n_storage_tokens=4, mask_k_bias=True, compact_arch_name="vits",
)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def rope_2d(h, w, head_dim, device, base=10000.0):
    quarter = head_dim // 4
    freqs = 1.0 / base ** (torch.arange(quarter, device=device).float() / quarter)
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device).float(),
        torch.arange(w, device=device).float(),
        indexing="ij",
    )
    ang = torch.cat([ys.flatten()[:, None] * freqs, xs.flatten()[:, None] * freqs], dim=-1)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class CTEncoder(nn.Module):
    def __init__(self, grid, taps):
        super().__init__()
        self.vit = _make_dinov3_vit(patch_size=16, in_chans=3, pretrained=False, **DINOV3_VITS16_KWARGS)
        self.input_size = grid * self.vit.patch_size
        self.taps = list(taps)
        self.dim = self.vit.embed_dim
        self.register_buffer("rgb_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("rgb_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, ct):
        x = ((ct + 1.0) / 2.0).clamp(0, 1).repeat(1, 3, 1, 1)
        x = (x - self.rgb_mean) / self.rgb_std
        x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)
        return list(self.vit.get_intermediate_layers(x, n=self.taps, norm=True))


class SDMEncoder(nn.Module):
    def __init__(self, img_size, grid, taps, depth=12):
        super().__init__()
        kwargs = dict(DINOV3_VITS16_KWARGS, depth=depth)
        self.vit = _make_dinov3_vit(
            patch_size=img_size // grid, in_chans=DIS_CHANNELS, pretrained=False, **kwargs)
        self.taps = list(taps)
        self.dim = self.vit.embed_dim

    def load_dinov3_blocks(self, state_dict):
        sd = {k: v for k, v in state_dict.items() if not k.startswith("patch_embed.")}
        return self.vit.load_state_dict(sd, strict=False)

    def forward(self, dis):
        return list(self.vit.get_intermediate_layers(dis, n=self.taps, norm=True))


class TimestepEmbedder(nn.Module):
    def __init__(self, dim, freq_dim=256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
        args = t.float()[:, None] * freqs[None]
        return self.mlp(torch.cat([torch.cos(args), torch.sin(args)], dim=-1))


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.w12 = nn.Linear(dim, 2 * hidden)
        self.w3 = nn.Linear(hidden, dim)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a) * b)


class DiTBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, skip=False):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.mlp = SwiGLU(dim, int(2 / 3 * mlp_ratio * dim))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, c, rope, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN(c).chunk(6, dim=-1)
        B, N, D = x.shape
        qkv = self.qkv(modulate(self.norm1(x), shift1, scale1)).view(B, N, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        q, k = apply_rope(self.q_norm(q), *rope), apply_rope(self.k_norm(k), *rope)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, D)
        x = x + gate1.unsqueeze(1) * self.proj(a)
        return x + gate2.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift2, scale2))


class FusionFormer(nn.Module):
    def __init__(self, dim_ct, dim_sdm, dim_v, num_tokens, dim=384, heads=6, dim_head=64, mlp_dim=768):
        super().__init__()
        def embed(d_in):
            return nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, dim), nn.LayerNorm(dim))
        self.embed_q, self.embed_k, self.embed_v = embed(dim_ct), embed(dim_sdm), embed(dim_v)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, dim))
        self.transformer = DoseDiffFusionTransformer(dim, heads, dim_head, mlp_dim)
        self.return_linear = nn.Linear(dim, dim_v)

    def forward(self, ct, sdm, v):
        q = self.embed_q(ct) + self.pos_embedding
        k = self.embed_k(sdm) + self.pos_embedding
        v = self.embed_v(v) + self.pos_embedding
        return self.return_linear(self.transformer(q, k, v))


class DoseDiT(nn.Module):
    """DoseDiT-S/4 (91.3 M parameters)."""

    def __init__(
        self,
        image_size=IMG_SIZE,
        in_channels=IN_CHANNELS,
        ct_channels=CT_CHANNELS,
        dis_channels=DIS_CHANNELS,
        grid=32,
        dim=384,
        heads=6,
        dose_depth=4,
        dit_depth=12,
        taps=(2, 5, 8, 11),
        sdm_depth=12,
        ms_fusion=True,
        uvit_skip=True,
        out_conv=True,
    ):
        super().__init__()
        if isinstance(image_size, (tuple, list)):
            img_size = int(image_size[0])
        else:
            img_size = int(image_size)
        assert (in_channels, ct_channels, dis_channels) == (IN_CHANNELS, CT_CHANNELS, DIS_CHANNELS)
        assert img_size % grid == 0 and dose_depth == len(taps)
        self.image_size = (img_size, img_size)
        self.in_channels = in_channels
        self.grid, self.patch = grid, img_size // grid
        self.ms_fusion, self.uvit_skip = ms_fusion, uvit_skip
        self.ct_enc = CTEncoder(grid, taps)
        self.sdm_enc = SDMEncoder(img_size, grid, taps, depth=sdm_depth)
        self.t_embedder = TimestepEmbedder(dim)
        self.dose_embed = nn.Conv2d(in_channels, dim, kernel_size=self.patch, stride=self.patch)
        self.dose_blocks = nn.ModuleList([DiTBlock(dim, heads) for _ in range(dose_depth)])
        if ms_fusion:
            self.ms_ct = nn.ModuleList([nn.Linear(self.ct_enc.dim, dim) for _ in taps])
            self.ms_sdm = nn.ModuleList([nn.Linear(self.sdm_enc.dim, dim) for _ in taps])
        self.fusion = FusionFormer(self.ct_enc.dim, self.sdm_enc.dim, dim, grid * grid, dim=dim, heads=heads)
        half = dit_depth // 2
        self.dit_blocks = nn.ModuleList(
            [DiTBlock(dim, heads, skip=uvit_skip and i >= half) for i in range(dit_depth)])
        self.final_norm = nn.RMSNorm(dim, eps=1e-6)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.final_linear = nn.Linear(dim, self.patch * self.patch * in_channels)
        for m in (self.final_adaLN[-1], self.final_linear):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        self.out_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1) if out_conv else nn.Identity()

    def load_dinov3_weights(self, ckpt_path: Optional[str]):
        """Load DINOv3-S/16 into CT encoder; copy transformer blocks into SDM (skip patch_embed)."""
        if not ckpt_path:
            print("[DoseDiT] no dinov3 ckpt; training CT/SDM from scratch")
            return
        print(f"[DoseDiT] loading DINOv3 weights from {ckpt_path}")
        obj = torch.load(ckpt_path, map_location="cpu")
        if isinstance(obj, dict):
            if "teacher" in obj:
                sd = {k.replace("backbone.", ""): v for k, v in obj["teacher"].items()
                      if "ibot" not in k and "dino_head" not in k}
            elif "model" in obj:
                sd = obj["model"]
            elif "state_dict" in obj:
                sd = obj["state_dict"]
            else:
                sd = obj
        else:
            raise ValueError(f"Unrecognized checkpoint format: {type(obj)}")
        miss, unexp = self.ct_enc.vit.load_state_dict(sd, strict=False)
        print(f"[DoseDiT] CT encoder load: missing={len(miss)} unexpected={len(unexp)}")
        miss2, unexp2 = self.sdm_enc.load_dinov3_blocks(sd)
        print(f"[DoseDiT] SDM encoder load (no patch_embed): missing={len(miss2)} unexpected={len(unexp2)}")

    def encoder_param_ids(self):
        ids = set(map(id, self.ct_enc.parameters()))
        ids.update(map(id, self.sdm_enc.parameters()))
        return ids

    def forward(self, x, timesteps, ct, dis, y=None):
        assert y is None, "DoseDiff OpenKBP path is not class-conditional"
        ct_feats = self.ct_enc(ct)
        sdm_feats = self.sdm_enc(dis)
        c = self.t_embedder(timesteps)
        rope = rope_2d(self.grid, self.grid, self.dose_blocks[0].head_dim, x.device)

        h = self.dose_embed(x).flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dose_blocks):
            h = blk(h, c, rope)
            if self.ms_fusion:
                h = h + self.ms_ct[i](ct_feats[i]) + self.ms_sdm[i](sdm_feats[i])

        h = self.fusion(ct_feats[-1], sdm_feats[-1], h) + h

        skips, half = [], len(self.dit_blocks) // 2
        for i, blk in enumerate(self.dit_blocks):
            h = blk(h, c, rope, skip=skips.pop() if (self.uvit_skip and i >= half) else None)
            if self.uvit_skip and i < half:
                skips.append(h)

        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        h = self.final_linear(modulate(self.final_norm(h), shift, scale))
        B, p, g = h.shape[0], self.patch, self.grid
        h = h.view(B, g, g, p, p, IN_CHANNELS).permute(0, 5, 1, 3, 2, 4).reshape(
            B, IN_CHANNELS, g * p, g * p)
        return self.out_conv(h)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
