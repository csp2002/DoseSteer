"""Deterministic DDIM sampler (eta=0) for MD-Dose's epsilon-prediction DDPM.

Timesteps range(0, T, T//N), e.g. N=4 -> [0, 250, 500, 750]. x0 is clamped to [0, 1]
each step and eps is recomputed from the clamped x0 before the update.
"""

import torch
import torch.nn as nn


class DDIMSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, ddim_steps, struc=True):
        super().__init__()
        self.model = model
        self.struc = struc
        betas = torch.linspace(beta_1, beta_T, T).double()
        self.register_buffer('alphas_bar', torch.cumprod(1. - betas, dim=0))
        # evenly spaced rounded subsequence; equals range(0, T, T//N) when N | T
        self.seq = sorted({int(round(i * T / ddim_steps)) for i in range(ddim_steps)})
        assert len(self.seq) == ddim_steps and self.seq[-1] < T

    def _eps(self, x, condition, t):
        if self.struc:
            return self.model(x, condition, t)
        return self.model(torch.cat([x, condition], dim=1), t)

    @torch.no_grad()
    def forward(self, x_T, condition):
        x = x_T
        prev_seq = [-1] + self.seq[:-1]
        x0 = None
        for t_cur, t_prev in zip(reversed(self.seq), reversed(prev_seq)):
            t = x.new_full((x.size(0),), t_cur, dtype=torch.long)
            eps = self._eps(x, condition, t)
            a_t = self.alphas_bar[t_cur].to(x.dtype)
            a_prev = self.alphas_bar[t_prev].to(x.dtype) if t_prev >= 0 else x.new_tensor(1.0)
            x0 = ((x - (1. - a_t).sqrt() * eps) / a_t.sqrt()).clamp(0., 1.)
            eps = (x - a_t.sqrt() * x0) / (1. - a_t).sqrt()
            x = a_prev.sqrt() * x0 + (1. - a_prev).sqrt() * eps
        return x0
