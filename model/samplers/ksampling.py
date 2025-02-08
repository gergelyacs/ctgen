# based on https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/external.py

import torch
from torch import nn
from .sample_utils import append_dims, append_zero
from .ksamplers import *

KSAMPLERS = { 
             'k_lms': sample_lms, 
             'k_euler': sample_euler, 
             'k_heun': sample_heun,
             'k_dpmpp_2m' : sample_dpmpp_2m,
             'k_dpmpp_3m_sde' : sample_dpmpp_3m_sde,
             'k_dpmpp_2m_sde' : sample_dpmpp_2m_sde,
             'k_dpmpp_sde': sample_dpmpp_sde,
             'k_dpmpp_2d_ancestral' : sample_dpmpp_2s_ancestral,
             'k_dpm_2_ancestral': sample_dpm_2_ancestral,
             'k_dpm_2': sample_dpm_2,
             'k_euler_ancestral': sample_euler_ancestral
             }

class DiscreteSchedule(nn.Module):
    """A mapping between continuous noise levels (sigmas) and a list of discrete noise
    levels."""

    def __init__(self, sigmas, quantize):
        super().__init__()
        self.register_buffer('sigmas', sigmas)
        self.register_buffer('log_sigmas', sigmas.log())
        self.quantize = quantize

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def get_sigmas(self, n=None):
        if n is None:
            return append_zero(self.sigmas.flip(0))
        t_max = len(self.sigmas) - 1
        t = torch.linspace(t_max, 0, n, device=self.sigmas.device)
        return append_zero(self.t_to_sigma(t))

    def sigma_to_t(self, sigma, quantize=None):
        quantize = self.quantize if quantize is None else quantize
        log_sigma = sigma.log()
        dists = log_sigma - self.log_sigmas[:, None]
        if quantize:
            return dists.abs().argmin(dim=0).view(sigma.shape)
        low_idx = dists.ge(0).cumsum(dim=0).argmax(dim=0).clamp(max=self.log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1
        low, high = self.log_sigmas[low_idx], self.log_sigmas[high_idx]
        w = (low - log_sigma) / (low - high)
        w = w.clamp(0, 1)
        t = (1 - w) * low_idx + w * high_idx
        return t.view(sigma.shape)

    def t_to_sigma(self, t):
        t = t.float()
        low_idx, high_idx, w = t.floor().long(), t.ceil().long(), t.frac()
        log_sigma = (1 - w) * self.log_sigmas[low_idx] + w * self.log_sigmas[high_idx]
        return log_sigma.exp()


class DiscreteEpsDDPMDenoiser(DiscreteSchedule):
    """A wrapper for discrete schedule DDPM models that output eps (the predicted
    noise)."""

    def __init__(self, model, alphas_cumprod, quantize, clip_denoised=False):
        super().__init__(((1 - alphas_cumprod) / alphas_cumprod) ** 0.5, quantize)
        self.inner_model = model
        self.sigma_data = 1.
        self.clip_denoised = clip_denoised
        self.cond = torch.tensor([])

    def get_scalings(self, sigma):
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_out, c_in

    def get_eps(self, *args, **kwargs):
        return self.inner_model(*args, **kwargs)

    def forward(self, input, sigma, **kwargs):
        c_out, c_in = [append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        inp = torch.cat([self.cond, input * c_in], dim=1)
        eps = self.get_eps(inp, time=self.sigma_to_t(sigma), **kwargs)
        ### IMPORTANT: clipping!
        if self.clip_denoised:
            return (input + eps * c_out).clamp(-1., 1.)
        else:
            return input + eps * c_out


class KSampler:
    def __init__(self, model, sampler='lms', clip_denoised=True, pbar=True):
        self.denoiser = DiscreteEpsDDPMDenoiser(model.apply_model, alphas_cumprod=model.alphas_cumprod, clip_denoised=clip_denoised, quantize=False)                                  
        self.sigmas = self.denoiser.get_sigmas(n=model.sampling_timesteps)

        if sampler not in KSAMPLERS:
            raise ValueError(f"Sampler {sampler} not found. Available samplers: {list(KSAMPLERS.keys())}")   
        self.sampler = sampler
        self.sample_fn = KSAMPLERS[sampler.lower()]
        self.pbar = pbar

    @staticmethod
    def get_samplers():
        return list(KSAMPLERS.keys())

    def sample(self, imgs, cond):
        imgs *= self.sigmas[0]
        self.denoiser.cond = cond
        return self.sample_fn(model=self.denoiser, x=imgs, sigmas=self.sigmas, disable=not self.pbar)



 
     