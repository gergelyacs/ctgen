import torch
from torch import nn
from tqdm import tqdm
from functools import partial
import numpy as np

# Original DDPM Sampler 
class DDPMSampler(object):
    def __init__(self, model, clip_denoised=True, pbar = True):
        super().__init__()

        self.device = model.device
        self.clip_denoised = clip_denoised
        self.denoiser = model
        self.sampling_timesteps = model.sampling_timesteps
        self.pbar = pbar

        self.register_buffer('posterior_mean_coef1', model.betas * torch.sqrt(model.alphas_cumprod_prev) / (1. - model.alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1. - model.alphas_cumprod_prev) * torch.sqrt(model.alphas) / (1. - model.alphas_cumprod))
     
        self.register_buffer('pred_coef1', torch.sqrt(1. / model.alphas_cumprod))
        self.register_buffer('pred_coef2', torch.sqrt(1. / model.alphas_cumprod - 1))
        # Coefficient for reverse diffusion posterior q(x_{t-1} | x_t, x_0)
        variance = model.betas * (1. - model.alphas_cumprod_prev) / (1. - model.alphas_cumprod)
        self.register_buffer('variance', variance)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', torch.log(variance.clamp(min=1e-20)))
    
    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != self.device:
                attr = attr.to(self.device)
        setattr(self, name, attr)

    # Predict desired image x_0 from x_t with noise z_t -> Output is predicted x_0
    def predict_start(self, x_t, t, noise):
        return self.pred_coef1[t] * x_t - self.pred_coef2[t] * noise
        #return extract(self.pred_coef1, t, x_t.shape) * x_t - \
        #    extract(self.pred_coef2, t, x_t.shape) * noise

    # Compute mean and log variance of posterior(reverse diffusion process) distribution
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    # Note that posterior q for reverse diffusion process is conditioned Gaussian distribution q(x_{t-1}|x_t, x_0)
    # Thus to compute desired posterior q, we need original image x_0 in ideal, 
    # but it's impossible for actual training procedure -> Thus we reconstruct desired x_0 and use this for posterior
    def p_mean_variance(self, x, t):
        batch_size = x.shape[0]
        noise_level = torch.full((batch_size,), t, device = x.device, dtype = torch.long)
        #noise_level = torch.FloatTensor([t]).repeat(batch_size, 1).to(x.device)
        pred_noise = self.model(x, noise_level)
    
        x_recon = self.predict_start(x, t, noise=pred_noise)
        #x_recon = self.predict_start(x, t=noise_level, noise=pred_noise)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)

        mean, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return mean, posterior_log_variance

    @torch.no_grad()
    def ddpm_sample(self, x):
        # duplicate random noise for each sample in the batch
        #imgs = torch.randn((sample_num, self.out_channels, data_size), device=self.device) 

        # if classes exists in kwargs, pass it to generate_cond
        for i in tqdm(reversed(range(0, self.sampling_timesteps)), desc = 'DDPM Sampler', total = self.sampling_timesteps, disable=not self.pbar):
            # Progress single step of reverse diffusion process
            # Given mean and log variance of posterior, sample reverse diffusion result from the posterior
            mean, log_variance = self.p_mean_variance(x=x, t=i)
            noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
            x = mean + noise * (0.5 * log_variance).exp()
        
        return x
     
    def sample(self, imgs, cond):
        self.model = lambda x, t: self.denoiser.apply_model(torch.cat([cond, x], dim=1).float(), t) 
        return self.ddpm_sample(x=imgs)
