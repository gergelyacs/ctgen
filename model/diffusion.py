"""
    Define Diffusion process framework to train desired model:
    Forward Diffusion process:
        Given original image x_0, apply Gaussian noise ε_t for each time step t
        After proper length of time step, image x_T reachs to pure Gaussian noise
    Objective of model f :
        model f is trained to predict actual added noise ε_t for each time step t
"""

import torch
from torch import nn
from tqdm import tqdm
from functools import partial
import numpy as np
import math
from model.samplers import DDIMSampler, PLMSSampler, DPMSolverSampler, KSampler, DDPMSampler

from utils import extract, exists
from einops import reduce
from functools import partial

import logging

logger = logging.getLogger(__name__)

SAMPLERS = {
    'ddim': DDIMSampler,
    'plms': PLMSSampler,
    'dpm_solver': DPMSolverSampler,
    'ddpm': DDPMSampler
}

SAMPLERS.update({k: partial(KSampler, sampler=k) for k in KSampler.get_samplers()})

class Diffusion(nn.Module):
    def __init__(self, model, device):
        super().__init__()
        self.out_channels = model.out_channel
        self.model = model.to(device)
        self.device = device
     
    def set_loss(self, loss_type):
        if loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='none')
        elif self.loss_function == "huber":
            self.loss_func = nn.SmoothL1Loss(reduction='none')
        else:
            raise NotImplementedError()

    def make_beta_schedule(self, schedule, n_timestep, linear_start=1e-4, linear_end=2e-2):
        if schedule == 'linear':
            betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
        elif schedule == 'warmup':
            warmup_frac=0.1
            betas = linear_end * np.ones(n_timestep, dtype=np.float64)
            warmup_time = int(n_timestep * warmup_frac)
            betas[:warmup_time] = np.linspace(linear_start, linear_end, warmup_time, dtype=np.float64)
        elif schedule == "cosine":
            cosine_s = 8e-3
            timesteps = torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)
        else:
            raise NotImplementedError(schedule)
        return betas

    def set_new_noise_schedule(self, schedule_opt,  p2_loss_weight_gamma = 0., p2_loss_weight_k = 1):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=self.device)

        betas = self.make_beta_schedule(**schedule_opt)
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.num_timesteps = int(len(betas))
        # Coefficient for forward diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('alphas', to_torch(alphas))
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))

         # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        self.register_buffer('p2_loss_weight', to_torch((p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma))
                                                                                    
    # Compute loss to train the model
    def p_losses(self, x_start, condition_x = None, classes=None):
        # Different starting time for each sample in the batch
        t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device).long()
      
        noise = torch.randn_like(x_start).to(x_start.device)

        # Perturbed image obtained by forward diffusion process at random time step t 
        # we select batch size elements from the time tensor (Each training sample has different time step)
        x_noisy = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

        # The model predict actual noise added at time step t
        if exists(classes):
            pred_noise = self.model(torch.cat([condition_x, x_noisy], dim=1).float(), time=t, classes=classes)
        else:
            pred_noise = self.model(torch.cat([condition_x, x_noisy], dim=1).float(), time=t)
    
        loss = self.loss_func(noise, pred_noise)  
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.p2_loss_weight, t, loss.shape)

        # average over batch
        return loss.mean()
        #return loss

    def forward(self, x, condition_x = None, classes=None):
        # scale to [-1, 1] (input is supposed to be in [0, 1])
        #x = normalize_to_neg_one_to_one(x)

        #if exists(condition_x):
            #condition_x = normalize_to_neg_one_to_one(condition_x)
            #condition_x = normalize_to_neg_one_to_one(condition_x)
        #else:
        if not exists(condition_x):
            condition_x = torch.tensor([], device=self.device)

        return self.p_losses(x, condition_x = condition_x, classes=classes)
        
    @torch.no_grad()
    def generate(self, data_size, **kwargs):
        if 'classes' in kwargs and exists(kwargs['classes']):
            classes = kwargs['classes']
            cond_scale = kwargs['cond_scale'] if 'cond_scale' in kwargs else 1.0
            sample_num = classes.shape[0]
        else:
            assert "sample_num" in kwargs, "Error: sample_num is not in kwargs"
            classes = None 
            sample_num = kwargs['sample_num']
            cond_scale = None

        if 'condition_x' in kwargs: 
            condition_x = kwargs['condition_x']
            # check if condition_x has sample_num samples
            assert condition_x.shape[0] == sample_num
            # scale condition_x to [-1, 1] (supposed to be in [0, 1])
            # NOTE: this does not distort the input, as we scale and shift with constants
            #condition_x = normalize_to_neg_one_to_one(condition_x)
        else:
            condition_x = torch.tensor([], device=self.device)

        if 'steps' in kwargs:
            self.sampling_timesteps = kwargs['steps'] 
            if 'ddim_sampling_eta' in kwargs:
                self.ddim_sampling_eta = kwargs['ddim_sampling_eta']
            else:
                self.ddim_sampling_eta = 0.0

            self.sampling_timesteps = min(self.sampling_timesteps, self.num_timesteps)
        else:
            self.sampling_timesteps = self.num_timesteps

        if exists(classes):
            self.apply_model = partial(self.model.forward_with_cond_scale, classes=classes, cond_scale=cond_scale)
        else:
            self.apply_model = self.model

        pbar = kwargs['pbar'] if 'pbar' in kwargs else True

        if 'sampler' in kwargs:
            sampler_name = kwargs['sampler'].lower()
            if sampler_name not in SAMPLERS:
                raise ValueError(f"Sampler {sampler_name} not found. Available samplers: {list(SAMPLERS.keys())}")

            logger.info ("Sampler: %s", sampler_name)
            sampler = SAMPLERS[kwargs['sampler'].lower()](model=self, pbar=pbar)
        else:
            # default sampler is DDPM
            sampler = DDPMSampler(model=self, pbar=pbar)

        # DDPM supports only the full sampling
        # is instance of DDPMSampler
        if isinstance(sampler, DDPMSampler):
            self.sampling_timesteps = self.num_timesteps 
            sampler.sampling_timesteps = self.sampling_timesteps
        
        logger.info ("sampling_timesteps: %s", self.sampling_timesteps)
        logger.info ("num_timesteps: %s", self.num_timesteps)

        imgs = torch.randn((sample_num, self.out_channels, data_size), device=self.device)
        return sampler.sample(imgs, condition_x)

   
    
