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

from utils import extract, exists, hash_tensor
from einops import reduce

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
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('pred_coef1', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('pred_coef2', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))

        # Coefficient for reverse diffusion posterior q(x_{t-1} | x_t, x_0)
        variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('variance', to_torch(variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
         # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        self.register_buffer('p2_loss_weight', to_torch((p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma))
                                                                                     
    # Predict desired image x_0 from x_t with noise z_t -> Output is predicted x_0
    def predict_start(self, x_t, t, noise):
        return self.pred_coef1[t] * x_t - self.pred_coef2[t] * noise

    # Compute mean and log variance of posterior(reverse diffusion process) distribution
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    # Note that posterior q for reverse diffusion process is conditioned Gaussian distribution q(x_{t-1}|x_t, x_0)
    # Thus to compute desired posterior q, we need original image x_0 in ideal, 
    # but it's impossible for actual training procedure -> Thus we reconstruct desired x_0 and use this for posterior
    def p_mean_variance(self, x, t, classes, cond_scale,  clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.full((batch_size,), t, device = x.device, dtype = torch.long)
        #noise_level = torch.FloatTensor([t]).repeat(batch_size, 1).to(x.device)
        if exists(classes):
            #print ("BA:", x.shape, noise_level.shape, classes.shape)
            pred_noise = self.model.forward_with_cond_scale(torch.cat([condition_x, x], dim=1), time=noise_level, classes=classes, cond_scale=cond_scale) 
        else:
            #print ("BB:", x.shape, noise_level.shape, classes.shape)
            pred_noise = self.model(torch.cat([condition_x, x], dim=1), time=noise_level)

        x_recon = self.predict_start(x, t, noise=pred_noise)

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        mean, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return mean, posterior_log_variance

    # Progress single step of reverse diffusion process
    # Given mean and log variance of posterior, sample reverse diffusion result from the posterior
    @torch.no_grad()
    def p_sample(self, x, t, classes, condition_x, cond_scale=1., clip_denoised=True, seeds=None):
        mean, log_variance = self.p_mean_variance(x=x, t=t, classes=classes, cond_scale=cond_scale, clip_denoised=clip_denoised, condition_x=condition_x)
        #if exists(seeds):
        #    rn = []
        #    for sample, seed in zip(x, seeds):
        #        torch.manual_seed(seed + t)
        #        rn.append(torch.randn_like(sample).unsqueeze(0))

        #    noise = torch.cat(rn).to(x.device) if t > 0 else torch.zeros_like(x)
        #else:
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return mean + noise * (0.5 * log_variance).exp()
    
    # Compute loss to train the model
    def p_losses(self, x_start, condition_x = None, classes=None, seeds=None):
        # Different starting time for each sample in the batch
        t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device).long()
      
        # generate random noise for each sample in the batch
        if exists(seeds):
            rn = []
            for sample, seed in zip(x_start, seeds):
                torch.manual_seed(seed)
                rn.append(torch.randn_like(sample).unsqueeze(0))

            noise = torch.cat(rn).to(x_start.device)
        else:
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

    def forward(self, x, seeds = None, condition_x = None, classes=None):
        # scale to [-1, 1] (input is supposed to be in [0, 1])
        #x = normalize_to_neg_one_to_one(x)

        #if exists(condition_x):
            #condition_x = normalize_to_neg_one_to_one(condition_x)
            #condition_x = normalize_to_neg_one_to_one(condition_x)
        #else:
        if not exists(condition_x):
            condition_x = torch.tensor([], device=self.device)

        return self.p_losses(x, condition_x = condition_x, classes=classes, seeds=seeds)
    
    @torch.no_grad()
    def generate(self, data_size, **kwargs):
        if 'classes' in kwargs:
            classes = kwargs['classes']
            cond_scale = kwargs['cond_scale']
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

        # duplicate random noise for each sample in the batch
        imgs = torch.randn((sample_num, self.out_channels, data_size), device=self.device).repeat(2,1,1)
       
        seeds = [hash_tensor(img) for img in imgs] 
        # labels: 0 for the first half of the batch, 1 for the other half
        #classes = torch.cat([torch.zeros(sample_num), torch.ones(sample_num)]).long().to(self.device)    
        classes = torch.cat([torch.zeros(sample_num), torch.zeros(sample_num)]).long().to(self.device)    

        cond_scale = 3.
 
        # if classes exists in kwargs, pass it to generate_cond
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            imgs = self.p_sample(imgs, i, classes=classes, condition_x=condition_x, cond_scale=cond_scale, seeds=seeds)

        #print (sample_num)
        #print (imgs[0, :, :64])
        #print (imgs[sample_num, :, :64])
        # l2 norm
        print("Average difference:", torch.abs(imgs[:sample_num, :, :] - imgs[sample_num:, :, :]).sum()) 
      
        #return unnormalize_to_zero_to_one(imgs)
        return imgs
    
