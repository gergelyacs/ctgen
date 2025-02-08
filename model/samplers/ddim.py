import torch
from tqdm import tqdm
from .ddpm import DDPMSampler

class DDIMSampler:
    # https://arxiv.org/abs/2010.02502

    def __init__(self, model, clip_denoised=True, pbar = True):
        self.model = model
        self.num_timesteps = model.num_timesteps
        self.ddim_sampling_eta = model.ddim_sampling_eta
        self.sampling_timesteps = model.sampling_timesteps
        self.clip_denoised = clip_denoised
        self.device = model.device
        self.ddpm = DDPMSampler(model, clip_denoised)
        self.pbar = pbar

    def sample(self, imgs, condition_x):
        times = torch.linspace(-1, self.num_timesteps - 1, steps=self.sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        sample_num = imgs.shape[0]

        for t, t_next in tqdm(time_pairs, desc = 'DDIM Sampler', disable = not self.pbar):
            time_cond = torch.full((sample_num,), t, device = self.device, dtype = torch.long)

            x = torch.cat([condition_x, imgs], dim=1)        
            pred_noise = self.model.apply_model(x, time_cond)

            x_start = self.ddpm.predict_start(imgs, t=t, noise=pred_noise)

            if self.clip_denoised:
                x_start.clamp_(-1., 1.)

            if t_next < 0:
                imgs = x_start
                continue

            alpha = self.model.alphas_cumprod[t]
            alpha_next = self.model.alphas_cumprod[t_next]

            sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(imgs)

            imgs = x_start * alpha_next.sqrt() + \
                    c * pred_noise + \
                    sigma * noise

        return imgs