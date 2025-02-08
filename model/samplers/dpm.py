# based on: https://github.com/Stability-AI/stablediffusion/blob/main/ldm/models/diffusion/dpm_solver/sampler.py

import torch

from .dpm_solver import DPM_Solver
from .sample_utils import NoiseScheduleVP, model_wrapper

class DPMSolverSampler(object):
    def __init__(self, model, pbar=True, **kwargs):
        super().__init__()
        self.model = model
        self.device = model.device
        self.pbar = pbar
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(model.device)
        self.register_buffer('alphas_cumprod', to_torch(model.alphas_cumprod))

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != self.device:
                attr = attr.to(self.device)
        setattr(self, name, attr)

    @torch.no_grad()
    def sample(self,
               img,
               cond,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        # sampling
     
        device = self.model.betas.device
     
        ns = NoiseScheduleVP('discrete', alphas_cumprod=self.alphas_cumprod)

        model_fn = model_wrapper(
            lambda x, t: self.model.apply_model(torch.cat([cond, x], dim=1), t),
            ns
        )

        '''
        Some advices for choosing the algorithm:
            - For **unconditional sampling** or **guided sampling with small guidance scale** by DPMs:
                Use singlestep DPM-Solver ("DPM-Solver-fast" in the paper) with `order = 3`.
                e.g.
                    >>> dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=False)
                    >>> x_sample = dpm_solver.sample(x, steps=steps, t_start=t_start, t_end=t_end, order=3,
                            skip_type='time_uniform', method='singlestep')
            - For **guided sampling with large guidance scale** by DPMs:
                Use multistep DPM-Solver with `predict_x0 = True` and `order = 2`.
                e.g.
                    >>> dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True)
                    >>> x_sample = dpm_solver.sample(x, steps=steps, t_start=t_start, t_end=t_end, order=2,
                            skip_type='time_uniform', method='multistep')
        We support three types of `skip_type`:
            - 'logSNR': uniform logSNR for the time steps. **Recommended for low-resolutional images**
            - 'time_uniform': uniform time for the time steps. **Recommended for high-resolutional images**.
            - 'time_quadratic': quadratic time for the time steps.
        '''

        dpm_solver = DPM_Solver(model_fn, ns, predict_x0=True, thresholding=False, pbar=self.pbar)
        x = dpm_solver.sample(img, steps=self.model.sampling_timesteps, 
                              skip_type="time_uniform", 
                              method="multistep", 
                              order=2,
                              lower_order_final=True)

        return x.to(device)

