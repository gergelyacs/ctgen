import torch
from tqdm import tqdm
from .sample_utils import *

class DPM_Solver:
    def __init__(self, model_fn, noise_schedule, predict_x0=False, thresholding=False, max_val=1., pbar=True):
        """Construct a DPM-Solver.
        We support both the noise prediction model ("predicting epsilon") and the data prediction model ("predicting x0").
        If `predict_x0` is False, we use the solver for the noise prediction model (DPM-Solver).
        If `predict_x0` is True, we use the solver for the data prediction model (DPM-Solver++).
            In such case, we further support the "dynamic thresholding" in [1] when `thresholding` is True.
            The "dynamic thresholding" can greatly improve the sample quality for pixel-space DPMs with large guidance scales.
        Args:
            model_fn: A noise prediction model function which accepts the continuous-time input (t in [epsilon, T]):
                ``
                def model_fn(x, t_continuous):
                    return noise
                ``
            noise_schedule: A noise schedule object, such as NoiseScheduleVP.
            predict_x0: A `bool`. If true, use the data prediction model; else, use the noise prediction model.
            thresholding: A `bool`. Valid when `predict_x0` is True. Whether to use the "dynamic thresholding" in [1].
            max_val: A `float`. Valid when both `predict_x0` and `thresholding` are True. The max value for thresholding.

        [1] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S Sara Mahdavi, Rapha Gontijo Lopes, et al. Photorealistic text-to-image diffusion models with deep language understanding. arXiv preprint arXiv:2205.11487, 2022b.
        """
        self.model = model_fn
        self.noise_schedule = noise_schedule
        self.predict_x0 = predict_x0
        self.thresholding = thresholding
        self.max_val = max_val
        self.pbar = pbar

    def noise_prediction_fn(self, x, t):
        """
        Return the noise prediction model.
        """
        return self.model(x, t)

    def data_prediction_fn(self, x, t):
        """
        Return the data prediction model (with thresholding).
        """
        noise = self.noise_prediction_fn(x, t)
        dims = x.dim()
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        x0 = (x - expand_dims(sigma_t, dims) * noise) / expand_dims(alpha_t, dims)
        if self.thresholding:
            p = 0.995  # A hyperparameter in the paper of "Imagen" [1].
            s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
            s = expand_dims(torch.maximum(s, self.max_val * torch.ones_like(s).to(s.device)), dims)
            x0 = torch.clamp(x0, -s, s) / s
        return x0

    def model_fn(self, x, t):
        """
        Convert the model to the noise prediction model or the data prediction model.
        """
        if self.predict_x0:
            return self.data_prediction_fn(x, t)
        else:
            return self.noise_prediction_fn(x, t)

    def get_time_steps(self, skip_type, t_T, t_0, N, device):
        """Compute the intermediate time steps for sampling.
        Args:
            skip_type: A `str`. The type for the spacing of the time steps. We support three types:
                - 'logSNR': uniform logSNR for the time steps.
                - 'time_uniform': uniform time for the time steps. (**Recommended for high-resolutional data**.)
                - 'time_quadratic': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            N: A `int`. The total number of the spacing of the time steps.
            device: A torch device.
        Returns:
            A pytorch tensor of the time steps, with the shape (N + 1,).
        """
        if skip_type == 'logSNR':
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
            logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1).to(device)
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == 'time_uniform':
            return torch.linspace(t_T, t_0, N + 1).to(device)
        elif skip_type == 'time_quadratic':
            t_order = 2
            t = torch.linspace(t_T ** (1. / t_order), t_0 ** (1. / t_order), N + 1).pow(t_order).to(device)
            return t
        else:
            raise ValueError(
                "Unsupported skip_type {}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'".format(skip_type))

    def get_orders_and_timesteps_for_singlestep_solver(self, steps, order, skip_type, t_T, t_0, device):
        """
        Get the order of each step for sampling by the singlestep DPM-Solver.
        We combine both DPM-Solver-1,2,3 to use all the function evaluations, which is named as "DPM-Solver-fast".
        Given a fixed number of function evaluations by `steps`, the sampling procedure by DPM-Solver-fast is:
            - If order == 1:
                We take `steps` of DPM-Solver-1 (i.e. DDIM).
            - If order == 2:
                - Denote K = (steps // 2). We take K or (K + 1) intermediate time steps for sampling.
                - If steps % 2 == 0, we use K steps of DPM-Solver-2.
                - If steps % 2 == 1, we use K steps of DPM-Solver-2 and 1 step of DPM-Solver-1.
            - If order == 3:
                - Denote K = (steps // 3 + 1). We take K intermediate time steps for sampling.
                - If steps % 3 == 0, we use (K - 2) steps of DPM-Solver-3, and 1 step of DPM-Solver-2 and 1 step of DPM-Solver-1.
                - If steps % 3 == 1, we use (K - 1) steps of DPM-Solver-3 and 1 step of DPM-Solver-1.
                - If steps % 3 == 2, we use (K - 1) steps of DPM-Solver-3 and 1 step of DPM-Solver-2.
        ============================================
        Args:
            order: A `int`. The max order for the solver (2 or 3).
            steps: A `int`. The total number of function evaluations (NFE).
            skip_type: A `str`. The type for the spacing of the time steps. We support three types:
                - 'logSNR': uniform logSNR for the time steps.
                - 'time_uniform': uniform time for the time steps. (**Recommended for high-resolutional data**.)
                - 'time_quadratic': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            device: A torch device.
        Returns:
            orders: A list of the solver order of each step.
        """
        if order == 3:
            K = steps // 3 + 1
            if steps % 3 == 0:
                orders = [3, ] * (K - 2) + [2, 1]
            elif steps % 3 == 1:
                orders = [3, ] * (K - 1) + [1]
            else:
                orders = [3, ] * (K - 1) + [2]
        elif order == 2:
            if steps % 2 == 0:
                K = steps // 2
                orders = [2, ] * K
            else:
                K = steps // 2 + 1
                orders = [2, ] * (K - 1) + [1]
        elif order == 1:
            K = 1
            orders = [1, ] * steps
        else:
            raise ValueError("'order' must be '1' or '2' or '3'.")
        if skip_type == 'logSNR':
            # To reproduce the results in DPM-Solver paper
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, K, device)
        else:
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, steps, device)[
                torch.cumsum(torch.tensor([0, ] + orders), dim=0).to(device)]
        return timesteps_outer, orders

    def denoise_to_zero_fn(self, x, s):
        """
        Denoise at the final step, which is equivalent to solve the ODE from lambda_s to infty by first-order discretization.
        """
        return self.data_prediction_fn(x, s)

    def dpm_solver_first_update(self, x, s, t, model_s=None, return_intermediate=False):
        """
        DPM-Solver-1 (equivalent to DDIM) from time `s` to time `t`.
        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (x.shape[0],).
            t: A pytorch tensor. The ending time, with the shape (x.shape[0],).
            model_s: A pytorch tensor. The model function evaluated at time `s`.
                If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
            return_intermediate: A `bool`. If true, also return the model value at time `s`.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        ns = self.noise_schedule
        dims = x.dim()
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_t = ns.marginal_std(s), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        if self.predict_x0:
            phi_1 = torch.expm1(-h)
            if model_s is None:
                model_s = self.model_fn(x, s)
            x_t = (
                    expand_dims(sigma_t / sigma_s, dims) * x
                    - expand_dims(alpha_t * phi_1, dims) * model_s
            )
            if return_intermediate:
                return x_t, {'model_s': model_s}
            else:
                return x_t
        else:
            phi_1 = torch.expm1(h)
            if model_s is None:
                model_s = self.model_fn(x, s)
            x_t = (
                    expand_dims(torch.exp(log_alpha_t - log_alpha_s), dims) * x
                    - expand_dims(sigma_t * phi_1, dims) * model_s
            )
            if return_intermediate:
                return x_t, {'model_s': model_s}
            else:
                return x_t

    def singlestep_dpm_solver_second_update(self, x, s, t, r1=0.5, model_s=None, return_intermediate=False,
                                            solver_type='dpm_solver'):
        """
        Singlestep solver DPM-Solver-2 from time `s` to time `t`.
        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (x.shape[0],).
            t: A pytorch tensor. The ending time, with the shape (x.shape[0],).
            r1: A `float`. The hyperparameter of the second-order solver.
            model_s: A pytorch tensor. The model function evaluated at time `s`.
                If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
            return_intermediate: A `bool`. If true, also return the model value at time `s` and `s1` (the intermediate time).
            solver_type: either 'dpm_solver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpm_solver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if solver_type not in ['dpm_solver', 'taylor']:
            raise ValueError("'solver_type' must be either 'dpm_solver' or 'taylor', got {}".format(solver_type))
        if r1 is None:
            r1 = 0.5
        ns = self.noise_schedule
        dims = x.dim()
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        s1 = ns.inverse_lambda(lambda_s1)
        log_alpha_s, log_alpha_s1, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(
            s1), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_s1, sigma_t = ns.marginal_std(s), ns.marginal_std(s1), ns.marginal_std(t)
        alpha_s1, alpha_t = torch.exp(log_alpha_s1), torch.exp(log_alpha_t)

        if self.predict_x0:
            phi_11 = torch.expm1(-r1 * h)
            phi_1 = torch.expm1(-h)

            if model_s is None:
                model_s = self.model_fn(x, s)
            x_s1 = (
                    expand_dims(sigma_s1 / sigma_s, dims) * x
                    - expand_dims(alpha_s1 * phi_11, dims) * model_s
            )
            model_s1 = self.model_fn(x_s1, s1)
            if solver_type == 'dpm_solver':
                x_t = (
                        expand_dims(sigma_t / sigma_s, dims) * x
                        - expand_dims(alpha_t * phi_1, dims) * model_s
                        - (0.5 / r1) * expand_dims(alpha_t * phi_1, dims) * (model_s1 - model_s)
                )
            elif solver_type == 'taylor':
                x_t = (
                        expand_dims(sigma_t / sigma_s, dims) * x
                        - expand_dims(alpha_t * phi_1, dims) * model_s
                        + (1. / r1) * expand_dims(alpha_t * ((torch.exp(-h) - 1.) / h + 1.), dims) * (
                                    model_s1 - model_s)
                )
        else:
            phi_11 = torch.expm1(r1 * h)
            phi_1 = torch.expm1(h)

            if model_s is None:
                model_s = self.model_fn(x, s)
            x_s1 = (
                    expand_dims(torch.exp(log_alpha_s1 - log_alpha_s), dims) * x
                    - expand_dims(sigma_s1 * phi_11, dims) * model_s
            )
            model_s1 = self.model_fn(x_s1, s1)
            if solver_type == 'dpm_solver':
                x_t = (
                        expand_dims(torch.exp(log_alpha_t - log_alpha_s), dims) * x
                        - expand_dims(sigma_t * phi_1, dims) * model_s
                        - (0.5 / r1) * expand_dims(sigma_t * phi_1, dims) * (model_s1 - model_s)
                )
            elif solver_type == 'taylor':
                x_t = (
                        expand_dims(torch.exp(log_alpha_t - log_alpha_s), dims) * x
                        - expand_dims(sigma_t * phi_1, dims) * model_s
                        - (1. / r1) * expand_dims(sigma_t * ((torch.exp(h) - 1.) / h - 1.), dims) * (model_s1 - model_s)
                )
        if return_intermediate:
            return x_t, {'model_s': model_s, 'model_s1': model_s1}
        else:
            return x_t

    def singlestep_dpm_solver_third_update(self, x, s, t, r1=1. / 3., r2=2. / 3., model_s=None, model_s1=None,
                                           return_intermediate=False, solver_type='dpm_solver'):
        """
        Singlestep solver DPM-Solver-3 from time `s` to time `t`.
        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (x.shape[0],).
            t: A pytorch tensor. The ending time, with the shape (x.shape[0],).
            r1: A `float`. The hyperparameter of the third-order solver.
            r2: A `float`. The hyperparameter of the third-order solver.
            model_s: A pytorch tensor. The model function evaluated at time `s`.
                If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
            model_s1: A pytorch tensor. The model function evaluated at time `s1` (the intermediate time given by `r1`).
                If `model_s1` is None, we evaluate the model at `s1`; otherwise we directly use it.
            return_intermediate: A `bool`. If true, also return the model value at time `s`, `s1` and `s2` (the intermediate times).
            solver_type: either 'dpm_solver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpm_solver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if solver_type not in ['dpm_solver', 'taylor']:
            raise ValueError("'solver_type' must be either 'dpm_solver' or 'taylor', got {}".format(solver_type))
        if r1 is None:
            r1 = 1. / 3.
        if r2 is None:
            r2 = 2. / 3.
        ns = self.noise_schedule
        dims = x.dim()
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        lambda_s2 = lambda_s + r2 * h
        s1 = ns.inverse_lambda(lambda_s1)
        s2 = ns.inverse_lambda(lambda_s2)
        log_alpha_s, log_alpha_s1, log_alpha_s2, log_alpha_t = ns.marginal_log_mean_coeff(
            s), ns.marginal_log_mean_coeff(s1), ns.marginal_log_mean_coeff(s2), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_s1, sigma_s2, sigma_t = ns.marginal_std(s), ns.marginal_std(s1), ns.marginal_std(
            s2), ns.marginal_std(t)
        alpha_s1, alpha_s2, alpha_t = torch.exp(log_alpha_s1), torch.exp(log_alpha_s2), torch.exp(log_alpha_t)

        if self.predict_x0:
            phi_11 = torch.expm1(-r1 * h)
            phi_12 = torch.expm1(-r2 * h)
            phi_1 = torch.expm1(-h)
            phi_22 = torch.expm1(-r2 * h) / (r2 * h) + 1.
            phi_2 = phi_1 / h + 1.
            phi_3 = phi_2 / h - 0.5

            if model_s is None:
                model_s = self.model_fn(x, s)
            if model_s1 is None:
                x_s1 = (
                        expand_dims(sigma_s1 / sigma_s, dims) * x
                        - expand_dims(alpha_s1 * phi_11, dims) * model_s
                )
                model_s1 = self.model_fn(x_s1, s1)
            x_s2 = (
                    expand_dims(sigma_s2 / sigma_s, dims) * x
                    - expand_dims(alpha_s2 * phi_12, dims) * model_s
                    + r2 / r1 * expand_dims(alpha_s2 * phi_22, dims) * (model_s1 - model_s)
            )
            model_s2 = self.model_fn(x_s2, s2)
            if solver_type == 'dpm_solver':
                x_t = (
                        expand_dims(sigma_t / sigma_s, dims) * x
                        - expand_dims(alpha_t * phi_1, dims) * model_s
                        + (1. / r2) * expand_dims(alpha_t * phi_2, dims) * (model_s2 - model_s)
                )
            elif solver_type == 'taylor':
                D1_0 = (1. / r1) * (model_s1 - model_s)
                D1_1 = (1. / r2) * (model_s2 - model_s)
                D1 = (r2 * D1_0 - r1 * D1_1) / (r2 - r1)
                D2 = 2. * (D1_1 - D1_0) / (r2 - r1)
                x_t = (
                        expand_dims(sigma_t / sigma_s, dims) * x
                        - expand_dims(alpha_t * phi_1, dims) * model_s
                        + expand_dims(alpha_t * phi_2, dims) * D1
                        - expand_dims(alpha_t * phi_3, dims) * D2
                )
        else:
            phi_11 = torch.expm1(r1 * h)
            phi_12 = torch.expm1(r2 * h)
            phi_1 = torch.expm1(h)
            phi_22 = torch.expm1(r2 * h) / (r2 * h) - 1.
            phi_2 = phi_1 / h - 1.
            phi_3 = phi_2 / h - 0.5

            if model_s is None:
                model_s = self.model_fn(x, s)
            if model_s1 is None:
                x_s1 = (
                        expand_dims(torch.exp(log_alpha_s1 - log_alpha_s), dims) * x
                        - expand_dims(sigma_s1 * phi_11, dims) * model_s
                )
                model_s1 = self.model_fn(x_s1, s1)
            x_s2 = (
                    expand_dims(torch.exp(log_alpha_s2 - log_alpha_s), dims) * x
                    - expand_dims(sigma_s2 * phi_12, dims) * model_s
                    - r2 / r1 * expand_dims(sigma_s2 * phi_22, dims) * (model_s1 - model_s)
            )
            model_s2 = self.model_fn(x_s2, s2)
            if solver_type == 'dpm_solver':
                x_t = (
                        expand_dims(torch.exp(log_alpha_t - log_alpha_s), dims) * x
                        - expand_dims(sigma_t * phi_1, dims) * model_s
                        - (1. / r2) * expand_dims(sigma_t * phi_2, dims) * (model_s2 - model_s)
                )
            elif solver_type == 'taylor':
                D1_0 = (1. / r1) * (model_s1 - model_s)
                D1_1 = (1. / r2) * (model_s2 - model_s)
                D1 = (r2 * D1_0 - r1 * D1_1) / (r2 - r1)
                D2 = 2. * (D1_1 - D1_0) / (r2 - r1)
                x_t = (
                        expand_dims(torch.exp(log_alpha_t - log_alpha_s), dims) * x
                        - expand_dims(sigma_t * phi_1, dims) * model_s
                        - expand_dims(sigma_t * phi_2, dims) * D1
                        - expand_dims(sigma_t * phi_3, dims) * D2
                )

        if return_intermediate:
            return x_t, {'model_s': model_s, 'model_s1': model_s1, 'model_s2': model_s2}
        else:
            return x_t

    def multistep_dpm_solver_second_update(self, x, model_prev_list, t_prev_list, t, solver_type="dpm_solver"):
        """
        Multistep solver DPM-Solver-2 from time `t_prev_list[-1]` to time `t`.
        Args:
            x: A pytorch tensor. The initial value at time `s`.
            model_prev_list: A list of pytorch tensor. The previous computed model values.
            t_prev_list: A list of pytorch tensor. The previous times, each time has the shape (x.shape[0],)
            t: A pytorch tensor. The ending time, with the shape (x.shape[0],).
            solver_type: either 'dpm_solver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpm_solver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if solver_type not in ['dpm_solver', 'taylor']:
            raise ValueError("'solver_type' must be either 'dpm_solver' or 'taylor', got {}".format(solver_type))
        ns = self.noise_schedule
        dims = x.dim()
        model_prev_1, model_prev_0 = model_prev_list
        t_prev_1, t_prev_0 = t_prev_list
        lambda_prev_1, lambda_prev_0, lambda_t = ns.marginal_lambda(t_prev_1), ns.marginal_lambda(
            t_prev_0), ns.marginal_lambda(t)
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0 = h_0 / h
        D1_0 = expand_dims(1. / r0, dims) * (model_prev_0 - model_prev_1)
        if self.predict_x0:
            if solver_type == 'dpm_solver':
                x_t = (
                        expand_dims(sigma_t / sigma_prev_0, dims) * x
                        - expand_dims(alpha_t * (torch.exp(-h) - 1.), dims) * model_prev_0
                        - 0.5 * expand_dims(alpha_t * (torch.exp(-h) - 1.), dims) * D1_0
                )
            elif solver_type == 'taylor':
                x_t = (
                        expand_dims(sigma_t / sigma_prev_0, dims) * x
                        - expand_dims(alpha_t * (torch.exp(-h) - 1.), dims) * model_prev_0
                        + expand_dims(alpha_t * ((torch.exp(-h) - 1.) / h + 1.), dims) * D1_0
                )
        else:
            if solver_type == 'dpm_solver':
                x_t = (
                        expand_dims(torch.exp(log_alpha_t - log_alpha_prev_0), dims) * x
                        - expand_dims(sigma_t * (torch.exp(h) - 1.), dims) * model_prev_0
                        - 0.5 * expand_dims(sigma_t * (torch.exp(h) - 1.), dims) * D1_0
                )
            elif solver_type == 'taylor':
                x_t = (
                        expand_dims(torch.exp(log_alpha_t - log_alpha_prev_0), dims) * x
                        - expand_dims(sigma_t * (torch.exp(h) - 1.), dims) * model_prev_0
                        - expand_dims(sigma_t * ((torch.exp(h) - 1.) / h - 1.), dims) * D1_0
                )
        return x_t

    def multistep_dpm_solver_third_update(self, x, model_prev_list, t_prev_list, t, solver_type='dpm_solver'):
        """
        Multistep solver DPM-Solver-3 from time `t_prev_list[-1]` to time `t`.
        Args:
            x: A pytorch tensor. The initial value at time `s`.
            model_prev_list: A list of pytorch tensor. The previous computed model values.
            t_prev_list: A list of pytorch tensor. The previous times, each time has the shape (x.shape[0],)
            t: A pytorch tensor. The ending time, with the shape (x.shape[0],).
            solver_type: either 'dpm_solver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpm_solver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        ns = self.noise_schedule
        dims = x.dim()
        model_prev_2, model_prev_1, model_prev_0 = model_prev_list
        t_prev_2, t_prev_1, t_prev_0 = t_prev_list
        lambda_prev_2, lambda_prev_1, lambda_prev_0, lambda_t = ns.marginal_lambda(t_prev_2), ns.marginal_lambda(
            t_prev_1), ns.marginal_lambda(t_prev_0), ns.marginal_lambda(t)
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        h_1 = lambda_prev_1 - lambda_prev_2
        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0, r1 = h_0 / h, h_1 / h
        D1_0 = expand_dims(1. / r0, dims) * (model_prev_0 - model_prev_1)
        D1_1 = expand_dims(1. / r1, dims) * (model_prev_1 - model_prev_2)
        D1 = D1_0 + expand_dims(r0 / (r0 + r1), dims) * (D1_0 - D1_1)
        D2 = expand_dims(1. / (r0 + r1), dims) * (D1_0 - D1_1)
        if self.predict_x0:
            x_t = (
                    expand_dims(sigma_t / sigma_prev_0, dims) * x
                    - expand_dims(alpha_t * (torch.exp(-h) - 1.), dims) * model_prev_0
                    + expand_dims(alpha_t * ((torch.exp(-h) - 1.) / h + 1.), dims) * D1
                    - expand_dims(alpha_t * ((torch.exp(-h) - 1. + h) / h ** 2 - 0.5), dims) * D2
            )
        else:
            x_t = (
                    expand_dims(torch.exp(log_alpha_t - log_alpha_prev_0), dims) * x
                    - expand_dims(sigma_t * (torch.exp(h) - 1.), dims) * model_prev_0
                    - expand_dims(sigma_t * ((torch.exp(h) - 1.) / h - 1.), dims) * D1
                    - expand_dims(sigma_t * ((torch.exp(h) - 1. - h) / h ** 2 - 0.5), dims) * D2
            )
        return x_t

    def singlestep_dpm_solver_update(self, x, s, t, order, return_intermediate=False, solver_type='dpm_solver', r1=None,
                                     r2=None):
        """
        Singlestep DPM-Solver with the order `order` from time `s` to time `t`.
        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (x.shape[0],).
            t: A pytorch tensor. The ending time, with the shape (x.shape[0],).
            order: A `int`. The order of DPM-Solver. We only support order == 1 or 2 or 3.
            return_intermediate: A `bool`. If true, also return the model value at time `s`, `s1` and `s2` (the intermediate times).
            solver_type: either 'dpm_solver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpm_solver' type.
            r1: A `float`. The hyperparameter of the second-order or third-order solver.
            r2: A `float`. The hyperparameter of the third-order solver.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if order == 1:
            return self.dpm_solver_first_update(x, s, t, return_intermediate=return_intermediate)
        elif order == 2:
            return self.singlestep_dpm_solver_second_update(x, s, t, return_intermediate=return_intermediate,
                                                            solver_type=solver_type, r1=r1)
        elif order == 3:
            return self.singlestep_dpm_solver_third_update(x, s, t, return_intermediate=return_intermediate,
                                                           solver_type=solver_type, r1=r1, r2=r2)
        else:
            raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))

    def multistep_dpm_solver_update(self, x, model_prev_list, t_prev_list, t, order, solver_type='dpm_solver'):
        """
        Multistep DPM-Solver with the order `order` from time `t_prev_list[-1]` to time `t`.
        Args:
            x: A pytorch tensor. The initial value at time `s`.
            model_prev_list: A list of pytorch tensor. The previous computed model values.
            t_prev_list: A list of pytorch tensor. The previous times, each time has the shape (x.shape[0],)
            t: A pytorch tensor. The ending time, with the shape (x.shape[0],).
            order: A `int`. The order of DPM-Solver. We only support order == 1 or 2 or 3.
            solver_type: either 'dpm_solver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpm_solver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if order == 1:
            return self.dpm_solver_first_update(x, t_prev_list[-1], t, model_s=model_prev_list[-1])
        elif order == 2:
            return self.multistep_dpm_solver_second_update(x, model_prev_list, t_prev_list, t, solver_type=solver_type)
        elif order == 3:
            return self.multistep_dpm_solver_third_update(x, model_prev_list, t_prev_list, t, solver_type=solver_type)
        else:
            raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))

    def dpm_solver_adaptive(self, x, order, t_T, t_0, h_init=0.05, atol=0.0078, rtol=0.05, theta=0.9, t_err=1e-5,
                            solver_type='dpm_solver'):
        """
        The adaptive step size solver based on singlestep DPM-Solver.
        Args:
            x: A pytorch tensor. The initial value at time `t_T`.
            order: A `int`. The (higher) order of the solver. We only support order == 2 or 3.
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            h_init: A `float`. The initial step size (for logSNR).
            atol: A `float`. The absolute tolerance of the solver. For image data, the default setting is 0.0078, followed [1].
            rtol: A `float`. The relative tolerance of the solver. The default setting is 0.05.
            theta: A `float`. The safety hyperparameter for adapting the step size. The default setting is 0.9, followed [1].
            t_err: A `float`. The tolerance for the time. We solve the diffusion ODE until the absolute error between the
                current time and `t_0` is less than `t_err`. The default setting is 1e-5.
            solver_type: either 'dpm_solver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpm_solver' type.
        Returns:
            x_0: A pytorch tensor. The approximated solution at time `t_0`.
        [1] A. Jolicoeur-Martineau, K. Li, R. Piché-Taillefer, T. Kachman, and I. Mitliagkas, "Gotta go fast when generating data with score-based models," arXiv preprint arXiv:2105.14080, 2021.
        """
        ns = self.noise_schedule
        s = t_T * torch.ones((x.shape[0],)).to(x)
        lambda_s = ns.marginal_lambda(s)
        lambda_0 = ns.marginal_lambda(t_0 * torch.ones_like(s).to(x))
        h = h_init * torch.ones_like(s).to(x)
        x_prev = x
        nfe = 0
        if order == 2:
            r1 = 0.5
            lower_update = lambda x, s, t: self.dpm_solver_first_update(x, s, t, return_intermediate=True)
            higher_update = lambda x, s, t, **kwargs: self.singlestep_dpm_solver_second_update(x, s, t, r1=r1,
                                                                                               solver_type=solver_type,
                                                                                               **kwargs)
        elif order == 3:
            r1, r2 = 1. / 3., 2. / 3.
            lower_update = lambda x, s, t: self.singlestep_dpm_solver_second_update(x, s, t, r1=r1,
                                                                                    return_intermediate=True,
                                                                                    solver_type=solver_type)
            higher_update = lambda x, s, t, **kwargs: self.singlestep_dpm_solver_third_update(x, s, t, r1=r1, r2=r2,
                                                                                              solver_type=solver_type,
                                                                                              **kwargs)
        else:
            raise ValueError("For adaptive step size solver, order must be 2 or 3, got {}".format(order))
        while torch.abs((s - t_0)).mean() > t_err:
            t = ns.inverse_lambda(lambda_s + h)
            x_lower, lower_noise_kwargs = lower_update(x, s, t)
            x_higher = higher_update(x, s, t, **lower_noise_kwargs)
            delta = torch.max(torch.ones_like(x).to(x) * atol, rtol * torch.max(torch.abs(x_lower), torch.abs(x_prev)))
            norm_fn = lambda v: torch.sqrt(torch.square(v.reshape((v.shape[0], -1))).mean(dim=-1, keepdim=True))
            E = norm_fn((x_higher - x_lower) / delta).max()
            if torch.all(E <= 1.):
                x = x_higher
                s = t
                x_prev = x_lower
                lambda_s = ns.marginal_lambda(s)
            h = torch.min(theta * h * torch.float_power(E, -1. / order).float(), lambda_0 - lambda_s)
            nfe += order
        
        return x

    def sample(self, x, steps=20, t_start=None, t_end=None, order=3, skip_type='time_uniform',
               method='singlestep', lower_order_final=True, denoise_to_zero=False, solver_type='dpm_solver',
               atol=0.0078, rtol=0.05,
               ):
        """
        Compute the sample at time `t_end` by DPM-Solver, given the initial `x` at time `t_start`.
        =====================================================
        We support the following algorithms for both noise prediction model and data prediction model:
            - 'singlestep':
                Singlestep DPM-Solver (i.e. "DPM-Solver-fast" in the paper), which combines different orders of singlestep DPM-Solver.
                We combine all the singlestep solvers with order <= `order` to use up all the function evaluations (steps).
                The total number of function evaluations (NFE) == `steps`.
                Given a fixed NFE == `steps`, the sampling procedure is:
                    - If `order` == 1:
                        - Denote K = steps. We use K steps of DPM-Solver-1 (i.e. DDIM).
                    - If `order` == 2:
                        - Denote K = (steps // 2) + (steps % 2). We take K intermediate time steps for sampling.
                        - If steps % 2 == 0, we use K steps of singlestep DPM-Solver-2.
                        - If steps % 2 == 1, we use (K - 1) steps of singlestep DPM-Solver-2 and 1 step of DPM-Solver-1.
                    - If `order` == 3:
                        - Denote K = (steps // 3 + 1). We take K intermediate time steps for sampling.
                        - If steps % 3 == 0, we use (K - 2) steps of singlestep DPM-Solver-3, and 1 step of singlestep DPM-Solver-2 and 1 step of DPM-Solver-1.
                        - If steps % 3 == 1, we use (K - 1) steps of singlestep DPM-Solver-3 and 1 step of DPM-Solver-1.
                        - If steps % 3 == 2, we use (K - 1) steps of singlestep DPM-Solver-3 and 1 step of singlestep DPM-Solver-2.
            - 'multistep':
                Multistep DPM-Solver with the order of `order`. The total number of function evaluations (NFE) == `steps`.
                We initialize the first `order` values by lower order multistep solvers.
                Given a fixed NFE == `steps`, the sampling procedure is:
                    Denote K = steps.
                    - If `order` == 1:
                        - We use K steps of DPM-Solver-1 (i.e. DDIM).
                    - If `order` == 2:
                        - We firstly use 1 step of DPM-Solver-1, then use (K - 1) step of multistep DPM-Solver-2.
                    - If `order` == 3:
                        - We firstly use 1 step of DPM-Solver-1, then 1 step of multistep DPM-Solver-2, then (K - 2) step of multistep DPM-Solver-3.
            - 'singlestep_fixed':
                Fixed order singlestep DPM-Solver (i.e. DPM-Solver-1 or singlestep DPM-Solver-2 or singlestep DPM-Solver-3).
                We use singlestep DPM-Solver-`order` for `order`=1 or 2 or 3, with total [`steps` // `order`] * `order` NFE.
            - 'adaptive':
                Adaptive step size DPM-Solver (i.e. "DPM-Solver-12" and "DPM-Solver-23" in the paper).
                We ignore `steps` and use adaptive step size DPM-Solver with a higher order of `order`.
                You can adjust the absolute tolerance `atol` and the relative tolerance `rtol` to balance the computatation costs
                (NFE) and the sample quality.
                    - If `order` == 2, we use DPM-Solver-12 which combines DPM-Solver-1 and singlestep DPM-Solver-2.
                    - If `order` == 3, we use DPM-Solver-23 which combines singlestep DPM-Solver-2 and singlestep DPM-Solver-3.
        =====================================================
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
        =====================================================
        Args:
            x: A pytorch tensor. The initial value at time `t_start`
                e.g. if `t_start` == T, then `x` is a sample from the standard normal distribution.
            steps: A `int`. The total number of function evaluations (NFE).
            t_start: A `float`. The starting time of the sampling.
                If `T` is None, we use self.noise_schedule.T (default is 1.0).
            t_end: A `float`. The ending time of the sampling.
                If `t_end` is None, we use 1. / self.noise_schedule.total_N.
                e.g. if total_N == 1000, we have `t_end` == 1e-3.
                For discrete-time DPMs:
                    - We recommend `t_end` == 1. / self.noise_schedule.total_N.
                For continuous-time DPMs:
                    - We recommend `t_end` == 1e-3 when `steps` <= 15; and `t_end` == 1e-4 when `steps` > 15.
            order: A `int`. The order of DPM-Solver.
            skip_type: A `str`. The type for the spacing of the time steps. 'time_uniform' or 'logSNR' or 'time_quadratic'.
            method: A `str`. The method for sampling. 'singlestep' or 'multistep' or 'singlestep_fixed' or 'adaptive'.
            denoise_to_zero: A `bool`. Whether to denoise to time 0 at the final step.
                Default is `False`. If `denoise_to_zero` is `True`, the total NFE is (`steps` + 1).
                This trick is firstly proposed by DDPM (https://arxiv.org/abs/2006.11239) and
                score_sde (https://arxiv.org/abs/2011.13456). Such trick can improve the FID
                for diffusion models sampling by diffusion SDEs for low-resolutional images
                (such as CIFAR-10). However, we observed that such trick does not matter for
                high-resolutional images. As it needs an additional NFE, we do not recommend
                it for high-resolutional images.
            lower_order_final: A `bool`. Whether to use lower order solvers at the final steps.
                Only valid for `method=multistep` and `steps < 15`. We empirically find that
                this trick is a key to stabilizing the sampling by DPM-Solver with very few steps
                (especially for steps <= 10). So we recommend to set it to be `True`.
            solver_type: A `str`. The taylor expansion type for the solver. `dpm_solver` or `taylor`. We recommend `dpm_solver`.
            atol: A `float`. The absolute tolerance of the adaptive step size solver. Valid when `method` == 'adaptive'.
            rtol: A `float`. The relative tolerance of the adaptive step size solver. Valid when `method` == 'adaptive'.
        Returns:
            x_end: A pytorch tensor. The approximated solution at time `t_end`.
        """
        t_0 = 1. / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        device = x.device
        if method == 'adaptive':
            with torch.no_grad():
                x = self.dpm_solver_adaptive(x, order=order, t_T=t_T, t_0=t_0, atol=atol, rtol=rtol,
                                             solver_type=solver_type)
        elif method == 'multistep':
            assert steps >= order
            timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device)
            assert timesteps.shape[0] - 1 == steps
            with torch.no_grad():
                vec_t = timesteps[0].expand((x.shape[0]))
                model_prev_list = [self.model_fn(x, vec_t)]
                t_prev_list = [vec_t]
                # Init the first `order` values by lower order multistep DPM-Solver.
                for init_order in tqdm(range(1, order), desc="DPM init order", disable=True):
                    vec_t = timesteps[init_order].expand(x.shape[0])
                    x = self.multistep_dpm_solver_update(x, model_prev_list, t_prev_list, vec_t, init_order,
                                                         solver_type=solver_type)
                    model_prev_list.append(self.model_fn(x, vec_t))
                    t_prev_list.append(vec_t)
                # Compute the remaining values by `order`-th order multistep DPM-Solver.
                for step in tqdm(range(order, steps + 1), desc="DPM multistep", disable=not self.pbar):
                    vec_t = timesteps[step].expand(x.shape[0])
                    if lower_order_final and steps < 15:
                        step_order = min(order, steps + 1 - step)
                    else:
                        step_order = order
                    x = self.multistep_dpm_solver_update(x, model_prev_list, t_prev_list, vec_t, step_order,
                                                         solver_type=solver_type)
                    for i in range(order - 1):
                        t_prev_list[i] = t_prev_list[i + 1]
                        model_prev_list[i] = model_prev_list[i + 1]
                    t_prev_list[-1] = vec_t
                    # We do not need to evaluate the final model value.
                    if step < steps:
                        model_prev_list[-1] = self.model_fn(x, vec_t)
        elif method in ['singlestep', 'singlestep_fixed']:
            if method == 'singlestep':
                timesteps_outer, orders = self.get_orders_and_timesteps_for_singlestep_solver(steps=steps, order=order,
                                                                                              skip_type=skip_type,
                                                                                              t_T=t_T, t_0=t_0,
                                                                                              device=device)
            elif method == 'singlestep_fixed':
                K = steps // order
                orders = [order, ] * K
                timesteps_outer = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=K, device=device)
            for i, order in enumerate(orders):
                t_T_inner, t_0_inner = timesteps_outer[i], timesteps_outer[i + 1]
                timesteps_inner = self.get_time_steps(skip_type=skip_type, t_T=t_T_inner.item(), t_0=t_0_inner.item(),
                                                      N=order, device=device)
                lambda_inner = self.noise_schedule.marginal_lambda(timesteps_inner)
                vec_s, vec_t = t_T_inner.tile(x.shape[0]), t_0_inner.tile(x.shape[0])
                h = lambda_inner[-1] - lambda_inner[0]
                r1 = None if order <= 1 else (lambda_inner[1] - lambda_inner[0]) / h
                r2 = None if order <= 2 else (lambda_inner[2] - lambda_inner[0]) / h
                x = self.singlestep_dpm_solver_update(x, vec_s, vec_t, order, solver_type=solver_type, r1=r1, r2=r2)
        if denoise_to_zero:
            x = self.denoise_to_zero_fn(x, torch.ones((x.shape[0],)).to(device) * t_0)
        return x

