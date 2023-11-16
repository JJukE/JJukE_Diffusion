""" LDM from https://github.com/labmlai/annotated_deep_learning_paper_implementations -> Not using for now """
from functools import partial
from typing import Callable, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, Size
from torch import nn
from einops import rearrange, repeat, reduce
from jjuke.utils import default

from .common import normal_kl, discretized_gaussian_log_likelihood
from .ddpm import DDPMTrainer, DDPMSampler # TODO: check if use DDIMSampler



class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean


# def gather(consts: torch.Tensor, t: torch.Tensor):
#     """Gather consts for $t$ and reshape to feature map shape"""
#     c = consts.gather(-1, t)
#     return c.reshape(-1, 1, 1, 1)


# class DenoiseDiffusion:
#     """
#     ## Denoise Diffusion
#     """

#     def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device):
#         """
#         * `eps_model` is $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$ model
#         * `n_steps` is $t$
#         * `device` is the device to place constants on
#         """
#         super().__init__()
#         self.eps_model = eps_model

#         # Create $\beta_1, \dots, \beta_T$ linearly increasing variance schedule
#         self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)

#         # $\alpha_t = 1 - \beta_t$
#         self.alpha = 1. - self.beta
#         # $\bar\alpha_t = \prod_{s=1}^t \alpha_s$
#         self.alpha_bar = torch.cumprod(self.alpha, dim=0)
#         # $T$
#         self.n_steps = n_steps
#         # $\sigma^2 = \beta$
#         self.sigma2 = self.beta

#     def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         #### Get $q(x_t|x_0)$ distribution

#         \begin{align}
#         q(x_t|x_0) &= \mathcal{N} \Big(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) \mathbf{I} \Big)
#         \end{align}
#         """

#         # [gather](utils.html) $\alpha_t$ and compute $\sqrt{\bar\alpha_t} x_0$
#         mean = gather(self.alpha_bar, t) ** 0.5 * x0
#         # $(1-\bar\alpha_t) \mathbf{I}$
#         var = 1 - gather(self.alpha_bar, t)
#         #
#         return mean, var

#     def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
#         """
#         #### Sample from $q(x_t|x_0)$

#         \begin{align}
#         q(x_t|x_0) &= \mathcal{N} \Big(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) \mathbf{I} \Big)
#         \end{align}
#         """

#         # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
#         if eps is None:
#             eps = torch.randn_like(x0)

#         # get $q(x_t|x_0)$
#         mean, var = self.q_xt_x0(x0, t)
#         # Sample from $q(x_t|x_0)$
#         return mean + (var ** 0.5) * eps

#     def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
#         """
#         #### Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$

#         \begin{align}
#         \textcolor{lightgreen}{p_\theta}(x_{t-1} | x_t) &= \mathcal{N}\big(x_{t-1};
#         \textcolor{lightgreen}{\mu_\theta}(x_t, t), \sigma_t^2 \mathbf{I} \big) \\
#         \textcolor{lightgreen}{\mu_\theta}(x_t, t)
#           &= \frac{1}{\sqrt{\alpha_t}} \Big(x_t -
#             \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)
#         \end{align}
#         """

#         # $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$
#         eps_theta = self.eps_model(xt, t)
#         # [gather](utils.html) $\bar\alpha_t$
#         alpha_bar = gather(self.alpha_bar, t)
#         # $\alpha_t$
#         alpha = gather(self.alpha, t)
#         # $\frac{\beta}{\sqrt{1-\bar\alpha_t}}$
#         eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
#         # $$\frac{1}{\sqrt{\alpha_t}} \Big(x_t -
#         #      \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)$$
#         mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
#         # $\sigma^2$
#         var = gather(self.sigma2, t)

#         # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
#         eps = torch.randn(xt.shape, device=xt.device)
#         # Sample
#         return mean + (var ** .5) * eps

#     def loss(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None):
#         """
#         #### Simplified Loss

#         $$L_{\text{simple}}(\theta) = \mathbb{E}_{t,x_0, \epsilon} \Bigg[ \bigg\Vert
#         \epsilon - \textcolor{lightgreen}{\epsilon_\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)
#         \bigg\Vert^2 \Bigg]$$
#         """
#         # Get batch size
#         batch_size = x0.shape[0]
#         # Get random $t$ for each sample in the batch
#         t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)

#         # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
#         if noise is None:
#             noise = torch.randn_like(x0)

#         # Sample $x_t$ for $q(x_t|x_0)$
#         xt = self.q_sample(x0, t, eps=noise)
#         # Get $\textcolor{lightgreen}{\epsilon_\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)$
#         eps_theta = self.eps_model(xt, t)

#         # MSE loss
#         return F.mse_loss(noise, eps_theta)


class LDMTrainer(DDPMTrainer):
    def __init__(
            self,
            betas: np.ndarray,
            model_mean_type="eps",
            model_var_type="fixed_small",
            loss_type="l2",
            p2_loss_weight_gamma=0.0,
            p2_loss_weight_k=1.0,
            # clip_denoised=True,
            cond_stage_config=None,
            num_timesteps_cond=None,
            scale_factor=1.,
            scale_by_std=False,
            cond_stage_type="image",
            cond_stage_forward=None,
            cond_stage_trainable=False,
            conditioning_type=None, # TODO
    ):
        """ General Latent Diffusion Trainer based on DDPM
        from https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/diffusion/ddpm.py
        First stage of the original LDM is skipped.
        
        Args:
            for DDPMTrainer:
                betas (np.ndarray): Noise schedule
                Hyperparameters for Diffusion
                model_mean_type (str, optional): Model mean type for calculating output. Defaults to "eps".
                model_var_type (str, optional): Model variation type for calculating output. Defaults to "fixed_small".
                clip_denoised (bool, optional): Clamping the output or not. Defaults to True.
                Hyperparameters for loss function
                    loss_type, p2_loss_weight_gamma, p2_loss_weight_k
            additional for LDM:
                num_timesteps_cond:
                scale_by_std:
        """
        # NOTE: clip_denoised = False here!
        self.clip_denoised = False
        
        super().__init__(betas, model_mean_type, model_var_type, loss_type, p2_loss_weight_gamma, p2_loss_weight_k, self.clip_denoised)

        # assert loss_type in "l2|rescaled_l2|l1|rescaled_l1|kl|rescaled_kl".split("|")
        # self.loss_type = loss_type
        # self.do_p2 = p2_loss_weight_gamma > 0.0

        # with self.register_diffusion_parameters():
        #     self.p2_loss_weight = (p2_loss_weight_k + self.alphas_cumprod / (1 - self.alphas_cumprod)) ** -p2_loss_weight_gamma
        
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= self.num_timesteps
        assert conditioning_type in [None, "concat", "crossattn"]
        assert (cond_stage_config is None) == (conditioning_type is None)
        
        self.conditioning_type = conditioning_type
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_type = cond_stage_type
        
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            with self.register_diffusion_parameters():
                self.scale_factor = torch.tensor(scale_factor)
        self.instantiate_cond_stage(cond_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.bbox_tokenizer = None # TODO: maybe for bounding box conditioning
        
        # add conditioning schedule
        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps-1, dtype=torch.long)
            ids = torch.round(torch.linspace(0, self.num_timesteps-1, self.num_timestepds_cond)).long()
            self.cond_ids[:self.num_timesteps_cond] = ids

    # TODO: check split_input_params..
    # def get_fold_unfold(self, x, kernel_size, stride, uf=1, df=1):
    #     """
        
    #     Args:
    #         x: image of shape (b, c, h, w)
        
    #     Returns:
    #         n number of image crops of shape (n, b, c, kernel_size[0], kernel_size[1])
    #     """
    #     b, c, h, w = x.shape
        
    #     # number of crops in img
    #     Ly = (h - kernel_size[0]) // stride[0] + 1
    #     Lx = (w - kernel_size[1]) // stride[1] + 1
        
    #     if uf == 1 and df == 1:
    #         fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
    #         unfold = nn.Unfold(**fold_params)
    #         fold = nn.Fold(output_size=x.shape[2:], **fold_params)
            
    #         weighting = self.get_weighting
    
    def get_learned_conditioning(self, cond):
        if self.cond_stage_forwrad is None:
            if hasattr(self.cond_stage_model, "encode") and callable(self.cond_stage_model.encode):
                cond = self.cond_stage_model.encode(cond)
                if isinstance(cond, DiagonalGaussianDistribution):
                    cond = cond.mode()
            else:
                cond = self.cond_stage_model(cond)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            cond = getattr(self.cond_stage_model, self.cond_stage_forward)(cond)
        return cond
    
    
    def forward(self, denoise_fn: Callable[[Tensor, Tensor], Tensor],
                x_start: Tensor, t: Tensor = None, cond: Tensor = None,
                noise: Tensor = None, return_info=False):
        batch_size = x_start.shape[0]
        t = default(t, lambda: torch.randint(0, self.num_timesteps, (batch_size, ), dtype=torch.long, device=self.device))
        
        if self.conditioning_type is not None:
            assert cond is not None
            if self.cond_stage_trainable:
                cond = self.get_learned_conditioning(cond)
            if self.shorten_cond_schedule:
                t_cond = self.cond_ids[t].to(self.device)
                cond = self.q_sample(x_start=cond, t=t_cond, noise=torch.randn_like(cond.float()))
        
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_t = self.q_sample(x_start, t, noise)

        losses = {}
        info = {"t": t, "eps": noise, "x_t": x_t, "x_start": x_start}

        if self.loss_type in ("kl", "rescaled_kl"):
            # \mathcal{L}_\text{kl}
            info.update(self.get_vlb(denoise_fn=denoise_fn, x_start=x_start, x_t=x_t, t=t))
            losses["loss"] = info["vlb"]
        
        elif self.loss_type in ("l2", "rescaled_l2", "l1", "rescaled_l1"):
            # \mathcal{L}_\text{vlb}
            model_out = denoise_fn(x_t, t, **cond)
            
            if self.model_var_type in ["learned", "learned_range"]:
                assert model_out.shape[1] == 2 * x_t.shape[1], "Output channel must be double of input channel for {}".format(self.model_var_type)
                model_out, tmp = torch.chunk(model_out, chunks=2, dim=1)
                frozen_out = torch.cat([model_out.detach(), tmp], dim=1)
                info.update(self.get_vlb(lambda *_: frozen_out, x_start, x_t, t)) # return the values of the variables regardless of its input
                losses["vlb"] = info["vlb"]
                if self.loss_type in ("rescaled_l1", "rescaled_l2"):
                    losses["vlb"] = losses["vlb"] * self.num_timesteps / 1000.
            # NOTE: vlb is not calculated when its fixed variance setting, but the fixed variances are used during sampling

            # \mathcal{L}_\text{simple}
            x_prev = self.q_posterior_mean_variance(x_start, x_t, t)[0]
            info["x_prev"] = x_prev
            
            target = {"eps": noise, "x_start": x_start, "x_prev": x_prev}[self.model_mean_type]
            info["target"] = target
            losses["recon"] = self.loss_fn(model_out, target)

            # \mathcal{L} = \mathcal{L}_\text{simple} * \lambda_\text{p2}
            if self.do_p2:
                losses["loss"] = losses["recon"] * self.p2_loss_weight[t]
            else:
                losses["loss"] = losses["recon"]
            
            # \mathcal{L}_\text{hybrid}
            if "vlb" in losses:
                losses["loss"] = losses["loss"] + losses["vlb"]
            
        return (losses, info) if return_info else losses # TODO: compare to LDM. Little different.


# TODO: check if the "condition" and "condition_cross" are right
class LDMSampler(DDPMSampler):
    def __init__(
            self,
            betas: np.ndarray,
            model_mean_type="eps",
            model_var_type="fixed_small",
            clip_denoised=False
    ):
        super().__init__(betas, model_mean_type, model_var_type, clip_denoised)
    

    def p_sample(self, denoise_fn: Callable[[Tensor, Tensor], Tensor],
                 x_t: Tensor, t: Tensor, condition=None, condition_cross=None):
        """ Sample x_{t-1} from p_theta(x_{t-1} | x_t) """
        out = self.p_mean_variance(denoise_fn=denoise_fn, x_t=x_t, t=t, condition=condition, condition_cross=condition_cross)
        noise = torch.randn_like(x_t)
        
        nonzero_mask = self.unsqueeze_as((t!=0), x_t)
        sample = out["model_mean"] + nonzero_mask * torch.exp(0.5 * out["model_log_var"]) * noise
        return sample
    

    def sample_progressive(self, denoise_fn: Callable[[Tensor, Tensor], Tensor], shape: Union[Sequence, Size],
                           noise: Tensor = None, condition=None, condition_cross=None) -> Tensor:
        """
        Generate samples

        Returns:
            Tensor: (num_samples, *shape)
        """
        batch_size = shape[0]
        sample = default(noise, lambda: torch.randn(shape, device=self.device))
        indices = list(range(self.num_timesteps))[::-1]
        for i in indices:
            t = torch.full((batch_size, ), i, dtype=torch.long, device=self.device)
            sample = self.p_sample(denoise_fn, sample, t, condition, condition_cross)
            yield sample
    

    def forward(self, denoise_fn: Callable[[Tensor, Tensor], Tensor], shape: Union[Sequence, Size],
                noise: Tensor = None, condition=None, condition_cross=None, num_samples=1) -> Tensor:
        """
        Generate samples, returning intermediate samples when num_samples > 1

        Returns:
            Tensor: (num_samples, *shape) or (*shape, )
        """
        sample = default(noise, lambda: torch.randn(shape, device=self.device))
        
        # variables for filtering intermediations
        sample_indices = np.linspace(0, self.num_timesteps, num_samples, dtype=np.int64).tolist()
        sample_list = []
        do_sampling = num_samples > 1
        for i, sample in enumerate(self.sample_progressive(
            denoise_fn, shape, sample, condition, condition_cross)):
            if do_sampling and i in sample_indices:
                sample_list.append(sample)
        
        if not do_sampling:
            return sample
        else:
            sample_list.append(sample)
            return torch.stack(sample_list)
