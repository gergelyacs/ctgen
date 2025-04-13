import torch
import torch.nn as nn
import numpy as np

from model.first_stage.autoencoder import Encoder, Decoder
from model.first_stage.loss import VAELossFn
from utils import Aggregator, count_params, exists
from tqdm import tqdm
from accelerate import Accelerator
from ema_pytorch import EMA
import logging
import os

logger = logging.getLogger(__name__)

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        #self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.mean.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2])

    def nll(self, sample, dims=[1,2]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean
    
class Autoencoder(nn.Module):
    def __init__(self, cfg: dict,  input_size : int):
        """
        Autoencoder 

        Args:
            in_channels: Image input channels
            latent_dim: Latent dimension 
        """
        super().__init__()
       
        # 2 * latent_dim because we need to represent the mean and variance of the latent distribution
        self.encoder = Encoder(latent_dim=2 * cfg.latent_dim, input_size=input_size, **cfg.autoencoder)

        # the sample will be drawn from the distribution 
        # the decoder will take the sample and reconstruct the image
        self.decoder = Decoder(latent_dim=cfg.latent_dim, input_size=input_size, **cfg.autoencoder)

   
    def encode(self, x: torch.Tensor):
        """ Forward pass through vector-quantized variational autoencoder.

        Args:
            x: Input image tensor.
        Returns:
            posterior: Posterior distribution of the latent representation.
        """
        z = self.encoder(x)
        return z
    
    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decode(x)
        return x_hat, z

    def decode(self, z: torch.Tensor):
        x_hat = self.decoder(z)
        return x_hat
    
    def get_latent(self, x: torch.Tensor):
        """ Get latent representation of input image.
        """
        return self.encode(x)

class VAE(nn.Module):
    def __init__(self, cfg: dict,  input_size : int,  accelerator):
        """
        Variational Autoencoder 

        Args:
            in_channels: Image input channels
            latent_dim: Latent dimension 
        """
        super().__init__()
        self.kl_weight = cfg.kl_weight

        self.ae = Autoencoder(cfg, input_size)

        logger.info ("==> Building VAE model with latent dimension: %s", cfg.latent_dim)
        logger.info ("==> Autoencoder configuration: %s", cfg.autoencoder)
        logger.info ("==> KL weight: %s", self.kl_weight)

        self.latent_size = self.ae.encoder.latent_size
        self.agg_metrics = []

        self.max_grad_norm = cfg.training.max_grad_norm

        logger.info(f"Number of encoder parameters: {count_params(self.ae.encoder):,}")
        logger.info(f"Number of decoder parameters: {count_params(self.ae.decoder):,}")

        if exists(accelerator):
            self.accelerator = accelerator
        else:
            self.accelerator = Accelerator(split_batches=True)

        if self.accelerator.is_main_process:
        
            if hasattr(cfg.training, 'ema_decay') and hasattr(cfg.training, 'ema_update_every'):
                self.ema_decay = cfg.training.ema_decay
                self.ema_update_every = cfg.training.ema_update_every
            else:  
                self.ema_decay = 0.995
                # Switch off EMA
                self.ema_update_every = float('inf')
         
            self.ema = EMA(self.ae, beta = self.ema_decay, update_every = self.ema_update_every)
            self.ema.to(self.accelerator.device)

   
    def encode(self, x: torch.Tensor):
        """ Forward pass through vector-quantized variational autoencoder.

        Args:
            x: Input image tensor.
        Returns:
            posterior: Posterior distribution of the latent representation.
        """
        moments = self.ae.encode(x)
        #print (moments.shape)
        #moments = rearrange(moments, 'b 1 (k n)  -> b k n', k=self.latent_dim)
   
        posterior = DiagonalGaussianDistribution(moments)
        return posterior
    
    def forward(self, x: torch.Tensor, sample_posterior=True):
        posterior = self.encode(x)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        x_hat = self.decode(z)
        return x_hat, posterior

    def decode(self, z: torch.Tensor):
        x_hat = self.ae.decode(z)
        return x_hat
    
    def get_latent(self, x: torch.Tensor, sample_posterior=True):
        """ Get latent representation of input image.
        """
        posterior = self.encode(x)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()

        return z

    def train_loop(self, train_cfg, train_loader, val_loader=None, start_epoch = 0, save_path=None, val_hook=None):   
        """ Initialize optimizer for training. """
        device = self.accelerator.device
        logger.info (f"Training on device: {device}")
        self.optimizer = torch.optim.Adam(self.ae.parameters(), train_cfg.lr)
        
        self.criterion = VAELossFn(**train_cfg.loss, kl_weight=self.kl_weight, last_decoder_layer=self.ae.decoder.out)
        self.criterion.to(device)

        self.ae.encode = self.accelerator.prepare(self.ae.encode)
        self.ae.decode = self.accelerator.prepare(self.ae.decode)
        self.optimizer = self.accelerator.prepare(self.optimizer)
        train_loader = self.accelerator.prepare(train_loader)
        val_loader = self.accelerator.prepare(val_loader)
        
        # start training
        for i in range(start_epoch, train_cfg.epoch):
            train_metrics = self.train_step(train_loader, i, warmup_epochs=train_cfg.warmup_epoch)    
            self.agg_metrics.append(train_metrics.get_avg("train_"))
            logger.info (f"Epoch: {i+1} / {train_cfg.epoch}; Training metrics: {train_metrics.get_avg()}")

            # we do distributed validation
            self.accelerator.wait_for_everyone()  

            if self.accelerator.is_main_process:
                self.ema.ema_model.eval()

            if exists(val_loader):
                val_metrics, recon_samples, orig_samples = self.validate(val_loader)
                # report validation loss
                logger.info (f"Validation metrics: {val_metrics.get_avg()}")  

                if self.accelerator.is_main_process:
                    if val_hook is not None:
                        val_hook(i, orig_samples, recon_samples, 'vae', train_cfg)

                self.agg_metrics[-1].update(val_metrics.get_avg("val_"))

            self.accelerator.wait_for_everyone()
            if exists(save_path) and self.accelerator.is_main_process:
                self.save(save_path, i + 1)
                logger.info (f"Model saved at: {save_path}")

                if (i + 1) % train_cfg.save_interval == 0:
                    filename, ext = os.path.splitext((os.path.basename(save_path)))
                    sp = os.path.join(os.path.dirname(save_path), f"{filename}-{i}{ext}")
                    self.save(sp, i + 1)
                    logger.info (f"Model saved at: {sp}")

            if self.accelerator.is_main_process:
                self.ema.ema_model.train()

            self.accelerator.wait_for_everyone() 


    def train_step(self, train_loader, curr_epoch, warmup_epochs=1):
        self.ae.train()
        accelerator = self.accelerator
        self.device = accelerator.device

        train_disc = self.criterion.disc_weight > 0 and curr_epoch >= warmup_epochs
        if train_disc:
            logger.info ("Discriminator is trained in the loss!")

        pbar = tqdm(train_loader, disable = not self.accelerator.is_main_process)
        pbar.set_description(f'Epoch {curr_epoch+1} (Training)')
        metrics = Aggregator()

        for x, _ in pbar:
            #x = x.to(device)

            #x_hat, posterior = self.forward(x)

            # compute loss
            # In https://arxiv.org/abs/2012.09841 they set the factor for the adversarial loss
            # to zero for the first iterations (suggestion: at least one epoch). Longer warm-ups
            # generally lead to better reconstructions.
            with accelerator.autocast():
                if not train_disc:
                    x_hat, posterior = self.forward(x)

                loss, logs = self.criterion(x_hat, x, posterior=posterior, disc_training=train_disc)
            accelerator.backward(loss)
            accelerator.wait_for_everyone()
            accelerator.clip_grad_norm_(self.ae.parameters(), max_norm=self.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            accelerator.wait_for_everyone()
        
            if train_disc:
                # update discriminator
                _, disc_logs = self.criterion.update_discriminator(x_hat, x)
                logs.update(disc_logs)
        
            metrics.update(logs)

            # EMA update
            if accelerator.is_main_process:
                self.ema.update()

            disp = ['rec_loss_low', 'rec_loss_high']  
            pbar.set_postfix({k: f"{v:.2f}" for k, v in metrics.get_avg().items() if k in disp}) 

        return metrics

    @torch.no_grad()
    def validate(self, val_loader):  
        self.eval()

        metrics = Aggregator()
        reconstructed_samples = torch.empty(0)
        orig_samples = torch.empty(0)
    
        for x, _ in tqdm(val_loader, desc="Validation"):
            #x = x.to(device)

            x_hat, p = self.forward(x)

            x_hat = self.accelerator.gather_for_metrics(x_hat)
            #p = self.accelerator.gather_for_metrics(p)
            x = self.accelerator.gather_for_metrics(x)
 
            # compute loss
            loss, logs = self.criterion(x_hat, x, posterior=p)

            # logging
            metrics.update(logs)
            # concat
            reconstructed_samples = torch.cat((reconstructed_samples, x_hat.cpu()), dim=0)
            orig_samples = torch.cat((orig_samples, x.cpu()), dim=0)
    
        return metrics, reconstructed_samples, orig_samples

    def save(self, save_path, epoch):
        if not self.accelerator.is_main_process:
            return
        
        state_dict = {'epoch' : epoch,
                      'metrics': self.agg_metrics,
                      'ema': self.ema.state_dict(),
                      'model_state_dict' : self.accelerator.get_state_dict(self.ae)}
        
        for key, param in state_dict.items():
            if isinstance(param, torch.Tensor):
                state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load(self, load_path):
        data = torch.load(load_path, weights_only=False)

        model = self.accelerator.unwrap_model(self.ae)
        model.load_state_dict(data['model_state_dict'])

        if self.accelerator.is_main_process:
            self.ema = EMA(self.ae, beta = self.ema_decay, update_every = self.ema_update_every)
            self.ema.load_state_dict(data['ema'])

        self.agg_metrics = data['metrics']
        self.eval()
        logger.info(f"VQVAE model loaded successfully from epoch {data['epoch']}.")
   
        return data['epoch']

    

