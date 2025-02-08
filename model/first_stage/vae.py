import torch
import torch.nn as nn
import numpy as np

from model.first_stage.autoencoder import Encoder, Decoder
from model.first_stage.loss import VAELossFn
from utils import Aggregator, count_params, exists
from tqdm import tqdm
import os

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        #self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 1, dim=1)
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
    
class VAE(nn.Module):
    def __init__(self, cfg: dict,  input_size : int):
        """
        Variational Autoencoder 

        Args:
            in_channels: Image input channels
            latent_dim: Latent dimension 
        """
        super(VAE, self).__init__()
        self.kl_weight = cfg.kl_weight

        print ("==> Building VAE model with latent dimension:", cfg.latent_dim)
        print ("==> Autoencoder configuration:", cfg.autoencoder)
        print ("==> KL weight:", self.kl_weight)

        # 2 * latent_dim because we need to represent the mean and variance of the latent distribution
        self.encoder = Encoder(latent_dim=2 * cfg.latent_dim, input_size=input_size, **cfg.autoencoder)

        # the sample will be drawn from the distribution 
        # the decoder will take the sample and reconstruct the image
        self.decoder = Decoder(latent_dim=cfg.latent_dim, input_size=input_size, **cfg.autoencoder)

        self.latent_size = self.encoder.latent_size
        self.agg_metrics = []

        print(f"Number of encoder parameters: {count_params(self.encoder):,}")
        print(f"Number of decoder parameters: {count_params(self.decoder):,}")

   
    def encode(self, x: torch.Tensor):
        """ Forward pass through vector-quantized variational autoencoder.

        Args:
            x: Input image tensor.
        Returns:
            posterior: Posterior distribution of the latent representation.
        """
        moments = self.encoder(x)
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
        x_hat = self.decoder(z)
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
        device = next(self.parameters()).device
        print ("Training on device:", device)
        optimizer = torch.optim.Adam(self.parameters(), train_cfg.lr)
        
        criterion = VAELossFn(**train_cfg.loss, kl_weight=self.kl_weight, last_decoder_layer=self.decoder.out)
        criterion.to(device)

        # start training
        for i in range(start_epoch, train_cfg.epoch):
            train_metrics = self.train_step(train_loader, optimizer, criterion, i, device)    
            self.agg_metrics.append(train_metrics.get_avg("train_"))
            print (f"Epoch: {i+1} / {train_cfg.epoch}; Training metrics:", train_metrics.get_avg())

            if val_loader is not None:
                val_metrics, recon_samples, orig_samples = self.validate(val_loader, criterion, device)
                # report validation loss
                print ("Validation metrics:", val_metrics.get_avg())  

                if val_hook is not None:
                    val_hook(i, orig_samples, recon_samples, 'vae')

                self.agg_metrics[-1].update(val_metrics.get_avg("val_"))

            if exists(save_path):
                self.save(save_path, i + 1)
                print ("Model saved at:", save_path)

                if (i + 1) % train_cfg.save_interval == 0:
                    filename, ext = os.path.splitext((os.path.basename(save_path)))
                    sp = os.path.join(os.path.dirname(save_path), f"{filename}-{i}{ext}")
                    self.save(sp, i + 1)
                    print ("Model saved at:", sp)


    def train_step(self, train_loader, optimizer, criterion, curr_epoch, device):
        self.train()

        pbar = tqdm(train_loader)
        pbar.set_description(f'Epoch {curr_epoch+1} (Training)')
        metrics = Aggregator()

        for x, _ in pbar:
            x = x.to(device)

            x_hat, posterior = self.forward(x)

            # compute loss
            # In https://arxiv.org/abs/2012.09841 they set the factor for the adversarial loss
            # to zero for the first iterations (suggestion: at least one epoch). Longer warm-ups
            # generally lead to better reconstructions.
            if criterion.disc_weight > 0 and curr_epoch != 0:
                # update generator
                loss, logs = criterion(x_hat, x, posterior=posterior, disc_training=True)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # update discriminator
                _, disc_logs = criterion.update_discriminator(x_hat, x)
                logs.update(disc_logs)
            else:
                loss, logs = criterion(x_hat, x, posterior=posterior)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            metrics.update(logs)
            disp = ['rec_loss_low', 'rec_loss_high']  
            pbar.set_postfix({k: f"{v:.2f}" for k, v in metrics.get_avg().items() if k in disp}) 


        return metrics

    @torch.no_grad()
    def validate(self, val_loader, criterion, device):  
        self.eval()

        metrics = Aggregator()
        reconstructed_samples = torch.empty(0)
        orig_samples = torch.empty(0)
    
        for x, _ in tqdm(val_loader, desc="Validation"):
            x = x.to(device)

            x_hat, p = self.forward(x)

            # compute loss
            loss, logs = criterion(x_hat, x, posterior=p)

            # logging
            metrics.update(logs)
            # concat
            reconstructed_samples = torch.cat((reconstructed_samples, x_hat.cpu()), dim=0)
            orig_samples = torch.cat((orig_samples, x.cpu()), dim=0)
    
        return metrics, reconstructed_samples, orig_samples

    def save(self, save_path, epoch):
        state_dict = {'epoch' : epoch,
                      'metrics': self.agg_metrics,
                      'model_state_dict' : self.state_dict()}
        for key, param in state_dict.items():
            if isinstance(param, torch.Tensor):
                state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load(self, load_path):
        tmp = torch.load(load_path)
        self.load_state_dict(tmp['model_state_dict'])
        self.agg_metrics = tmp['metrics']
        self.eval()
        return tmp['epoch']

    

