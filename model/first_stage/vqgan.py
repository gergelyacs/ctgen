import torch
import torch.nn as nn

from model.first_stage.quantizer import VectorQuantizer
from model.first_stage.autoencoder import Encoder, Decoder
from model.first_stage.loss import VQLossFn
from utils import Aggregator, count_params, exists
from tqdm import tqdm
from vector_quantize_pytorch import VectorQuantize
import os
from accelerate import Accelerator
from ema_pytorch import EMA
import logging

logger = logging.getLogger(__name__)

class VQVAE(nn.Module):
    def __init__(self, cfg, input_size : int):
        """
        Vector-quantized Autoencoder (paper: https://arxiv.org/abs/2012.09841)
        """
        super().__init__()
        n_embeddings = cfg.n_embeddings

        self.encoder = Encoder(latent_dim=cfg.latent_dim, input_size=input_size, **cfg.autoencoder)
        
        #self.vq = VectorQuantizer(n_embeddings, cfg.latent_dim)

        self.vq = VectorQuantize(codebook_size=n_embeddings, decay=0.8, sync_codebook=True, use_cosine_sim = True, dim=cfg.latent_dim)
        
        self.decoder = Decoder(latent_dim=cfg.latent_dim, input_size=input_size, **cfg.autoencoder)

    def forward(self, x: torch.Tensor):
        """ Forward pass through vector-quantized variational autoencoder.

        Args:
            x: Input image tensor.
        Returns:
            x_hat: Reconstructed image x
            z_e: Latent (un-quantized) representation of image x
            z_q: Quantized latent representation of image x
        """
        z_e = self.encoder(x)
        z_q, _, _ = self.vq(z_e.transpose(1, 2))
        z_q = z_q.transpose(1, 2)
        #z_q = self.vq(z_e)
       
        # preserve gradients
        z_q_ = z_e + (z_q - z_e).detach()
        x_hat = self.decoder(z_q_)

        return x_hat, z_e, z_q

    def encode(self, x: torch.Tensor):
        """ Encode input image.

        Args:
            x: Input image
        Returns:
            z_e: Encoded input image (un-quantized).
        """
        z_e = self.encoder(x)

        return z_e

    def quantize(self, z_e: torch.Tensor):
        """ Quantize latent representation.

        Args:
            z_e: Un-quantized latent representation (encoded image).
        Returns:
            z_q: Quantized embedding.
        """
        z_q, _, _ = self.vq(z_e.transpose(1, 2))
        z_q = z_q.transpose(1, 2)
        #z_q = self.vq(z_e)

        return z_q

    def decode(self, z_e: torch.Tensor):
        """ Decode latent representation to input image.

        Args:
            z_e: Un-quantized latent representation.
        Returns:
            x_hat: Reconstructed input image.
        """
        z_q, _, _ = self.vq(z_e.transpose(1, 2))
        z_q = z_q.transpose(1, 2)
        #z_q = self.vq(z_e)

        x_hat = self.decoder(z_q)
        
        return x_hat
    
    def get_latent(self, x: torch.Tensor):
        """ Get latent representation of input image.
        """
        z_e = self.encode(x)
        z_q = self.quantize(z_e)

        return z_q

class VQGAN(nn.Module):
    def __init__(self, cfg, input_size : int, accelerator): 
        """
        Vector-quantized GAN (paper: https://arxiv.org/abs/2012.09841)

        Args:
            latent_dim: Latent dimension of the embedding/codebook
            autoencoder_cfg: Dictionary containing the information for the encoder and decoder. 
            n_embeddings: Number of embeddings for the codebook
        """
        super().__init__()
        n_embeddings = cfg.n_embeddings

        self.ae = VQVAE(cfg, input_size)

        logger.info ("==> Building VQGAN model with latent dimension: %s", cfg.latent_dim)
        logger.info ("==> Autoencoder configuration: %s", cfg.autoencoder)
        logger.info ("==> Number of embeddings: %s", n_embeddings)

        logger.info(f"Number of encoder parameters: {count_params(self.ae.encoder):,}")
        logger.info(f"Number of quantizer parameters: {count_params(self.ae.vq):,}")
        logger.info(f"Number of decoder parameters: {count_params(self.ae.decoder):,}")

        self.latent_size = self.ae.encoder.latent_size
        self.agg_metrics = []

        self.max_grad_norm = cfg.training.max_grad_norm
        
        if exists(accelerator):
            self.accelerator = accelerator
        else:
            self.accelerator = Accelerator(split_batches=False)

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

    def forward(self, x: torch.Tensor):
        return self.ae(x)
  
    def encode(self, x: torch.Tensor):
        return self.ae.encode(x)    

    def decode(self, z_e: torch.Tensor):
        return self.ae.decode(z_e)
        
    def get_latent(self, x: torch.Tensor):
        return self.ae.get_latent(x)
    
    def train_loop(self, train_cfg, train_loader, val_loader=None, start_epoch = 0, save_path=None, val_hook=None):   
        """ Initialize optimizer for training. """
        #device = next(self.parameters()).device
        device = self.accelerator.device
        logger.info ("Training on device: %s", device)
        self.optimizer = torch.optim.Adam(self.ae.parameters(), lr=train_cfg.lr)
        
        self.criterion = VQLossFn(**train_cfg.loss, last_decoder_layer=self.ae.decoder.out)
        self.criterion.to(device)

        self.ae, self.optimizer = self.accelerator.prepare(self.ae, self.optimizer)
        train_loader = self.accelerator.prepare(train_loader)
        val_loader = self.accelerator.prepare(val_loader)

        # start training
        for i in range(start_epoch, train_cfg.epoch):
            train_metrics = self.train_step(train_loader, i, device, warmup_epochs=train_cfg.warmup_epoch)    
            self.agg_metrics.append(train_metrics.get_avg("train_"))
            logger.info (f"Epoch: {i+1} / {train_cfg.epoch}; Training metrics: {train_metrics.get_avg()}")

            # we do distributed validation
            self.accelerator.wait_for_everyone()  

            if self.accelerator.is_main_process:
                self.ema.ema_model.eval()

            if exists(val_loader):
                val_metrics, recon_samples, orig_samples = self.validate(val_loader, device)
                # report validation loss
                logger.info ("Validation metrics: %s", val_metrics.get_avg())  
                
                if self.accelerator.is_main_process:
                    if val_hook is not None:
                        val_hook(i, orig_samples, recon_samples, 'vqgan', train_cfg)

                self.agg_metrics[-1].update(val_metrics.get_avg("val_"))
            
            self.accelerator.wait_for_everyone()
            if exists(save_path) and self.accelerator.is_main_process:
                self.save(save_path, i + 1)
                logger.info ("Model saved at: %s", save_path)

                if (i + 1) % train_cfg.save_interval == 0:
                    filename, ext = os.path.splitext((os.path.basename(save_path)))
                    sp = os.path.join(os.path.dirname(save_path), f"{filename}-{i}{ext}")
                    self.save(sp, i + 1)
                    logger.info ("Model saved at: %s", sp)

            if self.accelerator.is_main_process:
                self.ema.ema_model.train()
            
            self.accelerator.wait_for_everyone()        
        
    def train_step(self, train_loader, curr_epoch, device, warmup_epochs=1):
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

            #x_hat, z_e, z_q = self.forward(x)

            # compute loss
            # In https://arxiv.org/abs/2012.09841 they set the factor for the adversarial loss
            # to zero for the first iterations (suggestion: at least one epoch). Longer warm-ups
            # generally lead to better reconstructions.
            # update (only) teacher parameters
             
            with accelerator.autocast():
                if not train_disc:
                    x_hat, z_e, z_q = self.forward(x)

                loss, logs = self.criterion(x_hat, x, z_e, z_q, disc_training=train_disc)
            accelerator.backward(loss)
            accelerator.wait_for_everyone()
            accelerator.clip_grad_norm_(self.ae.parameters(), max_norm=self.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            accelerator.wait_for_everyone()
            
            if train_disc:      
                # update discriminator
                #if logs["generator_loss"] > 0.1 or logs["rec_loss"] + logs["codebook_loss"] > 0.15:
                _, disc_logs = self.criterion.update_discriminator(x_hat, x)
                logs.update(disc_logs)

            metrics.update(logs)
            # update progress bar
            # add them to the progress bar

            # EMA update
            if accelerator.is_main_process:
                self.ema.update()

            disp = ['rec_loss_low', 'rec_loss_high']  
            pbar.set_postfix({k: f"{v:.2f}" for k, v in metrics.get_avg().items() if k in disp}) 

        return metrics

    @torch.no_grad()
    def validate(self, val_loader, device):  
        #self.ema.ema_model.eval()
               
        metrics = Aggregator()
        reconstructed_samples = torch.empty(0)
        orig_samples = torch.empty(0)
        
        for x, _ in tqdm(val_loader, desc="Validation", disable = not self.accelerator.is_main_process):
            #x = x.to(device)

            x_hat, z_e, z_q = self.forward(x)
            x_hat = self.accelerator.gather_for_metrics(x_hat)
            z_e = self.accelerator.gather_for_metrics(z_e)
            z_q = self.accelerator.gather_for_metrics(z_q)
            x = self.accelerator.gather_for_metrics(x)

            # compute loss, all threads compute the same loss
            #self.accelerator.wait_for_everyone()
            loss, logs = self.criterion(x_hat, x, z_e, z_q)    
            #print ("Validation loss:", loss)
            #self.accelerator.wait_for_everyone()
            
            # logging
            metrics.update(logs)
            # concat
            reconstructed_samples = torch.cat((reconstructed_samples, x_hat.cpu()), dim=0)
            orig_samples = torch.cat((orig_samples, x.cpu()), dim=0)
        
        return metrics, reconstructed_samples, orig_samples

    def save(self,  save_path, epoch):
        if not self.accelerator.is_local_main_process:
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
    


    