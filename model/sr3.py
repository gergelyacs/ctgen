"""
    SR3 Super-Resolution of CTG for Healthcare
"""

from model.unet import UNet1D
from model.diffusion import Diffusion
from model.unet_cond import UNet1D_cond
import torch 
import torch.nn.init as init
from utils import exists, count_params, Aggregator
from model.first_stage import VAE
from tqdm import tqdm
import os
from multiprocessing import cpu_count
from accelerate import Accelerator
from ema_pytorch import EMA

import logging

logger = logging.getLogger(__name__)

# Class to train & test desired model
class SR3():
    def __init__(self, device, transform_model, first_stage_model,
                    accelerator,
                    in_channel=2,
                    schedule_opt=None,
                    loss_type='l1', 
                    inner_channel=32, norm_groups=8, 
                    channel_mults=(1, 2, 4, 8, 8),
                    class_nums=None,
                    max_grad_norm=1.0,
                    ema_decay=0.995,
                    ema_update_every=float('inf'),
                    cond_drop_prob=0.5, **kwargs):
        super(SR3, self).__init__()
        self.device = device

        self.scale_factor = 1.0
     
        self.first_stage_model = first_stage_model
        self.first_stage_model.to(device)
        self.transform_model = transform_model
        self.transform_model.to(device)
        self.transform = lambda x: self.transform_model(x)[0]
        
        # in_channel = 2 * channel_num (low-res is the condition which is concatenated to the high-res input)
        if exists(class_nums):
            model = UNet1D_cond(in_channel=2 * in_channel, 
                        out_channel=in_channel,
                        class_nums=class_nums,
                        inner_channel=inner_channel,
                        norm_groups=norm_groups, 
                        channel_mults=channel_mults, 
                        cond_drop_prob=cond_drop_prob)
        else:
            model = UNet1D(in_channel=2 * in_channel, 
                        out_channel=in_channel,
                        inner_channel=inner_channel,
                        norm_groups=norm_groups, 
                        channel_mults=channel_mults)
      
        self.class_nums = class_nums
        self.diffusion = Diffusion(model, device)

        # Apply weight initialization & set loss & set noise schedule
        self.diffusion.apply(self.weights_init_orthogonal)
        self.diffusion.set_loss(loss_type) 
        if exists(schedule_opt):
            self.diffusion.set_new_noise_schedule(schedule_opt)
 
        params = count_params(self.diffusion)
        logger.info(f"Number of SR3 parameters: {params:,}")
        self.agg_metrics = []

        self.max_grad_norm = max_grad_norm
        self.ema_decay = ema_decay
        self.ema_update_every = ema_update_every
        
        if exists(accelerator):
            self.accelerator = accelerator
        else:
            self.accelerator = Accelerator(split_batches=False)

        if self.accelerator.is_main_process:
            self.ema = EMA(self.diffusion, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.accelerator.device)

    def weights_init_orthogonal(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('Linear') != -1 and not classname.find('LinearAttention') != -1:
            init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm1d') != -1:
            init.constant_(m.weight.data, 1.0)
            init.constant_(m.bias.data, 0.0)       

    def train_loop(self, train_cfg, train_loader, val_loader = None, start_epoch = 0, save_path=None, val_hook=None):   
        """ Initialize optimizer for training. """
        logger.info ("Training on device: %s", self.device)

        self.optimizer = torch.optim.Adam(self.diffusion.parameters(), lr=train_cfg.lr)
     
        self.diffusion, self.optimizer, train_loader, val_loader = self.accelerator.prepare(self.diffusion, self.optimizer, train_loader, val_loader)
        
        logger.info ("CPU count: %s", cpu_count())
     
        # start training
        for i in range(start_epoch, train_cfg.epoch):
            train_metrics = self.train_step(train_loader, i)   
            self.agg_metrics.append(train_metrics.get_avg("train_"))
            logger.info (f"Epoch: {i+1} / {train_cfg.epoch}; Training metrics: {train_metrics.get_avg()}")

            # we do distributed validation
            self.accelerator.wait_for_everyone()  

            if self.accelerator.is_main_process:
                self.ema.ema_model.eval()

            if exists(val_hook) and exists(val_loader):
                sr_samples, orig_samples, lr_samples, dec_samples, val_metrics = self.validate(val_loader)
                
                if self.accelerator.is_main_process:
                    val_hook(sr_samples, orig_samples, lr_samples, dec_samples, i)

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

    def train_step(self, train_loader, curr_epoch):
        self.diffusion.train()
        pbar = tqdm(train_loader, disable = not self.accelerator.is_main_process)
        pbar.set_description(f'Epoch {curr_epoch+1} (Training)')
        metrics = Aggregator()
        self.first_stage_model = self.first_stage_model.to(self.accelerator.device)

        accelerator = self.accelerator
        self.device = accelerator.device

        for i, samples in enumerate(pbar):            

            # check if imgs is tuple -> conditional generation
            if isinstance(samples, list):
                imgs, labels = samples
                #labels = labels.to(self.device)
            else:
                imgs, labels = samples, None

            # Initial imgs are high-resolution
            b, c, ts_len = imgs.shape

            #imgs = imgs.to(self.device)
         
            # low-resolution
            #save_signals(imgs, f'./test_orig_{i+1}.jpg')
            LR_imgs = self.transform(imgs).to(self.device)
          
            #LR_imgs = torch.from_numpy(upsample_time_series(downsample_time_series(imgs, SAMPLING_RATE), SAMPLING_RATE))
            #save_signals(LR_imgs, f'./test_lr_{i+1}.jpg')
            #logger.info ("Low-res shape: ", LR_imgs.shape)

            with accelerator.autocast():
                z = self.first_stage_model.get_latent(imgs)
                z_LR = self.first_stage_model.get_latent(LR_imgs)
                if i == 0 and curr_epoch == 0 and isinstance(self.first_stage_model, VAE):
                    self.scale_factor = (1. / z.flatten().std().detach() + 1. / z_LR.flatten().std().detach()) / 2.
                    logger.info ("Scale factor: %s", self.scale_factor.item())
            
                z = z * self.scale_factor
                z_LR = z_LR * self.scale_factor

                loss = self.diffusion(z, condition_x = z_LR, classes=labels)
                #loss = loss.sum() / int(b * c * ts_len)
            accelerator.backward(loss)
            accelerator.wait_for_everyone()
            accelerator.clip_grad_norm_(self.diffusion.parameters(), max_norm=self.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            accelerator.wait_for_everyone()

            metrics.update({'train_loss': loss.item()})

            # EMA update
            if accelerator.is_main_process:
                self.ema.update()

            disp = ['train_loss']  
            pbar.set_postfix({k: f"{v:.2f}" for k, v in metrics.get_avg().items() if k in disp}) 
            
        return metrics
    
    @torch.no_grad()
    def validate(self, val_loader):  
        #self.diffusion.eval()

        reconstructed_samples = torch.empty(0)
        orig_samples = torch.empty(0)
        lr_samples = torch.empty(0)
        dec_samples = torch.empty(0)
        metrics = Aggregator()
        for x, _ in val_loader:
            #x = x.to(self.device)  
            x_hat, x_lr = self.test(x)
            #x_hat = self.accelerator.gather_for_metrics(x_hat)
            #x_lr = self.accelerator.gather_for_metrics(x_lr)
            #x = self.accelerator.gather_for_metrics(x)
            x_dec, _, _ = self.first_stage_model.forward(x)
            x_dec = self.accelerator.gather_for_metrics(x_dec)

            # compute loss between x_hat and x
            if self.accelerator.use_distributed:
                metrics.update({"loss": self.diffusion.module.loss_func(x_hat, x).mean().item()})
            else:
                metrics.update({"loss": self.diffusion.loss_func(x_hat, x).mean().item()})

            # concat
            reconstructed_samples = torch.cat((reconstructed_samples, x_hat.cpu()), dim=0)
            orig_samples = torch.cat((orig_samples, x.cpu()), dim=0)
            lr_samples = torch.cat((lr_samples, x_lr.cpu()), dim=0)
            dec_samples = torch.cat((dec_samples, x_dec.cpu()), dim=0)
         
        return reconstructed_samples, orig_samples, lr_samples, dec_samples, metrics

    def test(self, imgs):
        imgs_lr = self.transform(imgs).to(self.device)
        if exists(self.class_nums):
            # generate random classes
            #classes = torch.randint(0, self.num_classes, (imgs.shape[0],), device=self.device) 
            classes = [torch.randint(0, self.class_nums[i], (imgs.shape[0],))[..., None].to(self.device) for i in range(len(self.class_nums))]
            classes = torch.cat(classes, dim=-1)
        else:
            classes = None

        return self.super_resolution(imgs_lr, classes=classes), imgs_lr
    
    # Progress whole reverse diffusion process
    def super_resolution(self, x_in, **kwargs):
        z_in = self.first_stage_model.get_latent(x_in)
        if self.accelerator.use_distributed:
            z_out = self.diffusion.module.generate(sample_num=x_in.shape[0], condition_x=z_in, data_size=self.first_stage_model.latent_size, pbar=self.accelerator.is_main_process, **kwargs)
        else:
            z_out = self.diffusion.generate(sample_num=x_in.shape[0], condition_x=z_in, data_size=self.first_stage_model.latent_size, pbar=self.accelerator.is_main_process, **kwargs)       
        return self.first_stage_model.decode(z_out / self.scale_factor)

    def save(self, save_path, epoch):
        if not self.accelerator.is_main_process:
            return
        
        state_dict = {'epoch' : epoch,
                      'scale_factor' : self.scale_factor,
                      'metrics': self.agg_metrics,
                      'ema': self.ema.state_dict(),
                      'model_state_dict' : self.accelerator.get_state_dict(self.diffusion)}
        
        for key, param in state_dict.items():
            if isinstance(param, torch.Tensor):
                state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load(self, load_path):
        data = torch.load(load_path, weights_only=False)

        diffusion = self.accelerator.unwrap_model(self.diffusion)
        diffusion.load_state_dict(data['model_state_dict'])

        self.scale_factor = data['scale_factor']
      
        if self.accelerator.is_main_process:
            self.ema = EMA(self.diffusion, beta = self.ema_decay, update_every = self.ema_update_every)
            self.ema.load_state_dict(data['ema'])
  
        self.agg_metrics = data['metrics']
        self.diffusion.eval()
        logger.info(f"SR3 model loaded successfully from epoch {data['epoch']}.")
        return data['epoch']
    
    
