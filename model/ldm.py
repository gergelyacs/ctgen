"""
    LDM: Latent Diffusion Model for CTG synthesis
"""

import torch
from tqdm import tqdm
from model.unet import UNet1D
from model.unet_cond import UNet1D_cond
from model.diffusion import Diffusion
import torch.nn.init as init
from utils import Aggregator, exists, count_params
from model.first_stage import VAE
import os
from accelerate import Accelerator
from ema_pytorch import EMA

import logging

logger = logging.getLogger(__name__)

class LDM():
    def __init__(self, device, first_stage_model, accelerator,
                    in_channel=2,
                    loss_type='l1',  
                    schedule_opt=None, 
                    inner_channel=32, norm_groups=8, 
                    channel_mults=(1, 2, 4, 8, 8), 
                    class_nums=None,
                    max_grad_norm=1.0,
                    ema_decay=0.995,
                    ema_update_every=float('inf'),
                    cond_drop_prob=0.5, **kwargs):
        super(LDM, self).__init__()
        self.device = device

        self.scale_factor = 1.0

        self.first_stage_model = first_stage_model
        self.first_stage_model.to(device)

        # in_channel must be latent_dim

        if exists(class_nums):
            logger.info ("Conditional generation!")
            model = UNet1D_cond(in_channel=in_channel, 
                        out_channel=in_channel,
                        class_nums=class_nums,
                        inner_channel=inner_channel,
                        norm_groups=norm_groups, 
                        channel_mults=channel_mults, 
                        cond_drop_prob=cond_drop_prob)
        else:
            logger.info ("Unconditional generation!")
            model = UNet1D(in_channel=in_channel, 
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
        logger.info(f"Number of LDM parameters: {params:,}")
        self.agg_metrics = []
 
        self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=kwargs['lr'].init)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', 
                                                                       factor=kwargs['lr'].factor, 
                                                                       patience=kwargs['lr'].patience, 
                                                                       min_lr=kwargs['lr'].min)
     
        self.max_grad_norm = max_grad_norm
        if exists(accelerator):
            self.accelerator = accelerator
        else:
            self.accelerator = Accelerator(split_batches=True)

        self.ema_decay = ema_decay
        self.ema_update_every = ema_update_every
        if self.accelerator.is_main_process:
            self.ema = EMA(self.diffusion, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.accelerator.device)
                
    def weights_init_orthogonal(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.orthogonal_(m.weight.data, gain=1)
            if exists(m.bias):
                m.bias.data.zero_()
        elif classname.find('Linear') != -1 and not classname.find('LinearAttention') != -1:
            init.orthogonal_(m.weight.data, gain=1)
            if exists(m.bias):
                m.bias.data.zero_()
        elif classname.find('BatchNorm1d') != -1:
            init.constant_(m.weight.data, 1.0)
            init.constant_(m.bias.data, 0.0)
    
    # Progress whole reverse diffusion process
    def synthetise(self, sample_num=None, classes=None, **kwargs):
        # if none of the sample_num and classes are given
        assert exists(sample_num) or exists(classes), "Error: Either sample_num or classes must be given."
        if exists(self.class_nums):
            # generate random classes
            if not exists(classes):
                #classes = torch.randint(0, self.num_classes, (sample_num,), device=self.device) 
                # draw random numbers from min to self.num_classes[i] for each class
                classes = [torch.randint(0, self.class_nums[i], (sample_num,))[..., None].to(self.device) for i in range(len(self.class_nums))]
                classes = torch.cat(classes, dim=-1)
            z = self.ema.ema_model.generate(data_size=self.first_stage_model.latent_size, classes=classes, **kwargs)
        else:
            z = self.ema.ema_model.generate(sample_num=sample_num, data_size=self.first_stage_model.latent_size, **kwargs)
        # For VQGAN we quantize the LDM output. This quarantees that the decoder
        # receives elements from the codebook.
        return self.first_stage_model.decode(z / self.scale_factor), classes    
    
    def train_loop(self, train_cfg, train_loader, start_epoch = 0, val_loader=None, save_path=None, val_hook=None):   
        """ Initialize optimizer for training. """
        
        self.diffusion, self.optimizer, train_loader = self.accelerator.prepare(self.diffusion, self.optimizer, train_loader)
       
        logger.info ("Training on device: %s", self.accelerator.device)
 
        # start training
        for i in range(start_epoch, train_cfg.epoch):
            logger.info (f"Learning rate: {self.lr_scheduler.get_last_lr()[0]:.6f}")
            train_metrics = self.train_step(train_loader, i)    
            self.agg_metrics.append(train_metrics.get_avg())
            logger.info (f"Epoch: {i+1} / {train_cfg.epoch}; Training metrics: {train_metrics.get_avg()}")
            # print get_last_lr
        
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
               
                if exists(val_hook):
                    self.ema.ema_model.eval()
                    # self.diffusion is the same as self.ema.ema_model
                    val_hook(self, i)
                    self.ema.ema_model.train()
                
                if exists(save_path):
                    self.save(save_path, i + 1)
                    logger.info ("Model saved at: %s", save_path)

                    if (i + 1) % train_cfg.save_interval == 0:
                        filename, ext = os.path.splitext((os.path.basename(save_path)))
                        sp = os.path.join(os.path.dirname(save_path), f"{filename}-{i}{ext}")
                        self.save(sp, i + 1)
                        logger.info ("Model saved at: %s", sp)
           
            # FIXME: validation loss
            self.lr_scheduler.step(train_metrics['train_loss'][-1])
            self.accelerator.wait_for_everyone()
       
    def train_step(self, train_loader, curr_epoch):
        self.diffusion.train()
        accelerator = self.accelerator
        self.device = accelerator.device
        
        pbar = tqdm(train_loader, disable = not self.accelerator.is_main_process)
        pbar.set_description(f'Epoch {curr_epoch+1} (Training)')
        metrics = Aggregator()
        self.first_stage_model = self.first_stage_model.to(self.device)

        for i, samples in enumerate(pbar):
            # check if imgs is tuple -> conditional generation
            if isinstance(samples, list):
                imgs, labels = samples
                #labels = labels.to(self.device)
            else:
                imgs, labels = samples, None

            #b, c, ts_len = imgs.shape
     
            #imgs = imgs.to(self.device)
            
            with accelerator.autocast():
                z = self.first_stage_model.get_latent(imgs)
                if i == 0 and curr_epoch == 0 and isinstance(self.first_stage_model, VAE):
                    self.scale_factor = 1. / z.flatten().std().detach()
                    logger.info ("Scale factor: %s", self.scale_factor.item())
                
                z = z * self.scale_factor

                loss = self.diffusion(z, classes=labels)
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

    def save(self,save_path, epoch):
        if not self.accelerator.is_main_process:
            return

        state_dict = {'scale_factor' : self.scale_factor,
                      'epoch' : epoch,
                      'metrics': self.agg_metrics,
                      'ema': self.accelerator.get_state_dict(self.ema),
                      'lr': self.lr_scheduler.state_dict(),
                      'optimizer' : self.accelerator.get_state_dict(self.optimizer),
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
        self.optimizer = self.accelerator.unwrap_model(self.optimizer)
        self.optimizer.load_state_dict(data['optimizer'])
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        self.lr_scheduler.load_state_dict(data['lr'])
        self.diffusion.eval()
        logger.info(f"LDM model loaded successfully from epoch {data['epoch']}.")
        return data['epoch']
      