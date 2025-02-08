"""
    LDM: Latent Diffusion Model for CTG synthesis
"""

import torch
from tqdm import tqdm
from model.unet import UNet1D
from model.unet_cond import UNet1D_cond
from model.diffusion import Diffusion
import torch.nn.init as init
from utils import Aggregator, exists, count_params, hash_tensor, merge_time_series
from model.first_stage import VAE
from model.first_stage import Sampling
import os

class LDM():
    def __init__(self, device, first_stage_model, 
                    in_channel=2,
                    loss_type='l1',  
                    schedule_opt=None, 
                    inner_channel=32, norm_groups=8, 
                    channel_mults=(1, 2, 4, 8, 8), 
                    num_classes=None,
                    cond_drop_prob=0.5):
        super(LDM, self).__init__()
        self.device = device

        self.scale_factor = 1.0

        #self.first_stage_model = first_stage_model
        
        self.first_stage_models = [first_stage_model.to(device), 
                                   Sampling(sampling_rate=first_stage_model.sampling_rate, input_size=first_stage_model.input_size, shift=first_stage_model.sampling_rate // 2).to(device)]

        # in_channel must be latent_dim

        model = UNet1D_cond(in_channel=in_channel, 
            out_channel=in_channel,
            num_classes=2,
            inner_channel=inner_channel,
            norm_groups=norm_groups, 
            channel_mults=channel_mults, 
            cond_drop_prob=cond_drop_prob)

        '''
        if exists(num_classes):
            model = UNet1D_cond(in_channel=in_channel, 
                        out_channel=in_channel,
                        num_classes=num_classes,
                        inner_channel=inner_channel,
                        norm_groups=norm_groups, 
                        channel_mults=channel_mults, 
                        cond_drop_prob=cond_drop_prob)
        else:
            model = UNet1D(in_channel=in_channel, 
                        out_channel=in_channel,
                        inner_channel=inner_channel,
                        norm_groups=norm_groups, 
                        channel_mults=channel_mults)
        '''
                        
        self.num_classes = num_classes
        self.diffusion = Diffusion(model, device)

        # Apply weight initialization & set loss & set noise schedule
        self.diffusion.apply(self.weights_init_orthogonal)
        self.diffusion.set_loss(loss_type)
        if exists(schedule_opt):
            self.diffusion.set_new_noise_schedule(schedule_opt)

        params = count_params(self.diffusion)
        print(f"Number of LDM parameters: {params:,}")

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

    def synthetise(self, sample_num=None):
        z = self.diffusion.generate(sample_num=sample_num, data_size=self.first_stage_models[0].latent_size)
        # z has shape (2 * sample_num, latent_size)
        merged = torch.from_numpy(merge_time_series(z, self.first_stage_models[0].sampling_rate)).float()
        components = self.first_stage_models[0].decode(z / self.scale_factor)
        # concat merged and components along the first dimension
        return torch.cat([merged, components], dim=0), None      
    
    '''
    # Progress whole reverse diffusion process
    def synthetise(self, sample_num=None, classes=None):
        # if none of the sample_num and classes are given
        assert exists(sample_num) or exists(classes), "Error: Either sample_num or classes must be given."
        if exists(self.num_classes):
            # generate random classes
            if not exists(classes):
                classes = torch.randint(0, self.num_classes, (sample_num,), device=self.device) 
            z = self.diffusion.generate(data_size=self.first_stage_model.latent_size, classes=classes, cond_scale=3.)
        else:
            z = self.diffusion.generate(sample_num=sample_num, data_size=self.first_stage_model.latent_size)
        # For VQGAN we quantize the LDM output. This quarantees that the decoder
        # receives elements from the codebook.
        return self.first_stage_model.decode(z / self.scale_factor), classes
    '''
    
    def train_loop(self, train_cfg, train_loader, start_epoch = 0, val_loader=None, save_path=None, val_hook=None):   
        """ Initialize optimizer for training. """
        print ("Training on device:", self.device)

        self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=train_cfg.lr)
     
        # start training
        for i in range(start_epoch, train_cfg.epoch):
            train_metrics = self.train_step(train_loader, i)    
            print (f"Epoch: {i+1} / {train_cfg.epoch}; Training metrics:", train_metrics.get_avg())

            if exists(val_hook):
                self.diffusion.eval()
                val_hook(self, i)
              
            if exists(save_path):
                self.save(save_path, i + 1)
                print ("Model saved at:", save_path)

                if (i + 1) % train_cfg.save_interval == 0:
                    filename, ext = os.path.splitext((os.path.basename(save_path)))
                    sp = os.path.join(os.path.dirname(save_path), f"{filename}-{i}{ext}")
                    self.save(sp, i + 1)
                    print ("Model saved at:", sp)
   
    def train_step(self, train_loader, curr_epoch):
        self.diffusion.train()
        pbar = tqdm(train_loader)
        pbar.set_description(f'Epoch {curr_epoch+1} (Training)')
        metrics = Aggregator()

        log = {}
        for i, samples in enumerate(pbar):

            # check if imgs is tuple -> conditional generation
            if isinstance(samples, list):
                imgs, labels = samples
                labels = labels.to(self.device)
            else:
                imgs, labels = samples, None

            b, c, ts_len = imgs.shape
     
            imgs = imgs.to(self.device)
            
            self.optimizer.zero_grad()
            # get latent representation from first stage model
            res_num = len(self.first_stage_models)
            z_res = [first_stage_model.get_latent(imgs) for first_stage_model in self.first_stage_models]  
            #print (imgs[0, :, :64])
            #print (z_res[0][0, :, :64])
            #print (z_res[1][0, :, :64])
            seeds = [hash_tensor(img) for img in imgs] * res_num
        
            z = torch.cat(z_res, dim=0)
            # labels: different integers for each stage
            labels = torch.cat([torch.zeros(z_res[i].shape[0]) + i for i in range(res_num)], dim=0).long().to(self.device)
            
            #if i == 0 and curr_epoch == 0 and isinstance(self.first_stage_model, VAE):
            #    self.scale_factor = 1. / z.flatten().std().detach()
            #    print ("Scale factor:", self.scale_factor.item())
            
            z = z * self.scale_factor

            loss = self.diffusion(z, classes=labels, seeds=seeds)
            #loss = loss.sum() / int(b * c * ts_len)
            loss.backward()
            self.optimizer.step()
            log['train_loss'] = loss.item() * b
            metrics.update(log)
            disp = ['train_loss']  
            pbar.set_postfix({k: f"{v:.2f}" for k, v in metrics.get_avg().items() if k in disp}) 

        return metrics

    def save(self, save_path, epoch):
        state_dict = {'scale_factor' : self.scale_factor,
                      'epoch' : epoch,
                      'model_state_dict' : self.diffusion.state_dict()}
        for key, param in state_dict.items():
            if isinstance(param, torch.Tensor):
                state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load(self, load_path):
        network = self.diffusion
        tmp = torch.load(load_path)
        self.scale_factor = tmp['scale_factor']
        network.load_state_dict(tmp['model_state_dict'])
        network.eval()
        print(f"LDM model loaded successfully from epoch {tmp['epoch']}.")
        return tmp['epoch']
    
    def load_old(self, load_path):
        network = self.diffusion
        tmp = torch.load(load_path)
        self.scale_factor = tmp['scale_factor']
        network.load_state_dict(tmp['model_state_dict'])
        network.eval()
        print("LDM model loaded successfully.")
      