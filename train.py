import os
import sys
import numpy as np
import argparse

from sklearn.model_selection import train_test_split
from utils import create_dir, plot_syn_data, unscale, save_signals
from torch.utils.data import DataLoader
from model import LDM, SR3
from model.first_stage import VAE, VQGAN, Sampling, Identity#, CS
from omegaconf import OmegaConf
from dataloader.loader import PyTablesDataset
import torch
from accelerate import Accelerator
from multiprocessing import cpu_count
import logging

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s: %(message)s')

np.set_printoptions(legacy='1.25')

parser = argparse.ArgumentParser(description="Training models for CTG generation")

parser.add_argument('--config', default='configs/sampling.yaml',
                    metavar='PATH', help='Path to model config file (default: configs/conf.yaml)')

def first_stage_hook(epoch, x, x_hat, model_type, train_cfg, num_samples=10, idxs=[5]):
    # plot a random sample
    x = unscale(x, cfg.scale_factors)[:num_samples]
    x_hat = unscale(x_hat, cfg.scale_factors)[:num_samples]
    #idxs = [5] #np.random.randint(0, x.shape[0], nsamples_to_plot)
    #_cfg = cfg.first_stage_model.training

    # select random samples from the last batch to plot
    for idx in idxs:
        name = f"{cfg.out_dir}/{train_cfg.out_dir}/{model_type}_sample_{epoch}_{idx}"
        plot_syn_data(x[idx].numpy(), x_hat[idx].numpy(), title=name)
        logger.info ("Reconstructed sample saved at: %s", name)

def ldm_hook(model, epoch):
    # generate 50 synthetic samples
    syn_imgs, classes = model.synthetise(sample_num=10)
    syn_imgs = unscale(syn_imgs, cfg.scale_factors)
    name = f"{cfg.out_dir}/{cfg.ldm.training.out_dir}/ldm_{epoch}.jpg"
    save_signals(syn_imgs, name, classes=classes)

def sr3_hook(sr_samples, orig_samples, LR_samples, dec_samples, epoch, num_samples=10):
    # generated data is scaled to [0,1]
    sr_imgs = unscale(sr_samples, cfg.scale_factors)[:num_samples]
    name = f"{cfg.out_dir}/{cfg.sr_model.training.out_dir}/sr_{epoch}.jpg"
    save_signals(sr_imgs, name)

    # Save example of test images to check training
    orig_imgs = unscale(orig_samples, cfg.scale_factors)[:num_samples]
    name = f"{cfg.out_dir}/{cfg.sr_model.training.out_dir}/orig_{epoch}.jpg"
    save_signals(orig_imgs, name)

    LR_imgs = unscale(LR_samples, cfg.scale_factors)[:num_samples]
    name = f"{cfg.out_dir}/{cfg.sr_model.training.out_dir}/lr_{epoch}.jpg"
    save_signals(LR_imgs, name)

    dec_imgs = unscale(dec_samples, cfg.scale_factors)[:num_samples]
    name = f"{cfg.out_dir}/{cfg.sr_model.training.out_dir}/dec_{epoch}.jpg"
    save_signals(dec_imgs, name)

def load_first_stage(encoder_cfg, out_dir, input_size, in_channels, accelerator=None):
    if encoder_cfg.type == 'vae':
        # load VAE
        logger.info (f"==> Loading VAE model from {encoder_cfg.model_path}...") 
        first_stage_model = VAE(cfg=encoder_cfg, input_size=input_size, accelerator=accelerator)
        in_channel = encoder_cfg.latent_dim
        first_stage_model.load(f"{out_dir}/{encoder_cfg.model_path}")      

    elif encoder_cfg.type == 'vqgan':
        # load VQGAN
        logger.info (f"==> Loading VQGAN model from {encoder_cfg.model_path}...")
        first_stage_model = VQGAN(cfg=encoder_cfg, input_size=input_size, accelerator=accelerator)
        in_channel = encoder_cfg.latent_dim
        first_stage_model.load(f"{out_dir}/{encoder_cfg.model_path}")      

    elif encoder_cfg.type == 'sampling':
        # Diffusion in undersampled time domain
        logger.info (f"==> Using Downsampling with rate {encoder_cfg.sampling_rate}")
        first_stage_model = Sampling(sampling_rate=encoder_cfg.sampling_rate, input_size=input_size)
        in_channel = in_channels

    elif encoder_cfg.type == 'identity':
        # Diffusion in time domain
        logger.info (f"==> Using Identity model")
        first_stage_model = Identity(input_size=input_size)
        in_channel = in_channels

    #elif encoder_cfg.type == 'cs':
    #    logger.info (f"==> Using Compressed Sensing model")
    #    first_stage_model = CS(cfg=encoder_cfg, input_size=input_size)
    #    in_channel = in_channels
    else:
        raise NotImplementedError(type)
    
    return first_stage_model, in_channel

def continue_training(model_path, model, _train_cfg, trainloader, testloader = None, val_hook=None):   
    if os.path.exists(model_path):
        logger.info (f"Loading model from {model_path}...")
        start_epoch = model.load(model_path)
    else:
        start_epoch = 0

    if start_epoch < _train_cfg.epoch:
        model.train_loop(_train_cfg, trainloader, val_loader=testloader,    
                     save_path=model_path, start_epoch=start_epoch, val_hook=val_hook)
    else:
        logger.info (f"Model already trained for {_train_cfg.epoch} epochs.")

def train_first_stage(encoder_cfg, dataset, accelerator=None):
    type = encoder_cfg.type.lower()
   
    out_dir = cfg.out_dir
    input_size = cfg.input_size
    train_idx, val_idx = train_test_split(np.arange(len(dataset)), test_size=cfg.validation_size)
    
    #train_idx = train_idx[:3000]

    # create two subsets
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)

    logger.info (f"Training samples: {len(train_dataset):,}; Validation samples: {len(val_dataset):,}")

    batch_size = encoder_cfg.training.batch_size

    # only 10 samples for testing 
    trainloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, num_workers=cpu_count(), shuffle=True)
    testloader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, num_workers=cpu_count(), shuffle=True)

    #trainloader = accelerator.prepare(trainloader)
    #testloader = accelerator.prepare(testloader)

    if type == 'vae':        
        model = VAE(cfg=encoder_cfg, input_size=input_size, accelerator=accelerator)
        model.to(cfg.device)
        
        continue_training(f"{out_dir}/{encoder_cfg.model_path}", model, encoder_cfg.training, trainloader, testloader, first_stage_hook)

    elif type == 'vqgan':
        model = VQGAN(cfg=encoder_cfg, input_size=input_size, accelerator=accelerator)
        model.to(cfg.device)
        
        continue_training(f"{out_dir}/{encoder_cfg.model_path}", model, encoder_cfg.training, trainloader, testloader, first_stage_hook)
    elif type == 'identity':
        return

    #elif type == 'cs':
    #    model = CS(cfg=cfg.first_stage_model, input_size=input_size)
    #    model.validate(testloader, 
    #                 save_path=f"{out_dir}/{encoder_cfg.model_path}", 
    #                 val_hook=first_stage_hook)
    #    model.save(f"{out_dir}/{encoder_cfg.model_path}")
    else:
        logger.error("Invalid model type. Please choose from vae, vqgan")
        sys.exit(1)

def train_ldm(dataset, accelerator=None): 
    batch_size = cfg.ldm.training.batch_size  
    #if hasattr(cfg.ldm, 'unconstrained') and cfg.ldm.unconstrained:
    #    class_nums = None 
    #else:
    #    class_nums = dataset.class_nums

    #train_idx, val_idx = train_test_split(np.arange(len(dataset)), test_size=100)
    #train_idx = train_idx[:1000]
    # create two subsets
    class_nums = dataset.class_nums
    #dataset = torch.utils.data.Subset(dataset, train_idx)

    trainloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=cpu_count(), shuffle=True)

    _train_cfg = cfg.ldm.training 
    # load first stage model
    first_stage_model, in_channel = load_first_stage(cfg.first_stage_model, cfg.out_dir, cfg.input_size, cfg.in_channels, accelerator)
    model = LDM(cfg.device, accelerator=accelerator, first_stage_model=first_stage_model, in_channel=in_channel, class_nums=class_nums, **_train_cfg)
    
    continue_training(f"{cfg.out_dir}/{cfg.ldm.model_path}", model, _train_cfg, trainloader, val_hook=ldm_hook)
  
def train_sr3(dataset, accelerator=None):
    # split idxs into training and validation with 80% and 20% using scikit
    train_idx, val_idx = train_test_split(np.arange(len(dataset)), test_size=cfg.validation_size)
    
    # create two subsets
    #train_idx = train_idx[:1000]
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)

    logger.info (f"Training samples: {len(train_dataset):,}; Validation samples: {len(val_dataset):,}")

    batch_size = cfg.sr_model.training.batch_size
 
    # only 10 samples for testing
 
    trainloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, num_workers=cpu_count(), shuffle=True)
    testloader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, num_workers=cpu_count(), shuffle=True)
    
    transform_model, _ = load_first_stage(cfg.first_stage_model, cfg.out_dir, cfg.input_size, cfg.in_channels)
    first_stage_model, in_channel = load_first_stage(cfg.sr_model.first_stage_model, cfg.out_dir, cfg.input_size, cfg.in_channels)   
    _train_cfg = cfg.sr_model.training   
    
    model = SR3(cfg.device, accelerator=accelerator, transform_model=transform_model, first_stage_model=first_stage_model, in_channel=in_channel, class_nums=dataset.class_nums, **_train_cfg)

    continue_training(f"{cfg.out_dir}/{cfg.sr_model.model_path}", model, _train_cfg, trainloader, testloader, sr3_hook)

def main():
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    for name, val in vars(args).items():
        logger.info("{:<16}: {}".format(name, val))

    # Load config file
    global cfg
    cfg = OmegaConf.load(args.config)

    # load training data: (sample_num, time, channels)
    #patient_data = np.load(cfg.training_data)
    _, ext = os.path.splitext((os.path.basename(cfg.training_data)))
    if ext == '.h5':
        logger.info (f"{cfg.training_data}")
        # since the data is stored in (time, channels) format, we need to transpose it for PyTorch
        # the data is already scaled into [-1, 1]
        if hasattr(cfg, 'class_file') and hasattr(cfg, 'classes'):
            patient_data = PyTablesDataset(cfg.training_data, transpose=True, class_file=cfg.class_file, classes=cfg.classes)
        else:
            patient_data = PyTablesDataset(cfg.training_data, transpose=True)
    else:
        raise NotImplementedError(ext)
    
    #mixed_precision_type='fp16'
    #amp = False

    accelerator = Accelerator(split_batches=True)
    #accelerator = None

    #assert cfg.input_size == patient_data.shape[1], f"Error: Input size in config file {cfg.input_size} does not match loaded data size {patient_data.shape[1]} in {patient_data.shape}" 
    #assert cfg.in_channels == patient_data.shape[2], f"Error: Number of channels in config file {cfg.in_channels} does not match loaded data size {patient_data.shape[2]} in {patient_data.shape}"  

    # change channel for torch:
    # (batch_size, time, channels) -> (batch_size, channels, time)

    # data is already normalized into [0, 1], we need the normalizing factors to dispay generated data
    logger.info("FHR max: %s", cfg.scale_factors.fhr_max)
    logger.info("FHR min: %s", cfg.scale_factors.fhr_min)
    logger.info("UC max: %s", cfg.scale_factors.uc_max)
    logger.info("UC min: %s", cfg.scale_factors.uc_min)

    # create output directory
    create_dir(cfg.out_dir)
    type = cfg.first_stage_model.type.lower()

    # if first stage model exists, load it
    if type in ['sampling', 'identity']:
        logger.info (f"=== {type} model does not require first-stage training.")
    else:
        logger.info (f"Training first stage {type.upper()} model...")
        create_dir(f"{cfg.out_dir}/{cfg.first_stage_model.training.out_dir}")
        train_first_stage(cfg.first_stage_model, patient_data, accelerator=accelerator)

    # if ldm model does not exist, train it
    if hasattr(cfg, 'ldm') and not os.path.exists(cfg.ldm.model_path):
        logger.info ("=== Training Latent Diffusion Model (LDM)...")
        create_dir(f"{cfg.out_dir}/{cfg.ldm.training.out_dir}") 
        train_ldm(patient_data, accelerator=accelerator)
    
    # if sr3 model is in the config file, train it
    if hasattr(cfg, 'sr_model'):
        
        if hasattr(cfg.sr_model, 'first_stage_model'):
            type = cfg.sr_model.first_stage_model.type.lower()
            logger.info (f"=== Training first stage {type.upper()} model for SR3...")
            create_dir(f"{cfg.out_dir}/{cfg.sr_model.first_stage_model.training.out_dir}")
            train_first_stage(cfg.sr_model.first_stage_model, patient_data, accelerator)

        if not os.path.exists(cfg.sr_model.model_path):
            logger.info ("Training Super Resolution (SR3) model...")
            create_dir(f"{cfg.out_dir}/{cfg.sr_model.training.out_dir}")
            train_sr3(patient_data, accelerator)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Exit training with keyboard interrupt!")
        sys.exit(0)

   