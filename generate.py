import os
import sys
import numpy as np
import argparse

from utils import create_dir, unscale, save_signals, exists
from model.ldm import LDM
from model.sr3 import SR3
from omegaconf import OmegaConf
import pandas as pd
from train import load_first_stage
import tables
from utils import load_classes, get_class_index, get_class_vector
import torch
from model.diffusion import SAMPLERS
import logging

logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s: %(message)s')

parser = argparse.ArgumentParser(description="Generating synthetic CTG")

parser.add_argument('--config', default='configs/sampling_czech.yaml',
                    metavar='PATH', help='Path to model config file (default: configs/conf.yaml)')

parser.add_argument('--samples', default='10', type=str,
                    help='Samples to generate (default: 10) OR path to file specifying the classes')

parser.add_argument('--out_dir', default='./output/',
                    metavar='PATH', help='Output directory (default: ./output)')

parser.add_argument('--steps', default=1000,
                    type=int, help='Sampling time steps (default: 1000)')

parser.add_argument('--out_h5', default='generated_samples.h5',
                    metavar='PATH', help='HDF5 output file (default: generated_samples.h5)')

parser.add_argument('--cond_scale', default='1', type=float,  
                    help='conditional scale (default: 1.0)')

parser.add_argument('--plot_samples', default='no',   
                    help='conditional scale (default: no)')

parser.add_argument('--sampler', default='ddpm',   
                    help='sampler (default: ddpm). Supported samplers: ' + ', '.join(SAMPLERS))

parser.add_argument('--plot',  help='Generate plots', action="store_true")

parser.add_argument('--csv', help='Generate csv files', action="store_true")


def load_sample_classes(file, classes):
    # load pandas classes
    df = pd.read_csv(file)
    # keep only classes attributes
    df = df[classes + ['count']]
    # iterate over the dataframe
    class_vecs = []
    counts = []
    for i, row in df.iterrows():
        class_vecs.append(row[classes].values)
        counts.append(int(row['count']))

    return np.array(class_vecs), np.array(counts)

def main():
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    
    for name, val in vars(args).items():
        logger.info("{:<16}: {}".format(name, val))

    # Load config file
    cfg = OmegaConf.load(args.config)

    if hasattr(cfg, 'class_file') and hasattr(cfg, 'classes'):
        logger.info ("Class file: %s", cfg.class_file)
        logger.info ("Classes: %s", cfg.classes)

        samples, class_nums = load_classes(cfg.class_file, cfg.classes)   

        # if args.samples is a csv file
        # the file should be a json file with the following format:
        #   Subclass1, Subclass2, Subclass3, ..., count
        #   0, 1, 0, 1, 0, 1,
        #   1, 0, 1, 0, 1, 2,
        #   ...
        # each row is a class (label) and the value is the number of samples to generate
        # if args.samples is a number, then it is the number of samples to generate
        # in this case, the samples are drawn from the training distribution of classes
        
        if os.path.exists(args.samples):
            # load the class per sample
            vecs, counts = load_sample_classes(args.samples, cfg.classes)

            sampled_classes = torch.tensor(np.repeat(vecs, counts, axis=0)).to(cfg.device)
            args.num = sampled_classes.shape[0]
        else:
            # Generate samples from the distribution of classes
            sample_num = int(args.samples)
            logger.info (f"===> Generating {sample_num} samples from the training distribution of classes...")
         
            # build the distribution of classes: 
            # (1) convert the class vectors of training samples to class indices
            # (2) build the distribution of indices
            # (3) draw samples from the distribution
            # (4) convert the sampled class indices back to class labels

            class_nums = torch.tensor(class_nums)
            class_vecs = torch.tensor([v for _, v in samples.items()])

            # (1) convert to class indices
            class_idxs = get_class_index(class_vecs, class_nums)
            logger.info ("Number of samples: %d", len(class_idxs))
            logger.info ("Class nums: %s", class_nums)  
            total_classes = class_nums.prod().item()
            logger.info ("Number of all classes: %d", total_classes)

            # (3) draw samples
            hist = np.bincount(class_idxs.numpy(), minlength=total_classes)
            hist = hist / hist.sum()
            # TOP-5 classes with probability
            top_classes = np.argsort(hist)[::-1][:5]
            logger.info ("TOP-5 classes: %s Prob: %s", top_classes, hist[top_classes])
            # sample indices
            sample_idxs = np.random.choice(total_classes, sample_num, p=hist)
            sampled_classes = get_class_vector(torch.tensor(sample_idxs), class_nums).long().to(cfg.device) 
            logger.info ("Randomly sampled indices: %s", sample_idxs) 
            logger.debug ("Corresponding class vectors: %s", sampled_classes)
            args.num = sampled_classes.shape[0]
    else:
        logger.info ("*** No class file provided!!!")
        class_nums = None
        sampled_classes = None

    batch_size = min(20, args.num)
    logger.info ("Batch size: %d", batch_size)

    # data is already normalized into [0, 1], we need the normalizing factors to dispay generated data
    logger.info("FHR max: %f", cfg.scale_factors.fhr_max)
    logger.info("FHR min: %f", cfg.scale_factors.fhr_min)
    logger.info("UC max: %f", cfg.scale_factors.uc_max)
    logger.info("UC min: %f", cfg.scale_factors.uc_min)

    assert not os.path.exists(cfg.ldm.model_path), "Error: LDM model does not exists."
   
    # create output directory
    create_dir(args.out_dir)
    _train_cfg = cfg.ldm.training

    first_stage_model, in_channel = load_first_stage(cfg.first_stage_model, cfg.out_dir, cfg.input_size, cfg.in_channels)
    ldm = LDM(cfg.device, accelerator=None, first_stage_model=first_stage_model, 
                        in_channel=in_channel,
                        class_nums=class_nums,
                        **_train_cfg)
    
    # load model
    ldm.load(f"{cfg.out_dir}/{cfg.ldm.model_path}")
    
    # if sr3 model is in the config file
    if hasattr(cfg, 'sr_model'):
        _train_cfg = cfg.sr_model.training
     
        transform_model = first_stage_model
        first_stage_model, in_channel = load_first_stage(cfg.sr_model.first_stage_model, cfg.out_dir, cfg.input_size, cfg.in_channels)   
   
        sr_model = SR3(cfg.device, 
                accelerator=None,
                transform_model=transform_model,
                first_stage_model=first_stage_model,
                in_channel=in_channel,
                class_nums=class_nums,
                **_train_cfg)
        
        sr_model.load(f"{cfg.out_dir}/{cfg.sr_model.model_path}")

    # create a new HDF5 file
    hdf5_file = tables.open_file(os.path.join(args.out_dir, args.out_h5), mode="w")
    # store windows in a new group
    group = hdf5_file.create_group("/", "samples", "Generated samples")
    # create an array to store the windows with blosc compression, high compression level
    out_array = hdf5_file.create_earray(group, "windows", tables.Float16Atom(), filters=tables.Filters(complib="blosc", complevel=7), shape=(0, cfg.input_size, 2))
    # create an array to store the labels with blosc compression, label is patient id
    out_labels = hdf5_file.create_earray(group, "labels", tables.Int32Atom(),filters=tables.Filters(complib="blosc"), shape=(0,))
    
    # Generate BATCH_SIZE samples at a time
    total = 0
    all_classes = []
    while total != args.num:
        logger.info (f"--- Generated samples: {total} ---")
        
        sample_num = batch_size if total + batch_size <= args.num else args.num - total

        sampled_classes_idxs = sampled_classes[total:total + sample_num] if exists(sampled_classes) else None

        logger.info (f"Latent Diffusion generation ({sample_num} samples)...")
        # generated ts has shape (batch_size, channels, time) and scaled to [-1, 1]
        if exists(sampled_classes_idxs):
            ts, classes = ldm.synthetise(cond_scale=args.cond_scale, sample_num=sample_num, classes=sampled_classes_idxs, steps=args.steps, sampler=args.sampler)   
        else:
            ts, classes = ldm.synthetise(sample_num=sample_num, steps=args.steps)

        # append the generated samples to the HDF5 file
        # transpose the samples to have time as the first dimension
        
        out_labels.append(np.arange(total, total + sample_num))   
        all_classes.extend(classes.cpu().detach().numpy())      
      
        if hasattr(cfg, 'sr_model'):
            logger.info("Upsampling the generated samples...")
            ts = sr_model.super_resolution(ts.to(cfg.device), cond_scale=args.cond_scale, classes=sampled_classes_idxs, sampler=args.sampler, steps=args.steps)   

        # store generated sample in training data format for evaluation:
        # scaled into (-1, 1) with shape (batch_size, time, channels)
        out_array.append(ts.transpose(2, 1).cpu().detach().numpy())

        ts = unscale(ts, cfg.scale_factors).cpu().detach().numpy()

        # save signals doesn't scale the data, and expects the shape (batch_size, channels, time)
        if args.plot:
            save_signals(ts, f'{args.out_dir}/samples_batch_{b}.jpg', classes=classes, cols=4)  
       
        # Save the samples to csv files
        if args.csv:
            ts_len = ts.shape[2]
            # transpose the samples to have time as the first dimension (for pandas)
            ts = np.transpose(ts, (0, 2, 1))
            for i in range(total, total + ts.shape[0]):
                df = pd.DataFrame(ts[i % batch_size], columns = ['FHR', 'UC'])
                # add column time as first column
                df.insert(0, 'Time', np.arange(0, ts_len, 1))
                # save the dataframe
                sub_dir = args.out_dir
                if exists(classes):
                    sub_dir += f"/class_{classes[i % batch_size]}"

                create_dir(sub_dir)
                df.to_csv(f"{sub_dir}/sample_{i}_{ts_len}.csv", index=False)

                #df = pd.DataFrame(ts[i % batch_size], columns = ['FHR', 'UC'])
                # add column time as first column
                #df.insert(0, 'Time', np.arange(0, ts_len, 1))
                # save the dataframe
                #df.to_csv(f"{args.out_dir}/sample_interp_{i}_{ts_len}.csv", index=False)

        total += ts.shape[0]

    # output classes to csv file, index is the sample number
    df = pd.DataFrame(all_classes, columns=cfg.classes)
    # add sample number as first column
    df.insert(0, 'ID', range(args.num))
    df.set_index('ID', inplace=True)
    df.to_csv(f"{args.out_dir}/generated_labels.csv")
    
    hdf5_file.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Exit training with keyboard interrupt!")
        sys.exit(0)

   