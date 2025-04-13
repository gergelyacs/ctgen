"""
It evaluates the generated data using the FID and IS metrics.
"""
from evaluation.metrics import fid, inception_score
import torch
import numpy as np
from evaluation.model.exp_train import ExpFCN
from argparse import ArgumentParser
from omegaconf import OmegaConf
from tqdm import tqdm
import os
from utils import exists
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s: %(message)s')

logger = logging.getLogger(__name__)

np.set_printoptions(legacy='1.25')

from dataloader.loader import PyTablesDataset

def compute_fid_is(model, x_real, x_syn, batch_size, downsample_dim=None):
    z_real = torch.tensor([])
    z_syn = torch.tensor([])
    p_yx = torch.tensor([])

    # get model device
    device = next(model.parameters()).device
   
    # create dataloader
    dataloader = torch.utils.data.DataLoader(x_real, batch_size=batch_size, shuffle=False)

    for x, _ in tqdm(dataloader, desc="Real data"):
        x = x.to(device)
        z_real = torch.cat((z_real, model(x).detach().cpu()), 0)

    # create dataloader
    dataloader = torch.utils.data.DataLoader(x_syn, batch_size=batch_size, shuffle=False)

    for x, _ in tqdm(dataloader, desc="Synthetic data"):
        if exists(downsample_dim):
            sample_rate = x.shape[1] // downsample_dim
            # keep every sample_rate-th sample
            x = x[..., ::sample_rate]
        x = x.to(device)
        out = model(x).detach().cpu()
        # FID:
        z_syn = torch.cat((z_syn, out), 0)
        # IS:
        p_yx = torch.cat((p_yx, torch.nn.functional.softmax(out, dim=-1)), 0)

    return fid(z_real.numpy(), z_syn.numpy()), inception_score(p_yx.numpy())[0] # return only the mean


def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to the config data file.",
                        default='configs/eval.yaml')
    return parser.parse_args(args=None if sys.argv[1:] else ['--help'])

if __name__ == '__main__':
    # load config
    args = load_args()

    for name, val in vars(args).items():
        logger.info("{:<16}: {}".format(name, val))

    cfg = OmegaConf.load(args.config)

    train_cfg = OmegaConf.load(cfg.train_cfg)

    # load training data: (sample_num, time, channels)
    real_dataset = PyTablesDataset(train_cfg.training_data, transpose=True, class_file=train_cfg.class_file, classes=train_cfg.classes)
    real_class_nums = real_dataset.class_nums
    real_dataset_size = len(real_dataset)

    train_idx, val_idx = train_test_split(np.arange(real_dataset_size), test_size=cfg.validation_size, random_state=cfg.random_seed)

    # evaluate on validation data
    # It should be used for FID computation
    val_dataset = torch.utils.data.Subset(real_dataset, val_idx)
    testloader = DataLoader(val_dataset, batch_size=1)#cfg.batch_size)

    # select random samples
    real_idx = np.random.choice(real_dataset_size, min(cfg.sample_num, real_dataset_size), replace=False)
    
    # load syn data: (sample_num, time, channels)
    # it must have the same classes as the real data
    syn_data_path = os.path.join(cfg.syn_dir, cfg.syn_data_file)
    syn_class_file = os.path.join(cfg.syn_dir, cfg.syn_class_file)
    syn_dataset = PyTablesDataset(syn_data_path, transpose=True, class_file=syn_class_file, classes=train_cfg.classes)
  
    downsample_dim = None
    if real_dataset.data.shape[1] != syn_dataset.data.shape[1]:
        logger.info ("Real and synthetic data have different dimensions.")
        logger.info ("Real data dimension: %s", real_dataset.data.shape)
        logger.info ("Synthetic data dimension: %s", syn_dataset.data.shape)
        logger.info ("Synthetic data will be downsampled to the real data dimension!")
        downsample_dim = real_dataset.data.shape[1] 

    # select random samples
    syn_idx = np.random.choice(len(syn_dataset), min(cfg.sample_num, len(syn_dataset)), replace=False)
    
    real_dataset = torch.utils.data.Subset(real_dataset, real_idx)
    syn_dataset = torch.utils.data.Subset(syn_dataset, syn_idx)

    syn_loader = torch.utils.data.DataLoader(syn_dataset, batch_size=1, shuffle=False)

    # loop on syn data
    #for (x, y) in testloader:
    #    print (f"Real data shape: {x.shape}, {y.shape}")
    #    print (f"Real data class: {y}")

    #for (x, y) in syn_loader:
    #    print (f"Syn data shape: {x.shape}, {y.shape}")
    #    print (f"Syn data class: {y}")

    # load checkpoint
    scores = {class_name: {'acc_real': 0, 'acc_syn' : 0, 'fid': 0, 'is': 0} for class_name in train_cfg.classes}

    # check with model for each class
    for class_idx, class_name in enumerate(train_cfg.classes):
        logger.info (f"===> Class: {class_name}")
        model_name = cfg.classifier_name + f'-{class_name}'
        model = ExpFCN.load_from_checkpoint(os.path.join(cfg.classifier_dir, model_name + '.ckpt'))
        # this is important: set the correct class index
        model.class_idx = class_idx
        #logger.info (model.class_nums, real_class_nums[class_idx])
        ##assert model.class_nums == real_class_nums[class_idx], f"Error: Class numbers do not match! (Real: {real_class_nums[class_idx]}, Model: {model.class_nums})"
        model.eval()

        # evaluate the model on validation data
        trainer = pl.Trainer(accelerator=cfg.accelerator, devices=cfg.devices)

        # test 
        res_real, res_syn = trainer.test(model, [testloader, syn_loader])
        acc_real = res_real['val/acc/dataloader_idx_0']
        logger.info ("Prediction accuracy on real validation data: %s", acc_real)
        scores[class_name]['acc_real'] = acc_real

        acc_syn = res_syn['val/acc/dataloader_idx_1']
        logger.info ("Prediction accuracy on synthetic data: %s", acc_syn)
        scores[class_name]['acc_syn'] = acc_syn

        # compute FID and IS
        
        fid_score, is_score = compute_fid_is(model.fcn.to("cuda"), real_dataset, syn_dataset, cfg.batch_size, downsample_dim)
        logger.info ("FID score: %s", fid_score)
        logger.info ("Inception score for generated data: %s", is_score)
        scores[class_name]['fid'] = fid_score
        scores[class_name]['is'] = is_score

    # logger.info statistics
    logger.info ("===> Evaluation results")
    for class_name, val in scores.items():
        logger.info (f"Classifier: {class_name}")
        logger.info (f"Prediction accuracy: real {val['acc_real']}, synthetic {val['acc_syn']}")
        logger.info (f"FID: {val['fid']}, IS: {val['is']}")
        logger.info ("-"*50)

    # avergae FID and IS
    fid_scores = [val['fid'] for val in scores.values()]
    is_scores = [val['is'] for val in scores.values()]

    # average, stddev, min, max. logger.info statistics single line with labels
    # FID: mean <mean>, stddev <stddev>, min <min>, max <max>
    # IS: mean <mean>, stddev <stddev>, min <min>, max <max>
    logger.info ("FID: mean {:.4f}, stddev {:.4f}, min {:.4f}, max {:.4f}".format(np.mean(fid_scores), np.std(fid_scores), np.min(fid_scores), np.max(fid_scores)))
    logger.info ("IS: mean {:.4f}, stddev {:.4f}, min {:.4f}, max {:.4f}".format(np.mean(is_scores), np.std(is_scores), np.min(is_scores), np.max(is_scores)))











    

