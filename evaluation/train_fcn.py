from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import numpy as np
from omegaconf import OmegaConf
import os
from torch.utils.data import DataLoader
import torch
import sys
from multiprocessing import cpu_count

from evaluation.model.exp_train import ExpFCN
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

from dataloader.loader import PyTablesDataset
import logging

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s: %(message)s')

np.set_printoptions(legacy='1.25')

def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to the config data  file.",
                        default='configs/eval.yaml')
    # parser.add_argument('--gpu_device_ind', nargs='+', default=[0], type=int)
    return parser.parse_args(args=None if sys.argv[1:] else ['--help'])

if __name__ == '__main__':
    # load config
    args = load_args()

    cfg = OmegaConf.load(args.config)
    train_cfg = OmegaConf.load(cfg.train_cfg)
    cfg.in_channels = train_cfg.in_channels 
      
    # load training data: (sample_num, time, channels)
    dataset = PyTablesDataset(train_cfg.training_data, transpose=True, class_file=train_cfg.class_file, classes=train_cfg.classes)
   
    # classifiers are trained for each class on the training data
    # Evaluation is done on the validation data
    train_idx, val_idx = train_test_split(np.arange(len(dataset)), test_size=cfg.validation_size, random_state=cfg.random_seed)

    #train_idx = train_idx[:1000]
    #val_idx = val_idx[:500]

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)

    logger.info (f"Training samples: {len(train_dataset):,}; Validation samples: {len(val_dataset):,}")
    # label distirbution
    
    batch_size = cfg.batch_size

    # only 10 samples for testing 
    trainloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=cpu_count() - 1, shuffle=True)
    testloader = DataLoader(val_dataset, num_workers=cpu_count() - 1, batch_size=batch_size)

    logger.info (f"Iterations per epoch (batch_size={cfg.epochs}): %s", len(trainloader))
    
    # train a classifier for each class defined in the training data
    for class_idx, class_name in enumerate(dataset.classes):
        logger.info (f"Class: {class_name}; Idx: {class_idx}")
        
        model_name = cfg.classifier_name + f'-{class_name}'

        # 1 Hz data should be enough for classification
        # SOTE: we have 4 Hz sampling rate -> downsampling_rate=4 to have 1 Hz
        # Czech: 1Hz -> downsampling_rate=1 
        train_exp = ExpFCN(cfg, 
                           len(train_dataset), 
                           dataset.class_nums[class_idx], 
                           class_idx, 
                           downsampling_rate=train_cfg.freq, 
                           iterations_per_epoch=len(trainloader))
    
        wandb_config = { 'batch_size': batch_size, 'epochs': cfg.epochs, 'learning_rate': cfg.lr, 'validation_size': cfg.validation_size }

        logger_wandb = WandbLogger(project='CTG-FCN-eval', name=model_name, config=wandb_config)

        # save best model 
        checkpoint_callback = ModelCheckpoint(
            monitor='val/loss',
            dirpath=cfg.classifier_dir, 
            save_top_k=1,
            filename=model_name
        )   
    
        # if checkpoint exists, load the model
        model_path = os.path.join(cfg.classifier_dir, model_name + '.ckpt')
        logger.info ("Model path: %s", model_path)
        if os.path.exists(model_path):
            logger.info ("Loading model from checkpoint: %s", model_name)
            train_exp = ExpFCN.load_from_checkpoint(model_path)
            _train_args = {'ckpt_path': model_path}
        else:
            logger.info ("Training model from scratch.")
            _train_args = {}

        trainer = pl.Trainer(logger=logger_wandb, 
                                callbacks=[checkpoint_callback,
                                LearningRateMonitor(logging_interval='step'), 
                                EarlyStopping(monitor='val/loss', mode='min', patience=10, verbose=False)],
                                accelerator=cfg.accelerator, devices=cfg.devices, max_epochs=cfg.epochs, check_val_every_n_epoch=1)
            
        trainer.fit(train_exp,
                    train_dataloaders=trainloader,
                    val_dataloaders=testloader,
                    **_train_args
                    )
        
        # test
        trainer.test(train_exp, testloader)
        logger_wandb.experiment.finish()

    
 

