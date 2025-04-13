# TODO: prtorch lightning does not save the hparams in the checkpoint.
# cannot continue training from the saved checkpoint
# see: class_idx, saved epoch number

from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score
import torch
import pytorch_lightning as pl
#import numpy as np
#from utils import get_class_index
import einops
from collections import Counter

from evaluation.model.fcn import FCNBaseline
import logging

logger = logging.getLogger(__name__)

class ExpFCN(pl.LightningModule):
    def __init__(self,
                 config: dict,
                 n_train_samples: int,
                 class_nums: int,
                 class_idx: int,
                 downsampling_rate : int = 1,
                 iterations_per_epoch: int = 0
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.class_nums = class_nums
        self.class_idx = class_idx
        self.iterations_per_epoch = iterations_per_epoch
        self.downsampling_rate = downsampling_rate
      
        #self.fcn = FCNBaseline(in_channels, np.cumprod(class_nums)[-1])
        logger.info (f"Class idx: {class_idx}; Class nums: {class_nums}")
        logger.info (f"Downsampling rate: {downsampling_rate}")
        self.fcn = FCNBaseline(config.in_channels, self.class_nums)
       
        self.criterion = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()

        # replace every s consecutve samples with their mean
        x = einops.reduce(x, 'b c (l s) -> b c l', 'mean', s=self.downsampling_rate)

        y = y[:, self.class_idx]
        
        #cnt = Counter(y.flatten().cpu().detach().numpy())
        #print (f"Class {self.class_idx}: {cnt}")

        #y = get_class_index(y, self.class_nums)
     
        yhat = self.fcn(x)  # (b n_classes)
        loss = self.criterion(yhat, y)

        # lr scheduler
        sch = self.lr_schedulers()
        sch.step()

        # log
        acc = accuracy_score(y.flatten().detach().cpu().numpy(),
                             yhat.argmax(dim=-1).flatten().cpu().detach().numpy())
        loss_hist = {'loss': loss, 'acc': acc}

        for k, v in loss_hist.items():
            self.log(f'train/{k}', v)

        return loss_hist

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y[:, self.class_idx]

        # replace every s consecutve samples with their mean
        x = einops.reduce(x, 'b c (l s) -> b c l', 'mean', s=self.downsampling_rate)

        #y = get_class_index(y, self.class_nums)
      
        yhat = self.fcn(x)  # (b n_classes)
      
        loss = self.criterion(yhat, y)

        # log
        acc = accuracy_score(y.flatten().detach().cpu().numpy(),
                             yhat.argmax(dim=-1).flatten().cpu().detach().numpy())
        loss_hist = {'loss': loss, 'acc': acc}

        for k, v in loss_hist.items():
            self.log(f'val/{k}', v)

        return loss_hist

    def configure_optimizers(self):
        opt = torch.optim.AdamW([{'params': self.parameters(), 'lr': self.config.lr}], )
        T_max = self.iterations_per_epoch * self.config.epochs
        return {'optimizer': opt, 'lr_scheduler': CosineAnnealingLR(opt, T_max, eta_min=0.000001)}

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        x = x.float()

        # replace every s consecutve samples with their mean
        x = einops.reduce(x, 'b c (l s) -> b c l', 'mean', s=self.downsampling_rate)

        y = y[:, self.class_idx]

        # check distribution of the classes
        
        #y = get_class_index(y, self.class_nums)
     
        yhat = self.fcn(x)  # (b n_classes)
        loss = self.criterion(yhat, y)

        # log
        yhat = yhat.argmax(dim=-1).flatten().cpu().detach().numpy()
        y = y.flatten().cpu().detach().numpy()
        acc = accuracy_score(y, yhat)
        loss_hist = {'loss': loss, 'acc': acc}

        for k, v in loss_hist.items():
            self.log(f'val/{k}', v)

        return loss_hist
