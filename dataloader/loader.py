import tables
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils import exists, load_classes

# create a dataloader class for the training data from tables file
# change channel for torch:
# (batch_size, time, channels) -> (batch_size, channels, time)

from collections import Counter

import logging

logger = logging.getLogger(__name__)

class PyTablesDataset(Dataset):
    def __init__(self, path, transpose=True, class_file=None, classes=None):   
        self.h5file = tables.open_file(path, mode='r')
        self.data = self.h5file.root.samples.windows
        self.user_id = self.h5file.root.samples.labels
        self.transpose = transpose
        self.classes = classes
        logger.info (f"=== PyTablesDataset: {path} ===")
        logger.info ("* Data shape: %s", self.data.shape)
        logger.info ("* Labels: %s", class_file)
        # if labels is not a list then exit
        if exists(class_file):      
            self.labels, self.class_nums = load_classes(class_file, classes)
            # get the allowed ids: only those with labels
            self.allowed_ids = [i for i, uid in enumerate(self.user_id) if uid in self.labels]

            cnt = Counter(self.labels.values())
            logger.info ("* Number of patients in classes file: %s", len(self.labels))
            logger.info ("* Number of patients with CTG: %s", np.unique(self.user_id).shape[0])
            logger.info ("* Number of patients with labelled CTG: %s", np.intersect1d(self.user_id, list(self.labels.keys())).shape[0])
            logger.info ("* Number of labelled records (windows): %s", len(self.allowed_ids))    
            logger.info ("* Number of occuring classes: %s", len(cnt))
            #logger.info ("* Class distribution:", cnt)
            logger.info ("* Number of subclasses: %s", len(self.class_nums))
            logger.info ("* Values per subclass: %s", self.class_nums)
        else:
            # if no classes are provided then use all data
            self.allowed_ids = range(len(self.data))
            self.class_nums = None
            self.labels = None
           
    def __len__(self):
        return len(self.allowed_ids)
    
    def __getitem__(self, idx):
        idx = self.allowed_ids[idx]
        record = self.data[idx].T.astype(np.float32) if self.transpose else self.data[idx].astype(np.float32)
        x = torch.tensor(record)
    
        if not exists(self.labels) or self.user_id[idx] not in self.labels:
            y = torch.tensor([]).long()
        else:
            y = torch.tensor(self.labels[self.user_id[idx]]).long()
       
        return x.clone(), y.clone()
        
    def close(self):
        self.h5file.close()
    
# same for numpy file
class NumpyDataset(Dataset):
    def __init__(self, path_data, path_label, transpose=True):  
        self.data = np.load(path_data).astype(np.float32)
        if transpose:
            self.data = self.data.transpose(0, 2, 1)
        self.labels = np.load(path_label)
        self.transpose = transpose  

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx])
    
# main function to load data

if __name__ == '__main__':
    path_hdf5 = 'training_data/czech_proc.h5'
    batch_size = 32
     
    # dataset = NumpyDataset(path_data, path_label)
    dataset = PyTablesDataset(path_hdf5)

    from sklearn.model_selection import train_test_split
    from tqdm import tqdm

    # split idxs into training and validation with 80% and 20% using scikit
    train_idx, val_idx = train_test_split(np.arange(len(dataset)), test_size=0.2)
    
    # create two subsets
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    
    batch_size = 16

    dataloader = DataLoader(train_dataset, batch_size=batch_size)
    
    for data in tqdm(dataloader):
        pass

