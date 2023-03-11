"""
This file contains the definition of different heterogeneous datasets used for training
"""
import torch
import numpy as np

from .base_dataset import BaseDataset

class MixedDataset(torch.utils.data.Dataset):

    def __init__(self, options, **kwargs):
        self.options = options
        if options.ft_dataset != '':
            ft_dataset = options.ft_dataset.split(',')
            options.openpose_train_weight = float(options.openpose_train_weight)
            options.gt_train_weight = float(options.gt_train_weight)
            # assert options.openpose_train_weight == 1. and options.gt_train_weight == 0
            self.dataset_list = ft_dataset
            self.dataset_dict = {ds: i for i, ds in enumerate(self.dataset_list)}
            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            self.partition = [1 / len(self.datasets) for _ in self.datasets]
        else:
            # use all the datasets in SPIN during pretraining.
            self.dataset_list = ['h36m', 'lsp-orig', 'mpii', 'lspet', 'coco', 'mpi-inf-3dhp']
            self.dataset_dict = {'h36m': 0, 'lsp-orig': 1, 'mpii': 2, 'lspet': 3, 'coco': 4, 'mpi-inf-3dhp': 5}
            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.partition = [.3, .6*len(self.datasets[1])/length_itw,
                 .6*len(self.datasets[2])/length_itw,
                 .6*len(self.datasets[3])/length_itw, 
                 .6*len(self.datasets[4])/length_itw,
                 .1]
            
        total_length = sum([len(ds) for ds in self.datasets])
        self.length = max([len(ds) for ds in self.datasets])
        """
        Data distribution inside each batch:
        30% H36M - 60% ITW - 10% MPI-INF
        """
        self.partition = np.array(self.partition).cumsum()

    def __getitem__(self, index):
        # if self.options.ft_dataset != '':
            # return self.datasets[0][index]
    
        p = np.random.rand()
        for i in range(len(self.partition)):
            if p <= self.partition[i]:
                return self.datasets[i][index % len(self.datasets[i])]
            
            
    def __len__(self):
        return self.length
