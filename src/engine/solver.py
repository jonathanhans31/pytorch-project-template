'''
Solver is used to instantiate the dataloader and model. 
It also handles all interactions between the dataloader and model, as well as, logging.

@Author: Jonathan Hans Soeseno
'''

import sys
import yaml
PATHS = yaml.safe_load(open("../../paths.yaml"))
for k in PATHS: sys.path.append(PATHS[k])

import numpy as np
import mlflow as mf
import cv2
from tqdm import tqdm
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import DataLoader


from dataloader import DummyDataset
from model import DummyModel

class Solver:
    DEFAULTS = {}   
    def __init__(self, config):
        self.__dict__.update(self.DEFAULTS, **config)
    
        random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        self.init_modules()
    
    def _init_model(self):
        self.model = DummyModel(self.model_params)
        
    def _init_loaders(self):
        self.train_loader = DataLoader(dataset=DummyDataset(self.loader_params["num_data"]), 
                                        batch_size=8, 
                                        shuffle=True)
        self.test_loader = DataLoader(dataset=DummyDataset(self.loader_params["num_data"]), 
                                        batch_size=32, 
                                        shuffle=False)

    def _init_optimizer(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.train_params["lr"])

    def init_modules(self):
        self._init_loaders()
        self._init_model()
        self._init_optimizer()

    def train(self,):
        self.model.train()
        num_epochs = self.train_params["num_epochs"]

        for epoch in tqdm(range(num_epochs)):

            losses = []
            for batch in self.train_loader:
                output_dict = self.model(batch)

                loss = output_dict["loss_l2"]
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())
            # Logging
            if epoch % 10 == 0:
                print("Epoch {}/{} | L2_loss: {:.4f}".format(epoch, num_epochs, np.mean(losses)))
            mf.log_metric("l2_loss", np.mean(losses), step=epoch)

        print("=== Finished ===")
        print("Epoch {}/{} | L2_loss: {:.4f}".format(epoch, num_epochs, np.mean(losses)))
        mf.log_metric("l2_loss", np.mean(losses), step=epoch)

    def inference(self, loader):
        self.model.eval()
        for batch in self.test_loader:
            output_dict = self.model(batch)

    def test(self):
        # Do some other things
        self.inference(self.train_loader)
        self.inference(self.test_loader)