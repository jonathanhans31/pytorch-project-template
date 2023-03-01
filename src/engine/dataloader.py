import numpy as np
import torch
import torchvision
import cv2
import json
import os
from PIL import Image
from torch.utils.data import DataLoader

class DummyDataset:
    def __init__(self, n_data):
        self._x = np.random.randn(n_data, 32).astype("float32")
        self._y = np.random.randn(n_data, 1).astype("float32")
        self.N = n_data

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return {"x": torch.tensor(self._x[index]), 
                "y": torch.tensor(self._y[index])}
