import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F

from archs.DummyNet import DummyNet

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

        # This self.engine can be defined as much more complex modules
        # self.engine = nn.Linear(input_dim, output_dim)
        self.engine = None

    def train_step(self, input_dict):
        '''
        # Parse inputs
        x = input_dict["x"]
        y = input_dict["y"]

        # Get model outputs
        pred = self.engine(x)

        # Compute losses
        assert y.shape == pred.shape
        loss_l2 = torch.mean((pred - y)**2)

        # Compose output_dict
        output_dict = {
            "loss_l2": loss_l2
        }
        return output_dict
        '''
        return {}

    def eval_step(self, input_dict):
        '''
        # Parse inputs
        x = input_dict["x"]

        # Get model outputs
        pred = self.engine(x)

        # Compose output_dict, sometimes redundancy helps with debugging
        output_dict = {
            "pred": pred,
            "y": input_dict["y"]
        }
        return output_dict        
        '''
        return {}

    def forward(self, input_dict):
        if self.training:
            # Returns losses under output_dict
            return self.train_step(input_dict)
        else:
            # Eval runs without gradient and returns predictions under output_dict
            with torch.no_grad():
                return self.eval_step(input_dict)


class DummyModel(nn.Module):
    def __init__(self, model_params):
        super(DummyModel, self).__init__()
        self.engine = DummyNet(input_dim = model_params["input_dim"], 
                                hidden_dim = model_params["hidden_dim"],
                                output_dim = model_params["output_dim"],
                                num_layers = model_params["num_layers"])

    def train_step(self, input_dict):
        
        # Parse inputs
        x = input_dict["x"]
        y = input_dict["y"]

        # Get model outputs
        pred = self.engine(x)

        # Compute losses
        assert y.shape == pred.shape
        loss_l2 = torch.mean((pred - y)**2)

        # Compose output_dict
        output_dict = {
            "loss_l2": loss_l2
        }
        return output_dict
        

    def eval_step(self, input_dict):
        # Parse inputs
        x = input_dict["x"]

        # Get model outputs
        pred = self.engine(x)

        # Compose output_dict, sometimes redundancy helps with debugging
        output_dict = {
            "pred": pred,
            "y": input_dict["y"]
        }
        return output_dict        
        
        

    def forward(self, input_dict):
        if self.training:
            # Returns losses under output_dict
            return self.train_step(input_dict)
        else:
            # Eval runs without gradient and returns predictions under output_dict
            with torch.no_grad():
                return self.eval_step(input_dict)