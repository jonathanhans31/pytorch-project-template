# Initialize paths
import os
import sys
import yaml
import json
import pickle as pkl
import torch
PATHS = yaml.full_load(open("../../paths.yaml"))
for k in PATHS: sys.path.append(PATHS[k])
import argparse

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')
    
def load_file(path):
    loader = {"yaml": yaml_load,
              "pkl": pkl_load,
              "json": json_load}

    ext = os.path.basename(path).split(".")[-1]
    if ext not in loader:
        raise RuntimeError("File extension is not supported by loader")

    return loader[ext](path)

def json_load(path):
    return json.load(open(path))

def yaml_load(path):
    return yaml.full_load(open(path))

def pkl_load(path):
    return pkl.load(open(path, "rb"))


def to_numpy(x):
    # x is already a numpy array
    if type(x) != type(torch.tensor(0)): 
        return x
    
    if x.is_cuda:
        return x.data.cpu().numpy()
    return x.data.numpy()

def to_tensor(x):
    if type(x) != type(torch.tensor(0)): 
        x = torch.tensor(x)
    if torch.cuda.is_available():
        return x.cuda()
    return x
