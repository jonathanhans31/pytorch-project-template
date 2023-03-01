import os 
from pathlib import Path
import argparse
import torch
import numpy as np
import mlflow as mf
import yaml
PATHS = yaml.full_load(open("../../paths.yaml"))

def init_mlflow(exp_name, run_name):
    uri_path = PATHS["EXP"]
    try:
        # For linux this line should work just fine
        mf.set_tracking_uri(uri_path)
        mf.set_experiment(exp_name)
        mf.start_run(run_name=run_name)
    except:
        # This line is required for windows to work
        mf.set_tracking_uri("file://"+uri_path)
        mf.set_experiment(exp_name)
        mf.start_run(run_name=run_name)
    mf.set_tag("status", "started")
        
def init_dirs(config):
    active_run = mf.active_run()
    config["root_dir"] = PATHS["EXP"] + "/{}/{}/artifacts/".format(active_run.info.experiment_id, active_run.info.run_id)
    for item in ["model_dir","sample_dir","log_dir"]:
        config[item] = config["root_dir"] + config[item]
        Path(config[item]).mkdir(parents=True, exist_ok=True)
    return config

def set_device(gpu_id):
    # Manage GPU availability
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    if gpu_id != "": 
        torch.cuda.set_device(0)
        
    else:
        n_threads = torch.get_num_threads()
        n_threads = min(n_threads, 8)
        torch.set_num_threads(n_threads)
        print("Using {} CPU Core".format(n_threads))
