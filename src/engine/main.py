'''
The main file
@Author: Jonathan Hans Soeseno
'''
# Initialize paths
import sys
import yaml
PATHS = yaml.safe_load(open("../../paths.yaml"))
for k in PATHS: sys.path.append(PATHS[k])

import os 
import argparse

from solver import Solver
from utils.general import str2bool
from utils.run_helper import init_mlflow, set_device, init_dirs

import mlflow as mf
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 

def main(args):        
    # Load config file
    if os.path.exists(PATHS["CONFIG"] + args.config_file) == False:         
        raise RuntimeError("config_file {} does not exist".format(args.config_file))
    config = yaml.safe_load(open(PATHS["CONFIG"] + args.config_file))  

    if args.debug:
        config["exp_name"] = "Debug"
        print("Running {} on Debug mode!".format(args.config_file))
    
    # Initialize the experiment
    init_mlflow(config["exp_name"], config["run_name"])

    # Setup directories and modify directories
    config = init_dirs(config)  
    set_device(args.gpu_id)

    # Store the config file and params
    yaml.dump(config, open(config["root_dir"]+"config.yaml","w"))

    # Run evaluations
    
    solver = Solver(config)
    if args.mode == "train":       
        solver.train()

    mf.set_tag("status", "finished")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True)        
    parser.add_argument('--gpu_id', type=str, default="0")
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument("--debug", type=str2bool, default=False)
    args = parser.parse_args()
    main(args)
