import hydra
from omegaconf.dictconfig import DictConfig
import argparse
import numpy as np
import torch
import sys
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
args = parser.parse_args()

@hydra.main(config_path='conf', config_name='config')
def get_args_global(conf):
    for key in conf.keys():
        args.__setattr__(key, conf[key])
@hydra.main(config_path='conf', config_name='config-dataset')
def get_args_dataset(conf):
    params = conf[args.dataset]
    for key in params.keys():
        args.__setattr__(key, params[key])
@hydra.main(config_path='conf', config_name='config-model')
def get_args_model(conf):
    if args.model in conf:
        params = conf[args.model][args.dataset]
        for key in params.keys():
            args.__setattr__(key, params[key])
get_args_global()
get_args_dataset()
# get_args_model()

args.cuda = torch.cuda.is_available()
device = torch.device('cuda' if args.cuda else 'cpu')
if args.cuda:
    torch.cuda.set_device(args.gpu_id)
args.device = device
print(args)

import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)