import glob
import numpy as np
import random
from functools import reduce
import matplotlib.pyplot as plt
from operator import concat
from torchvision import transforms as TR
from torchvision import datasets, transforms
from torch.utils.data import ChainDataset, DataLoader, Sampler
from torch.utils.data.dataset import ConcatDataset
import torchvision
import torch
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict
import yaml
import pandas as pd
import importlib
from Experiment1 import *
from Experiment2 import *
import sys

    
if __name__ == "__main__":
    exp_name = 'task-1-default-config'

    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(config_file) as cf_file:
        config = yaml.safe_load( cf_file.read())
        
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)
    
    # get the datasets
    datasets_path = importlib.import_module('data_loaders.'+config['stage_1']['datasets']['combined'])
    datasets_class = config['stage_1']['datasets']['combined']
    datasets_file = getattr(datasets_path, datasets_class)

    dataloader_buffer = datasets_file(config_file)
    dataloader_train = dataloader_buffer.getDataloader("train")
    dataloader_val = dataloader_buffer.getDataloader("val")
    dataloader_test = dataloader_buffer.getDataloader("test")
    
    # run the model
    model = Experiment1(config_file)
    model.training(dataloader_train,dataloader_val)
    model.test(dataloader_test)
    