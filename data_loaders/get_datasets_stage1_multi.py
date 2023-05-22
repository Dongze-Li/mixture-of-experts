import glob
import numpy
import random
from functools import reduce
import matplotlib.pyplot as plt
from operator import concat
from torchvision import transforms as TR
from torchvision import datasets, transforms
from torch.utils.data import ChainDataset, DataLoader, Sampler, Subset
from torch.utils.data.dataset import ConcatDataset
import torchvision
import pandas as pd
import torch
import math
from copy import copy
import importlib
import yaml


class get_datasets_stage1_multi(object):

    def __init__(self, config):

        config_file = config
        with open(config_file) as cf_file:
            self.config = yaml.safe_load( cf_file.read())

        batch_size = self.config['stage_1']['batchsize']

        # faces_dataset = importlib.import_module('data_loaders.' + config['stage_1']['datasets']['faces'])
        # faces_train = faces_dataset.dataset_train
        # faces_val = faces_dataset.dataset_val
        # faces_test = faces_dataset.dataset_test
        face_path = importlib.import_module('data_loaders.' + self.config['stage_1']['datasets']['faces'])
        data_face = getattr(face_path, self.config['stage_1']['datasets']['faces'])
        faces_dataset = data_face(config_file)

        faces_train = faces_dataset.getDataset("train")
        faces_val = faces_dataset.getDataset("val")
        faces_test = faces_dataset.getDataset("test")


        # objects_dataset = importlib.import_module('data_loaders.' + config['stage_1']['datasets']['objects'])
        # objects_train = objects_dataset.dataset_train
        # objects_val = objects_dataset.dataset_val
        # objects_test = objects_dataset.dataset_test
        object_path = importlib.import_module('data_loaders.' + self.config['stage_1']['datasets']['objects'])
        data_object = getattr(object_path, self.config['stage_1']['datasets']['objects'])
        object_dataset = data_object(config_file)

        objects_train = object_dataset.getDataset("train")
        objects_val = object_dataset.getDataset("val")
        objects_test = object_dataset.getDataset("test")


        # words_dataset = importlib.import_module('data_loaders.' + config['stage_1']['datasets']['words'])
        # words_train = words_dataset.dataset_train
        # words_val = words_dataset.dataset_val
        # words_test = words_dataset.dataset_test

        word_path = importlib.import_module('data_loaders.' + self.config['stage_1']['datasets']['words'])
        data_word = getattr(word_path, self.config['stage_1']['datasets']['words'])
        words_dataset = data_word(config_file)

        words_train = words_dataset.getDataset("train")
        words_val = words_dataset.getDataset("val")
        words_test = words_dataset.getDataset("test")


        dataset_combined_train = ConcatDataset((words_train, objects_train, faces_train))
        dataset_combined_val = ConcatDataset((words_val, objects_val, faces_val))
        dataset_combined_test = ConcatDataset((words_test, objects_test, faces_test))

        work = 1
        mem = False
        time = 10
        print("num_workers = ", work)
        self.dataloader_train = DataLoader(dataset_combined_train, batch_size=batch_size, shuffle=True, num_workers=work, pin_memory=mem, timeout=time)
        self.dataloader_val = DataLoader(dataset_combined_val, batch_size=batch_size, shuffle=True, num_workers=work, pin_memory=mem, timeout=time)
        self.dataloader_test = DataLoader(dataset_combined_test, batch_size=batch_size, shuffle=True, num_workers=work, pin_memory=mem, timeout=time)

        print("dataset_combined_train size:" + str(dataset_combined_train.cumulative_sizes[-1]))
        print("dataset_combined_val size:" + str(dataset_combined_val.cumulative_sizes[-1]))
        print("dataset_combined_test size:" + str(dataset_combined_test.cumulative_sizes[-1]))


    def getDataloader(self, split):
        if split == "train":
            return self.dataloader_train
        elif split == "val":
            return self.dataloader_val
        elif split == "test":
            return self.dataloader_test
        else:
            raise RuntimeError("Please request for a valid split: train, val , test. ")
