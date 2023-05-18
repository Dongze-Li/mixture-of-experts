from torchvision.datasets import VisionDataset
import torch
from PIL import Image
import os
import numpy as np


class CustomLeaf(VisionDataset):

    def __init__(self, dataset_path, split="train", transform=None, target_transform=None, size=None, num_classes=38):

        super(CustomLeaf, self).__init__(None, transform=transform, target_transform=target_transform)

        # make size configurable as an input parameter by variable factor
        self.size = size
        self.num_classes = num_classes
        self.dataset_path = dataset_path  # "/data" for project
        self.split = split
        if self.split == "train":
            self.start = 0
        elif self.split == "val":
            self.start = 100
        else:
            self.start = 120
            
        if self.split == "train":
            if size is None:
                self.size = 100
            elif size > 1000:
                raise RuntimeError("Training data out of bounce")
        else:
            if size is None:
                self.size = 20
            elif size > 100:
                raise RuntimeError("Data out of bounce")

        # go into the folder of split
        curr_file = os.path.abspath(os.path.realpath("CustomLeaf.py"))
        self._base_folder = os.path.join(os.path.dirname(curr_file), self.dataset_path, "plant_leaf")
        self._images_folder = os.path.join(self._base_folder, "leaf_tensor")
        self.format = ".pt"

        if not (os.path.exists(self._images_folder) and os.path.isdir(self._images_folder)):
            raise RuntimeError("Dataset not found or corrupted.")

        # obtain the leave categories
        self.leaf_classes = []
        extract_path = os.path.join(self._base_folder, "leaf_classes.csv")
        with open(extract_path, 'r') as f:
            for line in f:
                self.leaf_classes.append(line[:-1])

        self._labels = []
        self._images = []

        for i in range(self.num_classes):
            cate_image_path = os.path.join(self._images_folder, self.leaf_classes[i])
            for j in range(self.start+1, self.size + 1):
                img_name = "image (" + str(j) + ")" + self.format
                self._images.append(os.path.join(cate_image_path, img_name))
                self._labels.append(i)

    def __len__(self):
        # define the total number of samples in the dataset
        return len(self._images)

    def __getitem__(self, index):

        # retrieve an individual sample from the dataset as a tuple (data, target)
        data, label = self._images[index], self._labels[index]

        image = torch.load(data)
        image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
