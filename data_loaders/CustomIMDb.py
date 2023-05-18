from torchvision.datasets import VisionDataset
import torch
from PIL import Image
import os
import numpy as np


class CustomIMDb(VisionDataset):

    def __init__(self, dataset_path, split="train", transform=None, target_transform=None, start=0, size=0, tensor=False, shift=0, classes=100, front_face=True):

        super(CustomIMDb, self).__init__(None, transform=transform, target_transform=target_transform)

        # make size configurable as an input parameter by variable factor
        self.dataset_path = dataset_path  # "/data" for project
        self.split = split
        self.classes = classes
        self.start = start
            
        if size > 100 and self.classes > 50:
                raise RuntimeError("Training data out of bounce")
        elif size > 20 and self.classes > 50:
                raise RuntimeError("Data out of bounce")
        else:
            self.size = size

        self.front_face = front_face
        # go into the folder of split
        curr_file = os.path.abspath(os.path.realpath("CustomIMDb.py")) # \mixture\data_loaders\..
        if self.classes == 50:
            self._base_folder = os.path.join(os.path.dirname(curr_file), self.dataset_path, "IMDb_50")
            extract_path = os.path.join(self._base_folder, "selected_IMDb.txt")
        
        if  self.front_face == False:
            print("large IMDb dataset for faces: 270+ images per class")
            self._base_folder = os.path.join(os.path.dirname(curr_file), self.dataset_path, "IMDb_Face")
            if self.classes == 50:
                extract_path = os.path.join(self._base_folder, "large_IMDb.txt")
            else:
                extract_path = os.path.join(self._base_folder, "selected_IMDb.txt")    
        
            
        self._images_folder = os.path.join(self._base_folder, "IMDb_images")
        self.shift = shift
        if tensor:
            self.tensor = True
            self.format = ".pt"
        else:
            self.tensor = False
            self.format = ".jpg"

        if not (os.path.exists(self._images_folder) and os.path.isdir(self._images_folder)):
            raise RuntimeError("Dataset not found or corrupted.")

        # obtain the leave categories
        self.all_characters = []
            
            
        with open(extract_path, 'r') as f:
            for line in f:
                info = line.split(',')
                self.all_characters.append(info[0])

        self._labels = []
        self._images = []

        for i in range(self.classes):
            cate_image_path = os.path.join(self._images_folder, self.all_characters[i])
            for j in range(self.start+1, self.start+self.size + 1):
                img_name = str(j) + self.format
                self._images.append(os.path.join(cate_image_path, img_name))
                self._labels.append(i+self.shift)


    def __len__(self):
        # define the total number of samples in the dataset
        return len(self._images)

    def __getitem__(self, index):

        # retrieve an individual sample from the dataset as a tuple (data, target)
        data, label = self._images[index], self._labels[index]

        if self.tensor:
            # Convert tensor to PIL image
            image = torch.load(data)
            # normalize
            image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))
            # image = transforms.ToPILImage()(data)
        else:
            image = Image.open(data).convert("RGB")
            if self.transform:
                image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label
