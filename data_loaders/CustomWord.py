from torchvision.datasets import VisionDataset
import torch
from PIL import Image
import os
import numpy as np
import pathlib


class CustomWord(VisionDataset):

    def __init__(self, dataset_path, split="train", transform=None, target_transform=None, start=0, size=None, tensor=True, classes=100):

        super(CustomWord, self).__init__(None, transform=transform, target_transform=target_transform)

        self.categories = ["opt", "thing", "qual"]

        # make size configurable as an input parameter by variable factor
        self.size = size
        self.dataset_path = dataset_path  # "/data" for project
        self.split = split
        self.classes = classes
        self.start = start
        if self.split == "train":
            if size is None:
                self.size = 20
            elif size > 1000:
                raise RuntimeError("Training data out of bounce")
        else:
            if size is None:
                self.size = 5
            elif size > 100:
                raise RuntimeError("Data out of bounce")

        self.tensor = tensor

        # go into the folder of split
#         self._base_folder = os.path.join(os.path.dirname(os.getcwd()), self.dataset_path, "100_basic_english_words")
        curr_file = os.path.abspath(os.path.realpath("CustomWord.py"))
        self._base_folder = os.path.join(os.path.dirname(curr_file), self.dataset_path, "100_basic_english_words")
        self._images_folder = os.path.join(self._base_folder, self.split)
        if self.tensor:
            self._images_folder += "_tensor"
            self.format = ".pt"
        else:
            self.format = ".png"

        #print("folder: ", self._images_folder)

        if not (os.path.exists(self._images_folder) and os.path.isdir(self._images_folder)):
#             raise RuntimeError(pathlib.Path().absolute())
            raise RuntimeError("Dataset not found or corrupted.")

        # obtain the words
        self.word_dict = {}
        self.word = []
        extract_path = os.path.join(self._base_folder, "words.csv")
        with open(extract_path, 'r') as f:
            for line in f:
                omit = True
                category = self.categories[1]
                for word in line.split():
                    if omit:  # omit the categorical data
                        if word[:-1] == "Operations":
                            category = self.categories[0]
                        elif word[:-1] == "Qualities":
                            category = self.categories[2]
                        omit = False
                        continue
                    self.word_dict[category] = word.split(',')[:-1]
                    self.word = self.word + word.split(',')[:-1]

        self._labels = []
        self._images = []

        for i in range(self.classes):
            word = self.word[i]
            category = ""
            for c in self.categories:
                if word in self.word_dict[c]:
                    category = c
                    break
            for j in range(self.start+1, self.start+self.size + 1):  # 1000+1
                img_name = category + "_" + word + "_" + str(j) + self.format
                self._images.append(os.path.join(self._images_folder, img_name))
                self._labels.append(i)

    def __len__(self):
        # define the total number of samples in the dataset
        return len(self._images)

    def __getitem__(self, index):

        # retrieve an individual sample from the dataset as a tuple (data, target)
        data, label = self._images[index], self._labels[index]

        if self.tensor:
            # Convert tensor to PIL image
            image = torch.load(data)
            image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))
            # image = transforms.ToPILImage()(data)
        else:
            image = Image.open(data).convert("RGB")
            if self.transform:
                image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label
