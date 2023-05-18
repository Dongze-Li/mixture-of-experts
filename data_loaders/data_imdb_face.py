from torchvision import transforms as TR
from torchvision import datasets, transforms
from data_loaders.CustomIMDb import CustomIMDb
import yaml
import torch

# Return datasets in split as: dataset_word_train, dataset_word_val, dataset_word_test

#Transforms

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean

class data_imdb_face(object):

    def __init__(self, config):

        self.transform_augment = TR.Compose(
            [TR.ToTensor(),
            TR.Resize([224,224]),
            transforms.RandomRotation(degrees=5),
            AddGaussianNoise(),
            TR.RandomHorizontalFlip(p=0.5)]
        )
        self.transform_tensor = TR.Compose(
          [TR.ToTensor()]
        )

        # read the config file to retrieve dataset size
        with open(config) as cf_file:
            self.config = yaml.safe_load( cf_file.read())

        config_size = self.config['stage_1']['datasets']['faces_size']
        label_shift = int(self.config['stage_1']['num_words']) + int(self.config['stage_1']['num_objects'])
        num_classes = int(self.config['stage_1']['num_faces'])
        front = (self.config['stage_1']['datasets']['frontface'] == 1) # use frontface if 1

        # use 20% of the dataset
        # dataset_word_train = CustomWord('data', split='train', transform=transform_tensor, size=1000)

        # load full dataset from tensor files
        self.dataset_train = CustomIMDb('data', split='train', transform=self.transform_tensor, start=0, size=config_size[0], shift=label_shift, classes=num_classes, front_face=front)
        self.dataset_val = CustomIMDb('data', split='val', transform=self.transform_tensor, start=config_size[0], size=config_size[1], shift=label_shift, classes=num_classes, front_face=front)
        self.dataset_test = CustomIMDb('data', split='test', transform=self.transform_tensor,  start=config_size[0]+config_size[1], size=config_size[2], shift=label_shift, classes=num_classes, front_face=front)


    def getDataset(self, split):
        if split == "train":
            return self.dataset_train
        elif split == "val":
            return self.dataset_val
        elif split == "test":
            return self.dataset_test
        else:
            raise RuntimeError("Please request for a valid split: train, val , test. ")




