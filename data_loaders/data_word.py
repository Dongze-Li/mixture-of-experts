from torchvision import transforms as TR
from torchvision import datasets, transforms
from data_loaders.CustomWord import CustomWord
import yaml

# Return datasets in split as: dataset_word_train, dataset_word_val, dataset_word_test


class data_word(object):

    def __init__(self, config):

        #Transforms
        self.transform_mnist = TR.Compose(
            [TR.ToTensor(),
            TR.Resize([224,224]),
            TR.Lambda(lambda x: x.repeat(3, 1, 1))]
        )

        self.transform_augment = TR.Compose(
            [TR.ToTensor(),
            TR.Resize([224,224]),
            TR.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5)]
        )
        self.transform_tensor = TR.Compose(
          [TR.ToTensor(),
          TR.Resize([224, 224])]
        )

        #100_Basic_Words

        # read the config file to retrieve dataset size
        with open(config) as cf_file:
            self.config = yaml.safe_load( cf_file.read())
        config_size = self.config['stage_1']['datasets']['words_size']
        num_classes = int(self.config['stage_1']['num_words'])

        # use 20% of the dataset
        # dataset_word_train = CustomWord('data', split='train', transform=transform_tensor, size=1000)

        # load full dataset from tensor files
        self.dataset_train = CustomWord('data', split='train', transform=self.transform_tensor, start=0, size=config_size[0], tensor=False, classes=num_classes)
        self.dataset_val = CustomWord('data', split='val', transform=self.transform_tensor, start=0, size=config_size[1], tensor=False, classes=num_classes)
        self.dataset_test = CustomWord('data', split='test', transform=self.transform_tensor, start=0, size=config_size[2], tensor=False, classes=num_classes)


    def getDataset(self, split):
        if split == "train":
            return self.dataset_train
        elif split == "val":
            return self.dataset_val
        elif split == "test":
            return self.dataset_test
        else:
            raise RuntimeError("Please request for a valid split: train, val , test. ")
