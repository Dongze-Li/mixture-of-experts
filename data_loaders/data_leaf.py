from torchvision import transforms as TR
from torchvision import datasets, transforms
from data_loaders.CustomLeaf import CustomLeaf
import yaml

# Return datasets in split as: dataset_word_train, dataset_word_val, dataset_word_test


class data_leaf(object):

    def __init__(self, config):


        #Transforms
        self.transform_mnist = TR.Compose(
            [TR.ToTensor(),
            TR.Resize([224,224]),
            TR.Lambda(lambda x: x.repeat(3, 1, 1))]
        )

        self.selfransform_augment = TR.Compose(
            [TR.ToTensor(),
            TR.Resize([224,224]),
            TR.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5)]
        )
        self.transform_tensor = TR.Compose(
          [TR.ToTensor(),
          TR.Resize([224, 224])]
        )
        
        self.transform_resize = TR.Compose(
          [TR.Resize([224, 224])]
        )


        #100_Basic_Words

        # read the config file to retrieve dataset size
        with open(config) as cf_file:
            self.config = yaml.safe_load( cf_file.read())
        config_size = self.config['stage_2']['expert_dataset']['size']
        NUM_CLASSES = self.config['stage_2']['expert_dataset']['num_classes']

        # load full dataset from tensor files
        self.dataset_train = CustomLeaf('data', split='train', transform=self.transform_resize, size=int(config_size[0]), num_classes=NUM_CLASSES)
        self.dataset_val = CustomLeaf('data', split='val', transform=self.transform_resize, size=int(config_size[1]), num_classes=NUM_CLASSES)
        self.dataset_test = CustomLeaf('data', split='test', transform=self.transform_resize, size=int(config_size[2]), num_classes=NUM_CLASSES)
        
        
        print("lead train size: ", len(self.dataset_train))


        # dataset_train = CustomLeaf('data', split='train', transform=transform_tensor)
        # dataset_val = CustomLeaf('data', split='val', transform=transform_tensor)
        # dataset_test = CustomLeaf('data', split='test', transform=transform_tensor)


    def getDataset(self, split):
        if split == "train":
            return self.dataset_train
        elif split == "val":
            return self.dataset_val
        elif split == "test":
            return self.dataset_test
        else:
            raise RuntimeError("Please request for a valid split: train, val , test. ")
