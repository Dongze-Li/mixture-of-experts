import yaml
from data_loaders.CustomImageNet import custom_imagenet_dataset


class data_object(object):


    def __init__(self, config):
        with open(config) as cf_file:
            self.config = yaml.safe_load( cf_file.read())
        split_sizes = self.config['stage_1']['datasets']['objects_size']
        index_offset = int(self.config['stage_1']['num_words'])
        num_classes = self.config['stage_1']['num_objects']

        objects_path = self.config['stage_1']['datasets']['objects_path']

        self.dataset_train, self.dataset_val, self.dataset_test = custom_imagenet_dataset(objects_path, split_sizes=split_sizes, num_classes=num_classes, index_offset=index_offset)

    def getDataset(self, split):
        if split == "train":
            return self.dataset_train
        elif split == "val":
            return self.dataset_val
        elif split == "test":
            return self.dataset_test
        else:
            raise RuntimeError("Please request for a valid split: train, val , test. ")
