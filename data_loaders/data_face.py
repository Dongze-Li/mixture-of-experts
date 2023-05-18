# load data from 100Faces folder into dataloader for pytorch
# Reference from face.py

from scipy.misc import face
from torch.utils.data import Dataset
import cv2
import glob
import random
from functools import reduce
from operator import concat
from torchvision import transforms as TR
import yaml

with open('config.yaml') as cf_file:
    config = yaml.safe_load( cf_file.read())
cs = config['stage_1']['datasets']['faces_size']
REPEAT = config['stage_1']['datasets']['repeat']
num_faces = config['stage_1']['num_faces']

LBS = 200
NT = cs[0]
NV = cs[0]+cs[1]

data_path = 'data/100Faces/'

# under train_data_path, there are 100 folders, each folder contains 30 images for 1 person

image_paths = [] #to store image paths in list
classes = [] #to store class values

random.seed(42)

random_index = range(30)
# permute the index
random_index = random.sample(random_index, len(random_index))

train_image_paths = []
train_classes = []
val_image_paths = []
val_classes = []
test_image_paths = []   
test_classes = []

dplist = glob.glob(data_path + '/*')

for dpindex in range(num_faces):
    dp = dplist[dpindex]
    for i in range(30): 
        for j in range(REPEAT):
            if i < NT:
                train_image_paths.append(glob.glob(dp + '/*')[random_index[i]])
                train_classes.append(int(dp.split('/')[-1]))
            elif i < NV:
                val_image_paths.append(glob.glob(dp + '/*')[random_index[i]])
                val_classes.append(int(dp.split('/')[-1]))
            else:
                test_image_paths.append(glob.glob(dp + '/*')[random_index[i]])
                test_classes.append(int(dp.split('/')[-1]))

# print('image_path example: ', image_paths[0])
# print('class example: ', classes[0])


# split train valid from train paths (80,20)
# train_image_paths, val_image_paths, test_image_paths = image_paths[:NT], image_paths[NT:NV], image_paths[NV:]
# train_classes, val_classes, test_classes = classes[:NT], classes[NT:NV], classes[NV:]


class faces_dataset(Dataset):
    # MODEL THE LABEL SHIFT HERE
    def __init__(self, image_paths, classes, transform=None, label_shift=LBS):
        self.image_paths = image_paths
        self.classes = classes
        self.transform = transform
        self.label_shift = label_shift
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        # read image from image_path, already in rgb, convert size to 224x224
        image = cv2.imread(image_filepath)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 3x224x224 
        # image = image.transpose(2, 0, 1)
        if self.transform is not None:
            image = self.transform(image)

        label = self.classes[idx] + self.label_shift

        return image, label
    
transform_s=TR.Compose([TR.ToTensor(),TR.RandomHorizontalFlip(p=0.5),TR.RandomRotation(degrees=5)])

# create the train, valid, test dataset
dataset_train = faces_dataset(train_image_paths, train_classes, transform=transform_s)
dataset_val = faces_dataset(val_image_paths, val_classes, transform=transform_s)
dataset_test = faces_dataset(test_image_paths, test_classes, transform=transform_s)