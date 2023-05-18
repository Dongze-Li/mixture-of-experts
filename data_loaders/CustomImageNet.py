import torch
import numpy as np
from torch.utils.data import Subset, DataLoader
from torchvision import transforms, datasets

def custom_imagenet_dataset(data_directory, split_sizes=[20,5,5], num_classes: int=100, index_offset=0):
    assert len(split_sizes) == 3, "must have 3 split sizes: [train, val, test]"
    
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    
    full_dataset = datasets.ImageFolder(
        data_directory,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=7),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.07),
            transforms.Lambda(lambda x: torch.clamp(x, 0., 1.)),
            transforms.RandomAffine(degrees=(-7, 7), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-7, 7))
            #normalize
        ]),
        target_transform=lambda t: t + index_offset)


    _, class_offsets, class_counts = np.unique(full_dataset.targets, return_index=True, return_counts=True)

    # Limit categories to num_classes
    class_counts = class_counts[range(num_classes)]
    class_offsets = class_offsets[range(num_classes)]
        
    # Sample the right number of examples from each class
    classes = [np.random.choice(np.arange(num), min(sum(split_sizes), num), replace=False) + offset for num, offset in zip(class_counts, class_offsets)]
    
    # Divide classes into train/val/test, making sure each split gets the correct number of examples for every class
    splits = [split_sizes[0], split_sizes[0] + split_sizes[1]]
    train_indices, val_indices, test_indices = (np.concatenate(ind) for ind in zip(*[np.split(c, splits) for c in classes]))

    return Subset(full_dataset, train_indices), Subset(full_dataset, val_indices), Subset(full_dataset, test_indices)
