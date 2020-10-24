import numpy as np
import pandas as pd
import torch
from torchvision import transforms


def read_data(mode, *files):
    """
    @brief: Read data from csv files.
    """
    if mode == "train":
        train_set = pd.read_csv(files[0])
        val_set = pd.read_csv(files[1])

        train_images = train_set.iloc[:, 1:]
        train_labels = train_set.iloc[:, 0]
        val_images = val_set.iloc[:, 1:]
        val_labels = val_set.iloc[:, 0]

        return train_images, train_labels, val_images, val_labels

    else:
        test_set = pd.read_csv(files[0])
        test_images = test_set.iloc[:, 1:]

        return test_images


def train_transforms():
    """
    @brief: 
    """
    train_transform = transforms.Compose(([
        transforms.ToPILImage(),
        transforms.RandomCrop(28),
        transforms.RandomAffine(degrees=5, translate=(0.1, 0.1)),
        transforms.ToTensor(), # divides by 255
        ]))
    return train_transform


def test_transforms():
    test_transform = transforms.Compose(([
        transforms.ToPILImage(),
        transforms.ToTensor(), # divides by 255
        ]))
    return test_transform


class KannadaDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels=None, transforms=None):
        self.X = images
        self.y = labels
        self.transforms = transforms
                    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        #get item and convert it to numpy array
        data = self.X.iloc[i, :]
        data = np.array(data).astype(np.uint8).reshape(28, 28, 1) 
        
        # perform transforms if there are any
        if self.transforms:
            data = self.transforms(data)
        
        if self.y is not None: # train/val
            return (data, self.y[i])
        else: # test
            return data
