import random
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import pickle

import numpy as np
#import torchsample as ts

# borrowed from http://pytorch.org/tutorials/advanced/neural_style_tutorial.html
# and http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# define a training image loader that specifies transforms on images. See documentation for more details.

class ChairsDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, data_dir, transform):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.

        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """
        self.filenames = os.listdir(data_dir)
        self.filenames = [os.path.join(data_dir, f) for f in self.filenames]

        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        image = Image.open(self.filenames[idx]) # PIL image
        image = self.transform(image)
        return image

class ImagenetDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, data_dir, transform):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.

        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """

        self.train_data = []
        self.transform = transform

        base_folder = data_dir

        train_list = []
        for i in range(1,9):
            name = 'train_data_batch_'+str(i)
            train_list.append(name)

        for batch in train_list:
            file = os.path.join(base_folder, batch)
            with open(file, 'rb') as input:
                entry = pickle.load(input)
                self.train_data.append(entry['data'])
        self.train_data = np.concatenate(self.train_data)
        self.train_data = self.train_data.reshape((self.train_data.shape[0], 3, 64, 64))
        self.train_data = self.train_data.transpose((0, 2, 3, 1))

        #self.train_data = self.train_data/np.float32(255)
        #data_size = self.train_data.shape[0]
        #img_size = 64
        #img_size2 = img_size * img_size
        #self.train_data = np.dstack((self.train_data[:, :img_size2], self.train_data[:, img_size2:2*img_size2], self.train_data[:, 2*img_size2:]))
        #self.train_data = self.train_data.reshape((self.train_data.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)
        #X_train = self.train_data[0:data_size, :, :, :]

    def __len__(self):
        # return size of dataset
        return len(self.train_data)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        img = self.train_data[idx]
        image = Image.fromarray(img) # PIL image
        image = self.transform(image)
        return image

def fetch_dataloader(data_dir, batch_size, dataset):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
    types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    train_transformer = transforms.Compose([
                transforms.Resize((64,64)),        # resize the image to 64x64 (remove if images are already 64x64)
                transforms.RandomHorizontalFlip(),   # randomly flip image horizontally
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # transform it into a torch tensor

    if dataset == "Chairs":
        dataloader = DataLoader(ChairsDataset(data_dir, train_transformer), batch_size=batch_size, shuffle=True)

    elif dataset == "Cifar10":
        dataloader = torch.utils.data.DataLoader(
            datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transformer),
            batch_size=batch_size, shuffle=True)

    elif dataset == "Imagenet":
        dataloader = torch.utils.data.DataLoader(ImagenetDataset(data_dir, train_transformer),batch_size=batch_size, shuffle=True)

    return dataloader
