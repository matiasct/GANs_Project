import random
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
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

        self.labels = [1 for filename in self.filenames]
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
        return image, self.labels[idx]


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

    if dataset == "chairs":
        dataloader = DataLoader(ChairsDataset(data_dir, train_transformer), batch_size=batch_size, shuffle=True)

    elif dataset == "cifar10":
        dataloader = torch.utils.data.DataLoader(
            datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transformer),
            batch_size=batch_size, shuffle=True)

    elif dataset == "imagenet":
        dataloader = torch.utils.data.DataLoader(
            datasets.ImageFolder(data_dir, transform=train_transformer),
            batch_size=batch_size, shuffle=True)

    return dataloader
