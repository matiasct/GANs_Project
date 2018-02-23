import random
import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchsample as ts

# borrowed from http://pytorch.org/tutorials/advanced/neural_style_tutorial.html
# and http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# define a training image loader that specifies transforms on images. See documentation for more details.
train_transformer = transforms.Compose([
    transforms.Resize((64,64)),  # resize the image to 64x64 (remove if images are already 64x64)
    transforms.RandomHorizontalFlip(),   # randomly flip image horizontally
    ts.transforms.Rotate(20), # data augmentation: rotation
    ts.transforms.Rotate(-20), # data augmentation: rotation
    transforms.ToTensor()])  # transform it into a torch tensor


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
        image = Image.open(self.filenames[idx]).convert('RGB') # PIL image

        image = self.transform(image)
        return image, self.labels[idx]


def fetch_dataloader(data_dir, batch_size):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
    types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """

    dataloader = DataLoader(ChairsDataset(data_dir, train_transformer), batch_size=batch_size, shuffle=True)

    return dataloader
