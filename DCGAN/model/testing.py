import random
import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


filenames = os.listdir('chairs')
print(filenames)
filenames = [os.path.join('chairs', f) for f in filenames if f.endswith('.jpg')]
print(filenames)
