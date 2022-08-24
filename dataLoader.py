import os
from torch.utils.data import Dataset
import glob
from torchvision import transforms
from PIL import Image
import numpy as np
import torch


class carDataLoader(Dataset):
    def __init__(self, path, transform=None):
        super(Dataset, self).__init__()
        self.path = path
        # extract path of data
        self.dataPaths = glob.glob(os.path.join(self.path, "*.jpg"))
        print('Number of imgs: %d' % (len(self.dataPaths)))
        # self.transform = transform if transform is not None else transforms.Compose([transforms.ToTensor()])
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        imagePath = self.dataPaths[index]
        img = Image.open(imagePath)
        img = np.array(img)
        result = self.transform(img).float()
        return result

    # The __len__ function returns the number of samples in our dataset.
    def __len__(self):
        return len(self.dataPaths)
