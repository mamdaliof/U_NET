import os
from torch.utils.data import Dataset
import glob
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
import torch


class CarDataLoader(Dataset):
    def __init__(self, path, transform=None, split_dim=1):
        super(Dataset, self).__init__()
        self.path = path
        self.split_dim = split_dim
        # extract path of data
        self.dataPaths = glob.glob(os.path.join(self.path, "*.jpg"))
        print('Number of imgs: %d' % (len(self.dataPaths)))
        # self.transform = transform if transform is not None else transforms.Compose([transforms.ToTensor()])
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        image_path = self.dataPaths[index]
        img = Image.open(image_path)
        img = np.array(img)
        img = self.transform(img).float()
        if self.split_dim == 1:
            if img.shape[-1] % 2 == 1:
                new_shape = list(img.shape)
                new_shape[-1] -= 1
                img = TF.resize(img, size=new_shape[-2:])
            mask, original = img.split(int(img.shape[-1] / 2), len(img.shape) - 1)
        else:
            if img.shape[-2] % 2 == 1:
                new_shape = list(img.shape)
                new_shape[-2] -= 1
                img = TF.resize(img, size=new_shape[-2:])
            mask, original = img.split(int(img.shape[-1] / 2), len(img.shape) - 2)
        return original, mask

    # The __len__ function returns the number of samples in our dataset.
    def __len__(self):
        return len(self.dataPaths)
