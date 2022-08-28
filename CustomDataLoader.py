"""
Authors: Mohammad Hoseyni
Email: Mohammadhosini60@gmail.com
GitHub: mamdaliof
this class will import datasets as tensor in purpose to use in DataLoader in pytorch.
in purpose to split your picture from the middle horizontally you have to set split_dim = 1 in
__init__().
 in purpose to split your picture from the middle vertically you have to set split_dim = 0 in
__init__().
otherwise, use split_dim = None
you can send a transform manually to __init__().
"""
import glob
import os

import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CarDataLoader(Dataset):
    def __init__(self, path, device, transform=None, split_dim=1, ):
        super(Dataset, self).__init__()
        self.path = path
        self.split_dim = split_dim
        self.device = device
        self.dataPaths = glob.glob(os.path.join(self.path, "*.jpg"))  # extract path of data
        print('Number of imgs: %d' % (len(self.dataPaths)))
        self.transform = transform if transform is not None else \
            transforms.Compose([transforms.ToTensor(), T.Normalize((0, 0, 0), (1, 1, 1))])  # mean=0 variance=1

    def __getitem__(self, index):
        image_path = self.dataPaths[index]
        img = Image.open(image_path).convert("RGB")  # loac images
        img = np.array(img)  # convert images to numpy
        img = self.transform(img).float()
        if self.split_dim == 1:  # split horizontally
            if img.shape[-1] % 2 == 1:
                new_shape = list(img.shape)
                new_shape[-1] -= 1
                img = TF.resize(img, size=new_shape[-2:])
            mask, original = img.split(int(img.shape[-1] / 2), len(img.shape) - 1)
            mask.to(self.device)  # load to device
            original.to(self.device)  # load to device
            return mask, original
        elif self.split_dim == 0:  # split vertically
            if img.shape[-2] % 2 == 1:
                new_shape = list(img.shape)
                new_shape[-2] -= 1
                img = TF.resize(img, size=new_shape[-2:])
            mask, original = img.split(int(img.shape[-2] / 2), len(img.shape) - 2)
            mask.to(self.device)  # load to device
            original.to(self.device)  # load to device
            return mask, original
        img.to(self.device)  # load to device
        return img

    # The __len__ function returns the number of samples in our dataset.
    def __len__(self):
        return len(self.dataPaths)
