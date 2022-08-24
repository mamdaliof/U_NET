import os
import glob
import torch
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from dataLoader import carDataLoader
import model

def main():
    model.test()
if __name__ == '__main__':
    main()
