import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from CustomDataLoader import CarDataLoader
from Model import UNET


class train:
    def __init__(self, print_device=False, args):
        self.model = UNET(args.in_channles, args.out_channels)
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = CarDataLoader(args.train_path, split_dim=0)
        train_data = DataLoader(dataset, 1)
        if print_device:
            print(self.device)
        self.criterion = nn.CrossEntropyLoss
        self.optimizer=torch.optim.SGD(self.model.parameters(),args.main_lr)