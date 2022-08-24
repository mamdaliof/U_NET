import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from CustomDataLoader import CarDataLoader
from Model import UNET


class train:
    def __init__(self, args, print_device=False):
        self.model = UNET(args.in_channles, args.out_channels)
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_set = CarDataLoader(args.train_path, self.device, split_dim=1)
        self.train_data = DataLoader(train_set,
                                     batch_size=args.batch_size,
                                     pin_memory=False,
                                     shuffle=False,
                                     num_workers=2)
        val_set = CarDataLoader(args.val_path, self.device, split_dim=1)
        self.val_data = DataLoader(val_set,
                                   batch_size=args.batch_size,
                                   pin_memory=False,
                                   shuffle=False,
                                   num_workers=2)
        if print_device:
            print(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), args.main_lr)
        self.train_cost = []
        self.val_cost = []

    def forward(self, x):
        for epoch in self.args.epochs:
            for i, (original, mask) in enumerate(self.train_data):
                self.optimizer.zero_grad()
                rec_img = self.model(original.float())
                loss = self.criterion(mask, rec_img)
                loss.backward()
                self.optimizer.step()

            if (i % 200 == 0):
                self.train_cost.append(loss)
                print("epoch = ", epoch, "itr = ", i)
