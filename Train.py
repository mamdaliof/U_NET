"""
Authors: Mohammad Hoseyni
Email: Mohammadhosini60@gmail.com
GitHub: mamdaliof
this file, and it's class are using for the purpose of train the UNET model.
args are mandatory input and print_device is optional which show's your program is running on your GPU or CPU
args content include: epochs, batch_size, main_lr, train_path, val_path, in_channels and out_channels
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from CustomDataLoader import CarDataLoader
from Model import UNET


class train:
    def __init__(self, args, print_device=False):
        self.args = args
        self.model = UNET(args.in_channels, args.out_channels)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_set = CarDataLoader(args.train_path, self.device)
        self.train_data = DataLoader(train_set,
                                     batch_size=args.batch_size,
                                     pin_memory=False,
                                     shuffle=False,
                                     num_workers=2)
        val_set = CarDataLoader(args.val_path, self.device)
        self.val_data = DataLoader(val_set,
                                   batch_size=1,
                                   pin_memory=False,
                                   shuffle=False,
                                   num_workers=2)
        if print_device:
            print(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), args.main_lr)
        self.train_cost = []
        self.val_cost = []

    def execute(self):
        for epoch in range(1, self.args.epochs):
            self.optimizer = torch.optim.SGD(self.model.parameters(), self.args.main_lr / epoch)
            for i, (original, mask) in enumerate(self.train_data):
                self.optimizer.zero_grad()
                rec_img = self.model(original.float())
                loss = self.criterion(mask, rec_img)
                loss.backward()
                self.optimizer.step()
                if i % 10 == 1:
                    self.train_cost.append(loss)
                    print("epoch = ", epoch, "itr = ", i)
                    for org_val, mask_val in self.val_data:
                        self.model.eval()
                        with torch.no_grad():
                            rec_img = self.model(org_val)
                            plt.figure()
                            plt.imshow(rec_img.squeeze().permute(1, 2, 0).numpy())
                            plt.figure()
                            plt.imshow(mask_val.squeeze().permute(1, 2, 0).numpy())
                            plt.show()
                        break
