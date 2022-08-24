import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class doubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(doubleConv, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(),
                                  nn.Conv2d(out_channels, out_channels, kernel_size, stride=stride, padding=padding),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU()
                                  )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(self, input_channels, out_channels, max_pool=2, stride=2, features=None, kernel_size=3):
        super(UNET, self).__init__()
        if features is None:
            features = [64, 128, 256, 512]
        self.pool = nn.MaxPool2d(max_pool, stride)
        self.downs = []
        for feature in features:
            self.downs.append(doubleConv(input_channels, feature))
            input_channels = feature
        self.ups = []
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size, stride=stride))
            self.ups.append(doubleConv(feature * 2, feature))
        self.bottom_conv = doubleConv(features[-1], 2 * features[-1])
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottom_conv(x)
        skip_connections = list(reversed(skip_connections))
        for i in range(0, len(self.ups), 2):

            x = self.ups[i](x)
            skip_connection = skip_connections[i // 2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)

            x = self.ups[i + 1](concat_skip)
        return self.final_conv(x)


def test():
    x = torch.rand((5, 3, 572, 572))
    print(x.shape)
    model = UNET(3, 3)
    anw = model(x)
    print(anw.shape)
