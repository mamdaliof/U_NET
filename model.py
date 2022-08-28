"""
Authors: Mohammad Hoseyni
Email: Mohammadhosini60@gmail.com
GitHub: mamdaliof
you must set in_channels and out_channels in __init__()
kernel_size, stride, max_pool_stride, features and padding are optional
"""
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

"""
doubleConv provide a forward method which has two identical layer CNN -> BatchNorm -> Relu generally this function
generate double convolutional layer
"""


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


"""
int this class we use doubleConv and generate a Unet model 
"""


class UNET(nn.Module):
    def __init__(self, input_channels, out_channels, max_pool=2, stride=1,
                 padding=1, kernel_size=3, max_pool_stride=2, features=None):
        super(UNET, self).__init__()
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        if features is None:
            features = [64, 128, 256, 512]  # this is amount of features in each layer between input and output
        self.pool = nn.MaxPool2d(max_pool, stride=max_pool_stride)
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        # generate encoder
        for feature in features:
            self.downs.append(doubleConv(input_channels, feature, stride=self.stride, padding=self.padding,
                                         kernel_size=self.kernel_size))
            input_channels = feature
        # generate encoder
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=self.kernel_size, stride=self.stride))
            self.ups.append(doubleConv(feature * 2, feature, stride=self.stride, padding=self.padding,
                                       kernel_size=self.kernel_size))
        # double convolution in bottom of UNET
        self.bottom_conv = doubleConv(features[-1], 2 * features[-1], stride=self.stride, padding=self.padding,
                                      kernel_size=self.kernel_size)
        # generate final layer of convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []  # to record data before max pooling
        for down in self.downs:  # encode
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottom_conv(x)
        skip_connections = list(reversed(skip_connections))
        for i in range(0, len(self.ups), 2):  # decode
            x = self.ups[i](x)
            skip_connection = skip_connections[i // 2]
            if x.shape != skip_connection.shape:
                size = list(skip_connection.shape[2:])
                x = TF.resize(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i + 1](concat_skip)
        return self.final_conv(x)


def test():
    x = torch.rand((5, 3, 57, 571))
    print(x.shape)
    model = UNET(3, 3)
    anw = model(x)
    print(anw.shape)
