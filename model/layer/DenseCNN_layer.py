import torch
import torch.nn as nn

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):   # growth rate:
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList([nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1)
        ) for _ in range(num_layers)])

    def forward(self, x):
        X = x   # Giving a initial value
        for layer in self.layers:
            out = layer(x)
            X = torch.cat([X, out], dim=1)
        return X

class DenseCNN(nn.Module):
    def __init__(self, in_channle, Dense_layer, outsize):
        super(DenseCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channle, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.dense1 = DenseBlock(in_channels=16, growth_rate=8, num_layers=Dense_layer)
        self.trans1 = nn.Sequential(
            nn.BatchNorm2d(16 + 8 * Dense_layer),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16 + 8 * Dense_layer, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.dense2 = DenseBlock(in_channels=32, growth_rate=8, num_layers=Dense_layer)
        self.trans2 = nn.Sequential(
            nn.BatchNorm2d(32 + 8 * Dense_layer),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32 + 8 * Dense_layer, out_channels=in_channle, kernel_size=1, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        # self.fc = nn.Linear(16 * 16, outsize)  # Dataset 1 - 16*16  Dataset 2 - 21*21
        self.fc = nn.Linear(21 * 21, outsize)

    def forward(self, x):
        x = self.conv1(x)  
        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.trans2(x)
        x = x.reshape(x.size()[0], x.size()[1], -1)
        x = self.fc(x)


        return x
