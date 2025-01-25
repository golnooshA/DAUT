import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """Channel Attention Block"""
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y

class ResidualBlock(nn.Module):
    """Residual Block with Conv -> BN -> ReLU -> Conv -> BN."""
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Adjust dimensions if in_channels != out_channels
        self.adjust_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False) \
            if in_channels != out_channels else None

    def forward(self, x):
        identity = x
        if self.adjust_channels:
            identity = self.adjust_channels(identity)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)

class Generator(nn.Module):
    def __init__(self, input_channels=4, output_channels=4):
        super(Generator, self).__init__()

        # Encoder with residual blocks and Channel Attention blocks
        self.Conv1 = nn.Sequential(
            ResidualBlock(input_channels, 16),
            ChannelAttention(16)
        )
        self.Conv2 = nn.Sequential(
            ResidualBlock(16, 32),
            ChannelAttention(32)
        )
        self.Conv3 = nn.Sequential(
            ResidualBlock(32, 64),
            ChannelAttention(64)
        )
        self.Conv4 = nn.Sequential(
            ResidualBlock(64, 128),
            ChannelAttention(128)
        )

        # Decoder with skip connections
        self.Up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.Up2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.Up3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.Output = nn.Sequential(
            nn.Conv2d(16, output_channels, kernel_size=1),
            nn.Tanh()  # Scale output to [-1, 1]
        )

    def forward(self, x):
        # Encoder
        e1 = self.Conv1(x)
        e2 = self.Conv2(F.max_pool2d(e1, kernel_size=2))
        e3 = self.Conv3(F.max_pool2d(e2, kernel_size=2))
        e4 = self.Conv4(F.max_pool2d(e3, kernel_size=2))

        # Decoder
        d1 = self.Up1(e4) + e3
        d2 = self.Up2(d1) + e2
        d3 = self.Up3(d2) + e1

        # Output
        out = self.Output(d3)
        return [d1, d2, d3, out]

class Discriminator(nn.Module):
    def __init__(self, input_channels=8):
        super(Discriminator, self).__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64 * 2**i, 64 * 2**(i+1), kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64 * 2**(i+1)),
                nn.LeakyReLU(0.2, inplace=True)
            ) for i in range(3)
        ])

        self.final_layer = nn.Conv2d(512, 1, kernel_size=4, padding=1)

    def forward(self, x):
        x = self.initial_conv(x)
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        return x
