import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import ModuleList
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from net.block import from_rgb3, to_rgb, conv_block, up_conv, DisGeneralConvBlock, DisFinalBlock


# Initialize weights
def weights_init_normal(m):
    """Initialize weights with normal distribution."""
    if hasattr(m, 'weight') and m.weight is not None:
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)


# Depth Encoder for preprocessing depth features
class DepthEncoder(nn.Module):
    def __init__(self):
        super(DepthEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, depth):
        x = self.pool(F.relu(self.conv1(depth)))
        x = self.pool(F.relu(self.conv2(x)))
        return x


# Self-Attention Block
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, c, h, w = x.size()
        query = self.query(x).view(batch, -1, h * w).permute(0, 2, 1)
        key = self.key(x).view(batch, -1, h * w)
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        value = self.value(x).view(batch, -1, h * w)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch, c, h, w)
        return self.gamma * out + x


# Generator with Depth Encoder and Self-Attention
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.depth_encoder = DepthEncoder()
        self.Conv1 = conv_block(3, 16)  # RGB input
        self.Conv2 = conv_block(16, 32)
        self.Conv3 = conv_block(32, 64)
        self.Conv4 = conv_block(64, 128)
        self.Up1 = up_conv(128 + 32, 64)  # Include depth features
        self.Up2 = up_conv(64, 32)
        self.Up3 = up_conv(32, 16)
        self.attention = SelfAttention(128)  # Self-Attention on the bottleneck features
        self.Output = nn.Conv2d(16, 3, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the Generator.
        x: Input tensor (combined RGB and depth channels).
        """
        # Split the input into RGB and depth components
        input_rgb = x[:, :3, :, :]  # First 3 channels are RGB
        input_depth = x[:, 3:, :, :]  # Last channel is depth

        # Process depth features
        depth_features = self.depth_encoder(input_depth)

        # Process RGB features
        rgb_features1 = self.Conv1(input_rgb)
        rgb_features2 = self.Conv2(rgb_features1)
        rgb_features3 = self.Conv3(rgb_features2)
        rgb_features4 = self.Conv4(rgb_features3)

        # Attention on bottleneck features
        rgb_features4 = self.attention(rgb_features4)

        # Combine RGB and depth features
        combined_features = torch.cat([rgb_features4, depth_features], dim=1)

        # Decode
        out = self.Up1(combined_features)
        out = self.Up2(out)
        out = self.Up3(out)
        output = self.Output(out)

        return output


# Discriminator (unchanged)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.initial_conv = DisGeneralConvBlock(7, 64)  # Updated to handle 7 input channels
        self.layers = nn.ModuleList([
            DisGeneralConvBlock(128, 128),
            DisGeneralConvBlock(256, 256),
            DisGeneralConvBlock(512, 512),
        ])
        self.final_layer = DisFinalBlock(512)

        self.channel_adjust = nn.ModuleList([
            nn.Conv2d(64 + 64, 128, kernel_size=1),
            nn.Conv2d(128 + 128, 256, kernel_size=1),
            nn.Conv2d(256 + 256, 512, kernel_size=1),
        ])

    def forward(self, inputs, conditions):
        """
        inputs: List of tensors [low_res, mid_res, ..., high_res] for the target.
        conditions: List of tensors [low_res, mid_res, ..., high_res] for the condition.
        """
        assert len(inputs) == 4 and len(conditions) == 4, "Inputs must be lists of 4 tensors each."

        x = torch.cat([inputs[3], conditions[3]], dim=1)
        y = self.initial_conv(x)

        for i, layer in enumerate(self.layers):
            input_feature = inputs[2 - i]
            condition_feature = conditions[2 - i]

            if input_feature.size(2) != condition_feature.size(2) or input_feature.size(3) != condition_feature.size(3):
                condition_feature = F.interpolate(condition_feature, size=input_feature.size()[2:], mode="bilinear", align_corners=False)

            concat_features = torch.cat([input_feature, condition_feature], dim=1)

            if y.size(2) != concat_features.size(2) or y.size(3) != concat_features.size(3):
                y = F.interpolate(y, size=concat_features.size()[2:], mode="bilinear", align_corners=False)

            total_channels = concat_features.size(1) + y.size(1)
            if total_channels != self.channel_adjust[i].in_channels:
                self.channel_adjust[i] = nn.Conv2d(total_channels, self.channel_adjust[i].out_channels, kernel_size=1).to(concat_features.device)

            adjusted_features = self.channel_adjust[i](torch.cat([concat_features, y], dim=1))
            y = layer(adjusted_features)

        y = self.final_layer(y)
        return y
