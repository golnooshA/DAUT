import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import ModuleList
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from net.block import from_rgb3, to_rgb, conv_block, up_conv, DisGeneralConvBlock, DisFinalBlock

def weights_init_normal(m):
    """Initialize weights with normal distribution."""
    if hasattr(m, 'weight') and m.weight is not None:
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.Conv1 = conv_block(4, 16)  # Updated to 4 channels (RGB + Depth)
        self.Conv2 = conv_block(16, 32)
        self.Conv3 = conv_block(32, 64)
        self.Conv4 = conv_block(64, 128)
        self.Up1 = up_conv(128, 64)
        self.Up2 = up_conv(64, 32)
        self.Up3 = up_conv(32, 16)
        self.Output = nn.Conv2d(16, 3, kernel_size=1)

    def forward(self, x):
        e1 = self.Conv1(x)
        e2 = self.Conv2(e1)
        e3 = self.Conv3(e2)
        e4 = self.Conv4(e3)

        d1 = self.Up1(e4)
        d1 = F.interpolate(d1, size=e3.size()[2:], mode="bilinear", align_corners=False)
        d2 = self.Up2(d1 + e3)
        d2 = F.interpolate(d2, size=e2.size()[2:], mode="bilinear", align_corners=False)
        d3 = self.Up3(d2 + e2)
        d3 = F.interpolate(d3, size=e1.size()[2:], mode="bilinear", align_corners=False)

        out = self.Output(d3 + e1)
        return [d1, d2, d3, out]

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.initial_conv = DisGeneralConvBlock(7, 64)  # 7 channels: 3 (RGB) + 1 (Depth) + 3 (Generated RGB)
        self.layers = nn.ModuleList([
            DisGeneralConvBlock(128, 128),
            DisGeneralConvBlock(256, 256),
            DisGeneralConvBlock(512, 512),
        ])
        self.final_layer = DisFinalBlock(512)
        self.channel_adjust = nn.ModuleList([
            nn.Conv2d(128, 128, kernel_size=1),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.Conv2d(512, 512, kernel_size=1),
        ])

    def forward(self, inputs, conditions):
        assert len(inputs) == len(conditions) == 4, "Inputs and conditions must have 4 scales."
        
        # Initial concatenation
        x = torch.cat([inputs[3], conditions[3]], dim=1)  # Concatenate features at highest resolution
        y = self.initial_conv(x)

        # Iterate through scales
        for i, layer in enumerate(self.layers):
            input_feature = inputs[2 - i]
            condition_feature = conditions[2 - i]

            # Ensure input and condition features have matching sizes
            if input_feature.size()[2:] != condition_feature.size()[2:]:
                condition_feature = F.interpolate(condition_feature, size=input_feature.shape[2:], mode="bilinear", align_corners=False)

            concat_features = torch.cat([input_feature, condition_feature], dim=1)

            # Ensure `y` matches the size of `concat_features`
            if y.size()[2:] != concat_features.size()[2:]:
                y = F.interpolate(y, size=concat_features.shape[2:], mode="bilinear", align_corners=False)

            total_channels = concat_features.size(1) + y.size(1)

            # Dynamically adjust `channel_adjust` if necessary
            if total_channels != self.channel_adjust[i].in_channels:
                self.channel_adjust[i] = nn.Conv2d(total_channels, self.channel_adjust[i].out_channels, kernel_size=1).to(concat_features.device)

            adjusted_features = self.channel_adjust[i](torch.cat([concat_features, y], dim=1))
            y = layer(adjusted_features)

        # Final layer
        y = self.final_layer(y)
        return y



# import torch.nn as nn
# import torch.nn.functional as F
# import torch
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from net.block import conv_block, up_conv, DisGeneralConvBlock, DisFinalBlock

# # Lightweight Depth Encoder
# class DepthEncoder(nn.Module):
#     def __init__(self):
#         super(DepthEncoder, self).__init__()
#         self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
#         self.pool = nn.AvgPool2d(2, 2)

#     def forward(self, depth):
#         x = F.relu(self.conv1(depth))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
#         return x

# # Efficient Channel Attention
# class ChannelAttention(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=8):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
#         self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc2(F.relu(self.fc1(self.avg_pool(x))))
#         return x * self.sigmoid(avg_out)

# # Generator
# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.rgb_conv = conv_block(3, 16)  # For RGB input
#         self.depth_conv = conv_block(1, 16)  # For Depth input
#         self.merged_conv = conv_block(32, 32)  # Combine features

#         self.Conv2 = conv_block(32, 64)
#         self.Conv3 = conv_block(64, 128)
#         self.Up1 = up_conv(128, 64)
#         self.Up2 = up_conv(64, 32)
#         self.Output = nn.Conv2d(32, 3, kernel_size=1)

#     def forward(self, x):
#         rgb_features = self.rgb_conv(x[:, :3, :, :])  # Extract RGB features
#         depth_features = self.depth_conv(x[:, 3:, :, :])  # Extract Depth features

#         merged = torch.cat([rgb_features, depth_features], dim=1)
#         merged = self.merged_conv(merged)

#         e2 = self.Conv2(merged)
#         e3 = self.Conv3(e2)

#         d1 = self.Up1(e3)
#         d2 = self.Up2(d1)

#         out = self.Output(d2)
#         return out

# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.initial_conv = DisGeneralConvBlock(7, 64)  # 7 channels: 3 (RGB) + 1 (Depth) + 3 (Generated RGB)
#         self.layers = nn.ModuleList([
#             DisGeneralConvBlock(128, 128),
#             DisGeneralConvBlock(256, 256),
#             DisGeneralConvBlock(512, 512),
#         ])
#         self.final_layer = DisFinalBlock(512)
#         self.channel_adjust = nn.ModuleList([
#             nn.Conv2d(128, 128, kernel_size=1),
#             nn.Conv2d(256, 256, kernel_size=1),
#             nn.Conv2d(512, 512, kernel_size=1),
#         ])

#     def forward(self, inputs, conditions):
#         assert len(inputs) == len(conditions) == 4, "Inputs and conditions must have 4 scales."
        
#         # Initial concatenation
#         x = torch.cat([inputs[3], conditions[3]], dim=1)  # Concatenate features at highest resolution
#         y = self.initial_conv(x)

#         # Iterate through scales
#         for i, layer in enumerate(self.layers):
#             input_feature = inputs[2 - i]
#             condition_feature = conditions[2 - i]

#             # Ensure input and condition features have matching sizes
#             if input_feature.size()[2:] != condition_feature.size()[2:]:
#                 condition_feature = F.interpolate(condition_feature, size=input_feature.shape[2:], mode="bilinear", align_corners=False)

#             concat_features = torch.cat([input_feature, condition_feature], dim=1)

#             # Ensure `y` matches the size of `concat_features`
#             if y.size()[2:] != concat_features.size()[2:]:
#                 y = F.interpolate(y, size=concat_features.shape[2:], mode="bilinear", align_corners=False)

#             total_channels = concat_features.size(1) + y.size(1)

#             # Dynamically adjust `channel_adjust` if necessary
#             if total_channels != self.channel_adjust[i].in_channels:
#                 self.channel_adjust[i] = nn.Conv2d(total_channels, self.channel_adjust[i].out_channels, kernel_size=1).to(concat_features.device)

#             adjusted_features = self.channel_adjust[i](torch.cat([concat_features, y], dim=1))
#             y = layer(adjusted_features)

#         # Final layer
#         y = self.final_layer(y)
#         return y