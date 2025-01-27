import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision.models import vgg16
from torchvision.transforms import Normalize
from net.Ushape_attention import Generator, Discriminator
import numpy as np

# Paths
PATH_INPUT = './dataset/UIEB/input'
PATH_DEPTH = './DPT/output_monodepth/UIEB_Changed/'
PATH_GT = './dataset/UIEB/GT/'
SAVE_DIR = './save_model/'
os.makedirs(SAVE_DIR, exist_ok=True)

# Dataset Class
class DepthDataset(Dataset):
    def __init__(self, input_path, depth_path, gt_path):
        self.input_list = sorted([f for f in os.listdir(input_path) if f.endswith('.png')])
        self.depth_path = depth_path
        self.gt_path = gt_path

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        input_file = self.input_list[idx]
        # RGB Input
        input_img = cv2.imread(os.path.join(PATH_INPUT, input_file))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (256, 256))
        input_tensor = torch.from_numpy(input_img).permute(2, 0, 1).float() / 255.0

        # Depth Map
        depth_file = input_file.replace('.png', 'depth.png')
        depth_img = cv2.imread(os.path.join(PATH_DEPTH, depth_file), cv2.IMREAD_GRAYSCALE)
        depth_img = cv2.resize(depth_img, (256, 256))
        depth_tensor = torch.from_numpy(depth_img).unsqueeze(0).float() / 255.0

        # Concatenate RGB + Depth as Input
        real_A = torch.cat([input_tensor, depth_tensor], dim=0)

        # Ground Truth (RGB + Depth)
        gt_img = cv2.imread(os.path.join(PATH_GT, input_file))
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        gt_img = cv2.resize(gt_img, (256, 256))
        gt_tensor = torch.from_numpy(gt_img).permute(2, 0, 1).float() / 255.0

        # Combine GT RGB and Depth
        real_B = torch.cat([gt_tensor, depth_tensor], dim=0)

        return real_A, real_B

# Perceptual Loss Class
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg = vgg16(pretrained=True).features[:16]  # Use the first few layers of VGG16
        self.vgg = nn.Sequential(*list(vgg.children()))
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, fake, real):
        fake_norm = self.normalize(fake)
        real_norm = self.normalize(real)
        fake_features = self.vgg(fake_norm)
        real_features = self.vgg(real_norm)
        return torch.nn.functional.l1_loss(fake_features, real_features)

# Loss Functions
criterion_GAN = nn.MSELoss().cuda()
criterion_pixelwise = nn.L1Loss().cuda()
criterion_perceptual = VGGPerceptualLoss().cuda()

def histogram_loss(output, target):
    output_hist = torch.histc(output, bins=256, min=0, max=1)
    target_hist = torch.histc(target, bins=256, min=0, max=1)
    output_hist = output_hist / output_hist.sum()
    target_hist = target_hist / target_hist.sum()
    return torch.sum((output_hist - target_hist) ** 2)

def color_loss(output, target):
    return torch.mean(torch.abs(torch.mean(output, dim=(2, 3)) - torch.mean(target, dim=(2, 3))))

# Initialize Models
generator = Generator(input_channels=4, output_channels=4).cuda()
discriminator = Discriminator(input_channels=8).cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.00005, betas=(0.5, 0.999))

# Training Loop
n_epochs = 300
train_dataset = DepthDataset(PATH_INPUT, PATH_DEPTH, PATH_GT)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)

for epoch in range(n_epochs):
    for i, (real_A, real_B) in enumerate(train_loader):
        real_A, real_B = real_A.cuda(), real_B.cuda()

        # Train Discriminator
        optimizer_D.zero_grad()
        fake_B = torch.clamp(generator(real_A)[-1], 0, 1)
        pred_real = discriminator(torch.cat([real_A, real_B], dim=1))
        pred_fake = discriminator(torch.cat([real_A, fake_B.detach()], dim=1))
        loss_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
        loss_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
        loss_D = 0.5 * (loss_real + loss_fake)
        loss_D.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        pred_fake = discriminator(torch.cat([real_A, fake_B], dim=1))
        loss_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))
        loss_pixel = criterion_pixelwise(fake_B, real_B)
        loss_histogram = histogram_loss(fake_B, real_B)
        loss_perceptual = criterion_perceptual(fake_B[:, :3, :, :], real_B[:, :3, :, :])  # Only RGB channels
        loss_color = color_loss(fake_B[:, :3, :, :], real_B[:, :3, :, :])
        loss_G = loss_GAN + 15 * loss_pixel + 1.0 * loss_histogram + 0.2 * loss_perceptual + 0.1 * loss_color
        loss_G.backward()
        optimizer_G.step()

        print(f"[Epoch {epoch+1}/{n_epochs}] [Batch {i+1}/{len(train_loader)}] "
              f"[D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}] "
              f"[GAN loss: {loss_GAN.item():.4f}] [Pixel loss: {loss_pixel.item():.4f}] "
              f"[Histogram loss: {loss_histogram.item():.4f}] [Perceptual loss: {loss_perceptual.item():.4f}] "
              f"[Color loss: {loss_color.item():.4f}]")

    # Save model checkpoints every 5 epochs
    if (epoch + 1) % 5 == 0:
        torch.save(generator.state_dict(), os.path.join(SAVE_DIR, f'generator_epoch_{epoch+1}.pth'))
        torch.save(discriminator.state_dict(), os.path.join(SAVE_DIR, f'discriminator_epoch_{epoch+1}.pth'))

# Save final models after training completes
torch.save(generator.state_dict(), os.path.join(SAVE_DIR, 'generator_final.pth'))
torch.save(discriminator.state_dict(), os.path.join(SAVE_DIR, 'discriminator_final.pth'))

print("Training complete. Final models saved.")