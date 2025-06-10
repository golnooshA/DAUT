import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
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

# Initialize Models
generator = Generator(input_channels=4, output_channels=4).cuda()
discriminator = Discriminator(input_channels=8).cuda()

# Apply Xavier initialization
def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight)
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)

generator.apply(initialize_weights)
discriminator.apply(initialize_weights)

# Loss Functions
criterion_GAN = nn.MSELoss().cuda()
criterion_pixelwise = nn.L1Loss().cuda()

def histogram_loss(output, target):
    """KL-Divergence-based histogram loss with normalization."""
    hist_output = torch.histc(output, bins=256, min=0, max=1)
    hist_target = torch.histc(target, bins=256, min=0, max=1)

    hist_output = hist_output / hist_output.sum()  # Normalize
    hist_target = hist_target / hist_target.sum()  # Normalize

    hist_loss = F.kl_div(hist_output.log(), hist_target, reduction='sum')  # KL Divergence loss
    return hist_loss

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training Loop
n_epochs = 300
train_dataset = DepthDataset(PATH_INPUT, PATH_DEPTH, PATH_GT)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)

for epoch in range(n_epochs):
    for i, (real_A, real_B) in enumerate(train_loader):
        real_A, real_B = real_A.cuda(), real_B.cuda()

        # Train Discriminator
        optimizer_D.zero_grad()
        fake_B = generator(real_A)[-1]  # Use last output only
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

        # Scale loss terms
        loss_G = loss_GAN + 10 * loss_pixel + 0.5 * loss_histogram  # Increase histogram weight
        loss_G.backward()

        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=0.5)
        optimizer_G.step()

        # Log individual loss terms for debugging
        print(f"[Epoch {epoch+1}/{n_epochs}] [Batch {i+1}/{len(train_loader)}] "
              f"[D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}] "
              f"[GAN loss: {loss_GAN.item():.4f}] [Pixel loss: {loss_pixel.item():.4f}] [Histogram loss: {loss_histogram.item():.6f}]")

    # Save model checkpoints every 5 epochs
    if (epoch + 1) % 5 == 0:
        torch.save(generator.state_dict(), os.path.join(SAVE_DIR, f'generator_epoch_{epoch+1}.pth'))
        torch.save(discriminator.state_dict(), os.path.join(SAVE_DIR, f'discriminator_epoch_{epoch+1}.pth'))

# Save final models after training completes
torch.save(generator.state_dict(), os.path.join(SAVE_DIR, 'generator_final.pth'))
torch.save(discriminator.state_dict(), os.path.join(SAVE_DIR, 'discriminator_final.pth'))

print("Training complete. Final models saved.")
