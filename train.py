import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from net.Ushape_Trans import Generator, Discriminator, weights_init_normal

# Paths
PATH_INPUT = './dataset/UIEB/input'
PATH_DEPTH = './DPT/output_monodepth/UIEB_Changed'
PATH_GT = './dataset/UIEB/GT/'
SAVE_DIR = './save_model/'

os.makedirs(SAVE_DIR, exist_ok=True)

# Dataset Class
class DepthDataset(Dataset):
    def __init__(self, input_path, depth_path, gt_path):
        self.input_list = sorted([f for f in os.listdir(input_path) if f.endswith('.png')])
        self.depth_list = sorted([f for f in os.listdir(depth_path) if f.endswith('_depth.png')])
        self.gt_list = sorted([f for f in os.listdir(gt_path) if f.endswith('.png')])
        self.input_path = input_path
        self.depth_path = depth_path
        self.gt_path = gt_path

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        input_file = self.input_list[idx]

        # Load input image
        input_img = cv2.imread(os.path.join(self.input_path, input_file))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (256, 256))
        input_tensor = torch.from_numpy(input_img).permute(2, 0, 1).float() / 255.0

        # Load depth map
        depth_file = input_file.replace('.png', '_depth.png')  # Match depth naming convention
        depth_path = os.path.join(self.depth_path, depth_file)
        if not os.path.exists(depth_path):  # Handle missing depth map
            raise FileNotFoundError(f"Depth map not found for {depth_file}")

        depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        if depth_img is None:  # If OpenCV fails to read the depth map
            raise ValueError(f"Unable to read depth map: {depth_path}")
        depth_img = cv2.resize(depth_img, (256, 256))
        depth_tensor = torch.from_numpy(depth_img).unsqueeze(0).float() / 255.0

        # Combine input and depth
        real_A = torch.cat([input_tensor, depth_tensor], dim=0)

        # Load ground truth image
        gt_img = cv2.imread(os.path.join(self.gt_path, input_file))
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        gt_img = cv2.resize(gt_img, (256, 256))
        real_B = torch.from_numpy(gt_img).permute(2, 0, 1).float() / 255.0

        return real_A, real_B



# DataLoader
train_dataset = DepthDataset(PATH_INPUT, PATH_DEPTH, PATH_GT)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
print(f"Loaded dataset with {len(train_dataset)} valid samples.")

# Initialize Models
generator = Generator().cuda()
discriminator = Discriminator().cuda()
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Loss Functions
criterion_GAN = nn.MSELoss().cuda()
criterion_pixelwise = nn.L1Loss().cuda()

# Depth Loss
class DepthLoss(nn.Module):
    def __init__(self):
        super(DepthLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, generated_depth, gt_depth):
        return self.mse(generated_depth, gt_depth)

depth_loss_fn = DepthLoss().cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0005, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0005, betas=(0.5, 0.999))

# Training Loop
n_epochs = 300
save_freq = 5  # Save every 5 epochs

for epoch in range(n_epochs):  # Start from 0 without resuming
    print(f"Starting epoch {epoch+1}/{n_epochs}")
    for i, (real_A, real_B) in enumerate(train_loader):
        real_A, real_B = real_A.cuda(), real_B.cuda()

        # Prepare multi-scale versions of real_B and real_A
        real_B_scales = [
            F.interpolate(real_B, size=(32, 32)),
            F.interpolate(real_B, size=(64, 64)),
            F.interpolate(real_B, size=(128, 128)),
            real_B,
        ]

        real_A_scales = [
            F.interpolate(real_A, size=(32, 32)),
            F.interpolate(real_A, size=(64, 64)),
            F.interpolate(real_A, size=(128, 128)),
            real_A,
        ]

        # Train Generator
        optimizer_G.zero_grad()
        fake_B = generator(real_A)
        pred_fake = discriminator(fake_B, real_A_scales)

        # Compute losses
        loss_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))
        loss_pixel = criterion_pixelwise(fake_B[-1], real_B)
        loss_depth = depth_loss_fn(fake_B[-1][:, 3:, :, :], real_A[:, 3:, :, :])
        loss_G = loss_GAN + 100 * loss_pixel + 10 * loss_depth

        # Backpropagate and optimize
        loss_G.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()
        pred_real = discriminator(real_B_scales, real_A_scales)
        pred_fake = discriminator(fake_B, real_A_scales)

        # Discriminator losses
        loss_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
        loss_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
        loss_D = 0.5 * (loss_real + loss_fake)

        # Backpropagate and optimize
        loss_D.backward()
        optimizer_D.step()

        print(f"[Epoch {epoch+1}/{n_epochs}] [Batch {i+1}/{len(train_loader)}] [D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}]")

    # Save models every `save_freq` epochs
    if (epoch + 1) % save_freq == 0 or epoch == n_epochs - 1:
        torch.save(generator.state_dict(), os.path.join(SAVE_DIR, f'generator_epoch_{epoch+1}.pth'))
        torch.save(discriminator.state_dict(), os.path.join(SAVE_DIR, f'discriminator_epoch_{epoch+1}.pth'))
        torch.save({'optimizer_G': optimizer_G.state_dict(), 'optimizer_D': optimizer_D.state_dict()},
                   os.path.join(SAVE_DIR, f'optimizer_epoch_{epoch+1}.pth'))
        print(f"Saved models for epoch {epoch+1}.")

# Save final models
torch.save(generator.state_dict(), os.path.join(SAVE_DIR, 'generator_final.pth'))
torch.save(discriminator.state_dict(), os.path.join(SAVE_DIR, 'discriminator_final.pth'))
print(f"Final generator and discriminator models saved to {SAVE_DIR}.")
