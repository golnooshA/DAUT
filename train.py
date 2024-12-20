import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from net.Ushape_Trans import Generator, Discriminator, weights_init_normal
import numpy as np

# Paths
PATH_INPUT = './dataset/UIEB/input'
PATH_DEPTH = './DPT/output_monodepth/UIEB_Changed'
PATH_GT = './dataset/UIEB/GT/'
# SAVE_DIR = './save_model'
SAVE_DIR = '/content/drive/My Drive/My_Datasets/save_model/'

os.makedirs(SAVE_DIR, exist_ok=True)

# Dataset Class
class DepthDataset(Dataset):
    def __init__(self, input_path, depth_path, gt_path):
        self.input_list = sorted([f for f in os.listdir(input_path) if f.endswith(('.png', '.jpg'))])
        self.depth_path = depth_path
        self.gt_path = gt_path

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        input_file = self.input_list[idx]
        
        # Load input image
        input_img = cv2.imread(os.path.join(PATH_INPUT, input_file))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (256, 256))
        input_tensor = torch.from_numpy(input_img).permute(2, 0, 1).float() / 255.0

        # Load depth image
        depth_file = input_file.replace('.jpg', '.png').replace('.png', 'depth.png')
        depth_img = cv2.imread(os.path.join(PATH_DEPTH, depth_file), cv2.IMREAD_GRAYSCALE)
        if depth_img is None:
            print(f"[WARNING] Missing depth image: {depth_file}. Using placeholder.")
            depth_img = np.zeros((256, 256), dtype=np.uint8)
        depth_img = cv2.resize(depth_img, (256, 256))
        depth_tensor = torch.from_numpy(depth_img).unsqueeze(0).float() / 255.0

        # Combine input and depth tensors
        real_A = torch.cat([input_tensor, depth_tensor], dim=0)

        # Load GT image
        gt_img = cv2.imread(os.path.join(PATH_GT, input_file))
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        gt_img = cv2.resize(gt_img, (256, 256))
        real_B = torch.from_numpy(gt_img).permute(2, 0, 1).float() / 255.0

        return real_A, real_B


# Initialize Dataset and DataLoader
train_dataset = DepthDataset(PATH_INPUT, PATH_DEPTH, PATH_GT)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
print(f"Loaded dataset with {len(train_dataset)} valid samples.")

# Initialize Models
generator = Generator().cuda()
discriminator = Discriminator().cuda()

# Loss Functions
criterion_GAN = nn.MSELoss().cuda()
criterion_pixelwise = nn.L1Loss().cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0005, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0005, betas=(0.5, 0.999))

# Checkpoint Handling
start_epoch = 0
generator_checkpoint = os.path.join(SAVE_DIR, 'generator_epoch_25.pth')
discriminator_checkpoint = os.path.join(SAVE_DIR, 'discriminator_epoch_25.pth')
optimizer_checkpoint = os.path.join(SAVE_DIR, 'optimizer_epoch_25.pth')

if os.path.exists(generator_checkpoint) and os.path.exists(discriminator_checkpoint) and os.path.exists(optimizer_checkpoint):
    generator.load_state_dict(torch.load(generator_checkpoint))
    print(f"Loaded generator checkpoint from epoch 25.")
    
    checkpoint = torch.load(discriminator_checkpoint)
    model_dict = discriminator.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(pretrained_dict)
    discriminator.load_state_dict(model_dict)
    print(f"Loaded discriminator checkpoint with partial matching from epoch 25.")
    
    optimizer_state = torch.load(optimizer_checkpoint)
    optimizer_G.load_state_dict(optimizer_state['optimizer_G'])
    optimizer_D.load_state_dict(optimizer_state['optimizer_D'])
    print("Loaded optimizer states.")
    start_epoch = 25  # Update start_epoch only if checkpoints exist
else:
    print("No checkpoint found. Starting from scratch.")
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

# Training Parameters
n_epochs = 300
save_freq = 5  # Save every 5 epochs

# Training Loop
for epoch in range(start_epoch, n_epochs):  # Start from the resumed epoch
    for i, (real_A, real_B) in enumerate(train_loader):
        real_A, real_B = real_A.cuda(), real_B.cuda()

        # Generate fake_B and multi-scale inputs
        fake_B = generator(real_A)
        fake_B_scales = [F.interpolate(fake_B[-1], size=(s, s)) for s in [32, 64, 128, 256]]
        real_B_scales = [F.interpolate(real_B, size=(s, s)) for s in [32, 64, 128, 256]]
        real_A_scales = [F.interpolate(real_A, size=(s, s)) for s in [32, 64, 128, 256]]

        # Discriminator Forward Pass
        pred_real = discriminator(real_B_scales, real_A_scales)
        pred_fake = discriminator(fake_B_scales, real_A_scales)

        # Discriminator Loss
        loss_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
        loss_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
        loss_D = 0.5 * (loss_real + loss_fake)
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # Recompute fake_B for generator loss
        fake_B = generator(real_A)
        fake_B_scales = [F.interpolate(fake_B[-1], size=(s, s)) for s in [32, 64, 128, 256]]

        # Generator Loss
        pred_fake = discriminator(fake_B_scales, real_A_scales)
        loss_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))
        loss_pixel = criterion_pixelwise(fake_B[-1], real_B)
        loss_G = loss_GAN + 100 * loss_pixel
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        print(f"[Epoch {epoch+1}/{n_epochs}] [Batch {i+1}/{len(train_loader)}] [D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}]")

    # Save per-epoch models every `save_freq` epochs
    if (epoch + 1) % save_freq == 0 or epoch == n_epochs - 1:
        torch.save(generator.state_dict(), os.path.join(SAVE_DIR, f'generator_epoch_{epoch+1}.pth'))
        torch.save(discriminator.state_dict(), os.path.join(SAVE_DIR, f'discriminator_epoch_{epoch+1}.pth'))
        torch.save({'optimizer_G': optimizer_G.state_dict(), 'optimizer_D': optimizer_D.state_dict()},
                   os.path.join(SAVE_DIR, f'optimizer_epoch_{epoch+1}.pth'))
        print(f"Saved models and optimizers for epoch {epoch+1}.")

# Save final models
torch.save(generator.state_dict(), os.path.join(SAVE_DIR, 'generator_final.pth'))
torch.save(discriminator.state_dict(), os.path.join(SAVE_DIR, 'discriminator_final.pth'))
print(f"Final generator and discriminator models saved to {SAVE_DIR}.")
