# import os
# import cv2
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, Dataset
# import torch.nn.functional as F
# from net.Ushape_Trans import Generator, Discriminator
# import numpy as np

# # Define weights_init_normal
# def weights_init_normal(m):
#     """Initialize weights with normal distribution."""
#     if hasattr(m, 'weight') and m.weight is not None:
#         if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
#             torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
#         elif isinstance(m, nn.BatchNorm2d):
#             torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
#             torch.nn.init.constant_(m.bias.data, 0.0)

# # Paths
# PATH_INPUT = './dataset/UIEB/input'
# PATH_DEPTH = './DPT/output_monodepth/UIEB_Changed'
# PATH_GT = './dataset/UIEB/GT/'
# SAVE_DIR = './save_model/'

# os.makedirs(SAVE_DIR, exist_ok=True)

# # Dataset
# class DepthDataset(Dataset):
#     def __init__(self, input_path, depth_path, gt_path):
#         self.input_list = sorted([f for f in os.listdir(input_path) if f.endswith('.png')])
#         self.depth_path = depth_path
#         self.gt_path = gt_path

#     def __len__(self):
#         return len(self.input_list)

#     def __getitem__(self, idx):
#         input_file = self.input_list[idx]
#         input_img = cv2.imread(os.path.join(PATH_INPUT, input_file))
#         input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
#         input_img = cv2.resize(input_img, (256, 256))
#         input_tensor = torch.from_numpy(input_img).permute(2, 0, 1).float() / 255.0

#         depth_file = input_file.replace('.png', '_depth.png')  # Correct usage of replace
#         depth_path = os.path.join(PATH_DEPTH, depth_file)
#         depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

#         if depth_img is None:
#             print(f"[WARNING] Missing or unreadable depth image: {depth_path}. Using placeholder.")
#             depth_img = np.zeros((256, 256), dtype=np.uint8)  # Placeholder

#         depth_img = cv2.resize(depth_img, (256, 256))
#         depth_tensor = torch.from_numpy(depth_img).unsqueeze(0).float() / 255.0

#         real_A = torch.cat([input_tensor, depth_tensor], dim=0)

#         gt_img = cv2.imread(os.path.join(PATH_GT, input_file))
#         gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
#         gt_img = cv2.resize(gt_img, (256, 256))
#         real_B = torch.from_numpy(gt_img).permute(2, 0, 1).float() / 255.0

#         return real_A, real_B

# train_dataset = DepthDataset(PATH_INPUT, PATH_DEPTH, PATH_GT)
# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# # Initialize models
# generator = Generator().cuda()
# discriminator = Discriminator().cuda()
# generator.apply(weights_init_normal)
# discriminator.apply(weights_init_normal)

# # Loss functions
# criterion_GAN = nn.MSELoss().cuda()
# criterion_pixelwise = nn.L1Loss().cuda()

# # Optimizers
# optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0005, betas=(0.5, 0.999))
# optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0005, betas=(0.5, 0.999))

# # Training loop with checkpoint saving
# n_epochs = 300
# save_freq = 5  # Save every 5 epochs

# for epoch in range(n_epochs):
#     for i, (real_A, real_B) in enumerate(train_loader):
#         real_A, real_B = real_A.cuda(), real_B.cuda()

#         # Generate fake_B
#         fake_B = generator(real_A)[-1]

#         # Create multi-scale inputs
#         real_B_scales = [
#             F.interpolate(real_B, size=(32, 32), mode="bilinear", align_corners=False),
#             F.interpolate(real_B, size=(64, 64), mode="bilinear", align_corners=False),
#             F.interpolate(real_B, size=(128, 128), mode="bilinear", align_corners=False),
#             real_B,
#         ]
#         real_A_scales = [
#             F.interpolate(real_A, size=(32, 32), mode="bilinear", align_corners=False),
#             F.interpolate(real_A, size=(64, 64), mode="bilinear", align_corners=False),
#             F.interpolate(real_A, size=(128, 128), mode="bilinear", align_corners=False),
#             real_A,
#         ]
#         fake_B_scales = [
#             F.interpolate(fake_B, size=(32, 32), mode="bilinear", align_corners=False),
#             F.interpolate(fake_B, size=(64, 64), mode="bilinear", align_corners=False),
#             F.interpolate(fake_B, size=(128, 128), mode="bilinear", align_corners=False),
#             fake_B,
#         ]

#         # Discriminator loss
#         pred_real = discriminator(real_B_scales, real_A_scales)
#         pred_fake = discriminator(fake_B_scales, real_A_scales)
#         loss_D = 0.5 * (criterion_GAN(pred_real, torch.ones_like(pred_real)) +
#                         criterion_GAN(pred_fake, torch.zeros_like(pred_fake)))
#         optimizer_D.zero_grad()
#         loss_D.backward()
#         optimizer_D.step()

#         # Recompute fake_B for generator loss
#         fake_B = generator(real_A)[-1]
#         fake_B_scales = [
#             F.interpolate(fake_B, size=(32, 32), mode="bilinear", align_corners=False),
#             F.interpolate(fake_B, size=(64, 64), mode="bilinear", align_corners=False),
#             F.interpolate(fake_B, size=(128, 128), mode="bilinear", align_corners=False),
#             fake_B,
#         ]

#         # Generator loss
#         pred_fake = discriminator(fake_B_scales, real_A_scales)
#         loss_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))
#         loss_pixel = criterion_pixelwise(fake_B, real_B)
#         loss_G = loss_GAN + 100 * loss_pixel
#         optimizer_G.zero_grad()
#         loss_G.backward()
#         optimizer_G.step()

#         print(f"[Epoch {epoch+1}/{n_epochs}] [Batch {i+1}/{len(train_loader)}] [D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}]")

#     # Save models every `save_freq` epochs
#     if (epoch + 1) % save_freq == 0 or epoch == n_epochs - 1:
#         generator_path = os.path.join(SAVE_DIR, f'generator_epoch_{epoch+1}.pth')
#         discriminator_path = os.path.join(SAVE_DIR, f'discriminator_epoch_{epoch+1}.pth')
#         torch.save(generator.state_dict(), generator_path)
#         torch.save(discriminator.state_dict(), discriminator_path)
#         print(f"Saved models for epoch {epoch+1}: {generator_path}, {discriminator_path}")

# # Save final models
# torch.save(generator.state_dict(), os.path.join(SAVE_DIR, 'generator_final.pth'))
# torch.save(discriminator.state_dict(), os.path.join(SAVE_DIR, 'discriminator_final.pth'))
# print("Final models saved as generator_final.pth and discriminator_final.pth")


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
PATH_DEPTH = './DPT/output_monodepth/UIEB/'
PATH_GT = './dataset/UIEB/GT/'
# SAVE_DIR = 'C:/Users/golno/OneDrive/Desktop/Depth-Aware-U-shape-Transformer/save_model/'
SAVE_DIR = '/content/drive/My Drive/My_Datasets/save_model/'

os.makedirs(SAVE_DIR, exist_ok=True)

class DepthDataset(Dataset):
    def __init__(self, input_path, depth_path, gt_path):
        self.input_list = sorted([f for f in os.listdir(input_path) if f.endswith('.png')])
        self.depth_list = sorted([f for f in os.listdir(depth_path) if f.endswith('.png')])
        self.gt_list = sorted([f for f in os.listdir(gt_path) if f.endswith('.png')])
        self.input_path = input_path
        self.depth_path = depth_path
        self.gt_path = gt_path

    def __len__(self):
        return len(self.input_list)

    # Updated DepthDataset
    def __getitem__(self, idx):
        input_file = self.input_list[idx]
        input_img = cv2.imread(os.path.join(self.input_path, input_file))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (256, 256))
        input_tensor = torch.from_numpy(input_img).permute(2, 0, 1).float() / 255.0

        depth_file = input_file.replace('.jpg', '.png')
        depth_img = cv2.imread(os.path.join(self.depth_path, depth_file), cv2.IMREAD_GRAYSCALE)
        depth_img = cv2.resize(depth_img, (256, 256))
        depth_img = (depth_img / (np.max(depth_img) + 1e-8)) * 255  # Normalize to 0-255
        depth_tensor = torch.from_numpy(depth_img).unsqueeze(0).float() / 255.0

        # Ensure input_tensor has 3 channels and depth_tensor is concatenated
        assert input_tensor.shape[0] == 3, f"Expected RGB input to have 3 channels, got {input_tensor.shape[0]}"
        assert depth_tensor.shape[0] == 1, f"Expected depth input to have 1 channel, got {depth_tensor.shape[0]}"

        real_A = torch.cat([input_tensor, depth_tensor], dim=0)
        assert real_A.shape[0] == 4, f"Expected real_A to have 4 channels, got {real_A.shape[0]}"

        gt_img = cv2.imread(os.path.join(self.gt_path, input_file))
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        gt_img = cv2.resize(gt_img, (256, 256))
        real_B = torch.from_numpy(gt_img).permute(2, 0, 1).float() / 255.0

        return real_A, real_B

train_dataset = DepthDataset(PATH_INPUT, PATH_DEPTH, PATH_GT)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
print(f"Loaded dataset with {len(train_dataset)} valid samples.")

# Initialize models
generator = Generator().cuda()
discriminator = Discriminator().cuda()
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Loss functions
criterion_GAN = nn.MSELoss().cuda()
criterion_pixelwise = nn.L1Loss().cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0005, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0005, betas=(0.5, 0.999))

# Resume from checkpoint
start_epoch = 0
n_epochs = 300
save_freq = 5  # Save every 5 epochs

for epoch in range(start_epoch, n_epochs):
    for i, (real_A, real_B) in enumerate(train_loader):
        print(f"real_A shape: {real_A.shape}, real_B shape: {real_B.shape}")       
        real_A, real_B = real_A.cuda(), real_B.cuda()

        # Multi-scale real_B and real_A
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

        # Generate fake_B and create multi-scale versions
        fake_B = generator(real_A)
        fake_B_scales = [
            F.interpolate(fake_B[-1], size=(32, 32)),
            F.interpolate(fake_B[-1], size=(64, 64)),
            F.interpolate(fake_B[-1], size=(128, 128)),
            fake_B[-1],
        ]

        # Discriminator forward passes
        pred_fake = discriminator(fake_B_scales, real_A_scales)

        # Generator loss
        loss_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))
        loss_pixel = criterion_pixelwise(fake_B[-1], real_B)
        loss_G = loss_GAN + 100 * loss_pixel
        optimizer_G.zero_grad()
        loss_G.backward(retain_graph=True)
        optimizer_G.step()

        # Recompute fake_B
        fake_B = generator(real_A)
        fake_B_scales = [
            F.interpolate(fake_B[-1], size=(32, 32)),
            F.interpolate(fake_B[-1], size=(64, 64)),
            F.interpolate(fake_B[-1], size=(128, 128)),
            fake_B[-1],
        ]

        pred_real = discriminator(real_B_scales, real_A_scales)
        pred_fake = discriminator(fake_B_scales, real_A_scales)

        # Discriminator loss
        loss_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
        loss_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
        loss_D = 0.5 * (loss_real + loss_fake)
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        print(f"[Epoch {epoch+1}/{n_epochs}] [Batch {i+1}/{len(train_loader)}] [D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}]")

    # Save per-epoch models every save_freq epochs
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