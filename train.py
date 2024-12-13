import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from net.Ushape_Trans import Generator, Discriminator, weights_init_normal


# Paths
PATH_INPUT = './dataset/UIEB/input'
PATH_DEPTH = './DPT/output_monodepth/UIEB/'
PATH_GT = './dataset/UIEB/GT/'
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

    def __getitem__(self, idx):
        input_file = self.input_list[idx]
        input_img = cv2.imread(os.path.join(self.input_path, input_file))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (256, 256))
        input_tensor = torch.from_numpy(input_img).permute(2, 0, 1).float() / 255.0

        depth_file = input_file.replace('.jpg', '.png')
        depth_img = cv2.imread(os.path.join(self.depth_path, depth_file), cv2.IMREAD_GRAYSCALE)
        depth_img = cv2.resize(depth_img, (256, 256))
        depth_tensor = torch.from_numpy(depth_img).unsqueeze(0).float() / 255.0

        real_A = torch.cat([input_tensor, depth_tensor], dim=0)

        gt_img = cv2.imread(os.path.join(self.gt_path, input_file))
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        gt_img = cv2.resize(gt_img, (256, 256))
        real_B = torch.from_numpy(gt_img).permute(2, 0, 1).float() / 255.0

        return real_A, real_B

train_dataset = DepthDataset(PATH_INPUT, PATH_DEPTH, PATH_GT)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
print(f"Loaded dataset with {len(train_dataset)} valid samples.")

generator = Generator().cuda()
discriminator = Discriminator().cuda()
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

criterion_GAN = nn.MSELoss().cuda()
criterion_pixelwise = nn.L1Loss().cuda()

optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0005, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0005, betas=(0.5, 0.999))

n_epochs = 300
for epoch in range(n_epochs):
    print(f"Starting epoch {epoch+1}/{n_epochs}")
    for i, (real_A, real_B) in enumerate(train_loader):
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
        loss_G.backward(retain_graph=True)  # Retain graph for reuse
        optimizer_G.step()

        # Recompute fake_B to avoid reusing the graph
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
        loss_D.backward()  # No retain_graph needed here
        optimizer_D.step()

        # Save per-epoch models
        torch.save(generator.state_dict(), os.path.join(SAVE_DIR, f'generator_epoch_{epoch+1}.pth'))
        torch.save(discriminator.state_dict(), os.path.join(SAVE_DIR, f'discriminator_epoch_{epoch+1}.pth'))

        # Save final models after training
        if epoch == n_epochs - 1:
            torch.save(generator.state_dict(), os.path.join(SAVE_DIR, 'generator_final.pth'))
            torch.save(discriminator.state_dict(), os.path.join(SAVE_DIR, 'discriminator_final.pth'))
            print(f"Final generator and discriminator models saved to {SAVE_DIR}")
            
        print(f"[Epoch {epoch+1}/{n_epochs}] [Batch {i+1}/{len(train_loader)}] [D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}]")

