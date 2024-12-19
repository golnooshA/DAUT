import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from net.Ushape_Trans import Generator, Discriminator, weights_init_normal

def main():
    # Paths
    PATH_INPUT = './dataset/UIEB/input'
    PATH_DEPTH = './DPT/output_monodepth/UIEB_Changed'
    PATH_GT = './dataset/UIEB/GT/'
    SAVE_DIR = '/content/drive/My Drive/My_Datasets/save_model/'

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
            depth_file = input_file.replace('.png', '_depth.png')
            depth_path = os.path.join(self.depth_path, depth_file)
            if not os.path.exists(depth_path):
                raise FileNotFoundError(f"Depth map not found for {depth_file}")

            depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
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

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0005, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0005, betas=(0.5, 0.999))

    # Mixed Precision Scaler
    scaler = torch.amp.GradScaler()

    # Training Loop
    n_epochs = 300
    save_freq = 5  # Save every 5 epochs
    accumulation_steps = 4

    for epoch in range(n_epochs):
        print(f"Starting epoch {epoch + 1}/{n_epochs}")
        generator.train()
        discriminator.train()
        
        for i, (real_A, real_B) in enumerate(train_loader):
            real_A, real_B = real_A.cuda(), real_B.cuda()

            ### ----------- Train Generator ----------- ###
            optimizer_G.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                # Generate fake images
                fake_B = generator(real_A)
                fake_B = F.interpolate(fake_B, size=real_B.shape[2:], mode="bilinear", align_corners=False)
                
                # Discriminator's prediction for fake images
                pred_fake = discriminator([fake_B], [real_A])
                
                # Losses for Generator
                loss_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))
                loss_pixel = criterion_pixelwise(fake_B, real_B)
                loss_G = loss_GAN + 100 * loss_pixel

            scaler.scale(loss_G / accumulation_steps).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer_G)
                scaler.update()

            ### ----------- Train Discriminator ----------- ###
            optimizer_D.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                # Real images (discriminator)
                pred_real = discriminator([real_B], [real_A])
                
                # Fake images (detached to stop gradients flowing back to the generator)
                fake_B_detached = fake_B.detach()
                pred_fake = discriminator([fake_B_detached], [real_A])
                
                # Losses for Discriminator
                loss_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
                loss_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
                loss_D = 0.5 * (loss_real + loss_fake)

            scaler.scale(loss_D / accumulation_steps).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer_D)
                scaler.update()

            # Print progress
            print(
                f"[Epoch {epoch + 1}/{n_epochs}] [Batch {i + 1}/{len(train_loader)}] "
                f"[D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}]"
            )

        # Save models every `save_freq` epochs
        if (epoch + 1) % save_freq == 0 or epoch == n_epochs - 1:
            torch.save(generator.state_dict(), os.path.join(SAVE_DIR, f'generator_epoch_{epoch + 1}.pth'))
            torch.save(discriminator.state_dict(), os.path.join(SAVE_DIR, f'discriminator_epoch_{epoch + 1}.pth'))
            print(f"Saved models for epoch {epoch + 1}.")

    # Save final models
    torch.save(generator.state_dict(), os.path.join(SAVE_DIR, 'generator_final.pth'))
    torch.save(discriminator.state_dict(), os.path.join(SAVE_DIR, 'discriminator_final.pth'))
    print(f"Final generator and discriminator models saved to {SAVE_DIR}.")


if __name__ == "__main__":
    main()
