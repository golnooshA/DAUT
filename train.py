import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from net.Ushape_Trans import Generator, Discriminator, weights_init_normal
import numpy as np
from skimage.color import rgb2lab
from skimage.measure import shannon_entropy

# Paths
PATH_INPUT = './dataset/UIEB/input'
PATH_DEPTH = './DPT/output_monodepth/UIEB_Changed'
PATH_GT = './dataset/UIEB/GT/'
# SAVE_DIR = './save_model'
SAVE_DIR = '/content/drive/My Drive/My_Datasets/save_model/'

os.makedirs(SAVE_DIR, exist_ok=True)

# UCIQE Calculation
def calculate_uciqe(image):
    """Calculate UCIQE for a single image."""
    img_lab = rgb2lab(image)
    chroma = np.sqrt(img_lab[:, :, 1] ** 2 + img_lab[:, :, 2] ** 2)
    saturation = chroma / img_lab[:, :, 0].clip(1e-8, None)
    return np.mean(chroma) * 0.4680 + np.std(saturation) * 0.2745 + shannon_entropy(img_lab[:, :, 0]) * 0.2576

# Laplacian Kernel
def apply_laplacian(image):
    """Apply Laplacian operator for edge detection."""
    batch_size, channels, height, width = image.size()
    laplacian_kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    laplacian_kernel = laplacian_kernel.to(image.device)  # Move to the same device as the image
    laplacian_kernel = laplacian_kernel.expand(channels, 1, 3, 3)  # Expand to match input channels
    edge = F.conv2d(image, laplacian_kernel, padding=1, groups=channels)  # Apply depthwise convolution
    return edge

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

def color_consistency_loss(output, target):
    """Encourage similar color distributions."""
    return criterion_pixelwise(output.mean(dim=(2, 3)), target.mean(dim=(2, 3)))

def edge_aware_loss(output, target):
    """Preserve edges in generated images."""
    edge_output = apply_laplacian(output)
    edge_target = apply_laplacian(target)
    return criterion_pixelwise(edge_output, edge_target)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0005, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0005, betas=(0.5, 0.999))

# Resume from checkpoint
start_epoch = 0
generator_checkpoint = os.path.join(SAVE_DIR, 'generator_epoch_235.pth')
discriminator_checkpoint = os.path.join(SAVE_DIR, 'discriminator_epoch_235.pth')
optimizer_checkpoint = os.path.join(SAVE_DIR, 'optimizer_epoch_235.pth')

if os.path.exists(generator_checkpoint) and os.path.exists(discriminator_checkpoint) and os.path.exists(optimizer_checkpoint):
    generator.load_state_dict(torch.load(generator_checkpoint))
    checkpoint = torch.load(discriminator_checkpoint)
    model_dict = discriminator.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(pretrained_dict)
    discriminator.load_state_dict(model_dict)
    optimizer_states = torch.load(optimizer_checkpoint)
    optimizer_G.load_state_dict(optimizer_states['optimizer_G'])
    optimizer_D.load_state_dict(optimizer_states['optimizer_D'])
    print("Resuming from checkpoint at epoch 235.")
    start_epoch = 235
else:
    print("No checkpoint found. Starting from scratch.")

# Training Parameters
n_epochs = 300
save_freq = 5  # Save every 5 epochs

# Training Loop
for epoch in range(start_epoch, n_epochs):
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
        loss_color = color_consistency_loss(fake_B[-1], real_B)
        loss_edges = edge_aware_loss(fake_B[-1], real_B)

        loss_G = loss_GAN + 100 * loss_pixel + 10 * loss_color + 10 * loss_edges
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        print(f"[Epoch {epoch+1}/{n_epochs}] [Batch {i+1}/{len(train_loader)}] [D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}]")

    # Save per-epoch models and optimizers
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
