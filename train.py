import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from net.Ushape_Trans import Generator, Discriminator
from torchvision.models import vgg19, VGG19_Weights
import torchvision.transforms as transforms

# Paths
PATH_INPUT = './dataset/UIEB/input'
PATH_DEPTH = './DPT/output_monodepth/UIEB_Changed/'
PATH_GT = './dataset/UIEB/GT/'
# SAVE_DIR = './save_model/'

SAVE_DIR = '/content/drive/My Drive/My_Datasets/save_model/'

os.makedirs(SAVE_DIR, exist_ok=True)

# Dataset class
class DepthDataset(Dataset):
    def __init__(self, input_path, depth_path, gt_path):
        self.input_list = sorted([f for f in os.listdir(input_path) if f.endswith('.png')])
        self.depth_list = sorted([f for f in os.listdir(depth_path) if f.endswith('.png')])
        self.gt_list = sorted([f for f in os.listdir(gt_path) if f.endswith('.png')])
        self.input_path = input_path
        self.depth_path = depth_path
        self.gt_path = gt_path
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        input_file = self.input_list[idx]
        input_img = cv2.imread(os.path.join(self.input_path, input_file))
        if input_img is None:
            raise FileNotFoundError(f"RGB image not found: {os.path.join(self.input_path, input_file)}")
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (256, 256)) / 255.0

        depth_file = input_file.replace('.png', 'depth.png')
        depth_img = cv2.imread(os.path.join(self.depth_path, depth_file), cv2.IMREAD_GRAYSCALE)
        if depth_img is None:
            raise FileNotFoundError(f"Depth image not found: {os.path.join(self.depth_path, depth_file)}")
        depth_img = cv2.resize(depth_img, (256, 256)) / 255.0

        real_A = torch.cat([
            torch.from_numpy(input_img.transpose(2, 0, 1)).float(),
            torch.from_numpy(depth_img).unsqueeze(0).float()
        ], dim=0)

        gt_img = cv2.imread(os.path.join(self.gt_path, input_file))
        if gt_img is None:
            raise FileNotFoundError(f"Ground truth image not found: {os.path.join(self.gt_path, input_file)}")
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        gt_img = cv2.resize(gt_img, (256, 256)) / 255.0

        real_B = torch.cat([
            torch.from_numpy(gt_img.transpose(2, 0, 1)).float(),
            torch.from_numpy(depth_img).unsqueeze(0).float()
        ], dim=0)

        return real_A, real_B

# Initialize weights
def initialize_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def filter_invalid_files(dataset):
    valid_indices = []
    for idx in range(len(dataset)):
        try:
            dataset[idx]
            valid_indices.append(idx)
        except FileNotFoundError as e:
            print(e)
    return valid_indices

if __name__ == "__main__":
    train_dataset = DepthDataset(PATH_INPUT, PATH_DEPTH, PATH_GT)
    valid_indices = filter_invalid_files(train_dataset)
    train_dataset = torch.utils.data.Subset(train_dataset, valid_indices)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    print(f"Loaded dataset with {len(train_dataset)} valid samples.")

    generator = Generator(input_channels=4, output_channels=4).cuda()
    discriminator = Discriminator(input_channels=8).cuda()

    generator.apply(initialize_weights)
    discriminator.apply(initialize_weights)

    criterion_GAN = nn.BCEWithLogitsLoss().cuda()
    criterion_pixelwise = nn.L1Loss().cuda()

    vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:16].cuda().eval()
    for param in vgg.parameters():
        param.requires_grad = False

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=1e-4)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=1e-4)

    scheduler_G = ReduceLROnPlateau(optimizer_G, mode='min', factor=0.5, patience=10, verbose=True)
    scheduler_D = ReduceLROnPlateau(optimizer_D, mode='min', factor=0.5, patience=10, verbose=True)

    n_epochs = 300
    for epoch in range(n_epochs):
        generator.train()
        discriminator.train()

        print(f"Starting Epoch {epoch+1}/{n_epochs}")

        for i, (real_A, real_B) in enumerate(train_loader):
            real_A, real_B = real_A.cuda(), real_B.cuda()

            optimizer_G.zero_grad()
            fake_B = generator(real_A)

            if isinstance(fake_B, (list, tuple)):
                fake_B = fake_B[-1]

            pred_fake = discriminator(torch.cat([real_A, fake_B], dim=1))
            loss_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))
            loss_pixel = criterion_pixelwise(fake_B, real_B)

            fake_features = vgg(fake_B[:, :3])
            real_features = vgg(real_B[:, :3])
            loss_perceptual = criterion_pixelwise(fake_features, real_features)

            loss_G = loss_GAN + 10 * loss_pixel + loss_perceptual
            loss_G.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()
            pred_real = discriminator(torch.cat([real_A, real_B], dim=1))
            pred_fake = discriminator(torch.cat([real_A, fake_B.detach()], dim=1))

            loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
            loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            loss_D.backward()
            optimizer_D.step()

            print(f"[Epoch {epoch+1}/{n_epochs}] [Batch {i+1}/{len(train_loader)}] [D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}]")

        scheduler_G.step(loss_G.item())
        scheduler_D.step(loss_D.item())

        if (epoch + 1) % 5 == 0:
            torch.save(generator.state_dict(), os.path.join(SAVE_DIR, f'generator_epoch_{epoch+1}.pth'))
            torch.save(discriminator.state_dict(), os.path.join(SAVE_DIR, f'discriminator_epoch_{epoch+1}.pth'))
            torch.save({
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict()
            }, os.path.join(SAVE_DIR, f'optimizers_epoch_{epoch+1}.pth'))

    print("Training complete. Models saved.")
