#!/usr/bin/env python
# coding: utf-8

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image
from net.Ushape_Trans import Generator, Discriminator
from net.utils import batch_PSNR
from loss.LAB import lab_Loss
from loss.LCH import lch_Loss
from utility import data as data_utils
import pytorch_ssim
import time
import datetime

# Set environment variables
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Define paths (update these with your dataset paths)
PATH_INPUT = './dataset/LSUI/'        # Input images directory
PATH_DEPTH = './DPT/output_monodepth/LSUI/'        # Depth maps directory
PATH_GT = './dataset/LSUI/GT/'       # Ground truth directory
SAVE_DIR = './save_model/'           # Directory to save models

# Ensure the save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

# Dataset preprocessing
def load_dataset(input_path, depth_path, gt_path):
    """
    Load input images, depth maps, and ground truth images for training.
    """
    input_images = []
    gt_images = []
    input_list = sorted(os.listdir(input_path))
    
    for filename in input_list:
        # Load input image
        img_input = cv2.imread(os.path.join(input_path, filename))
        img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
        img_input = cv2.resize(img_input, (256, 256))
        
        # Load depth map
        depth_file = filename.split('.')[0] + ".png"
        img_depth = cv2.imread(os.path.join(depth_path, depth_file), 0)
        img_depth = cv2.resize(img_depth, (256, 256)).reshape((256, 256, 1))
        img_depth = (img_depth / np.max(img_depth)) * 255
        
        # Concatenate input and depth
        combined_input = np.concatenate((img_input, img_depth), axis=2)
        input_images.append(combined_input)
        
        # Load ground truth
        img_gt = cv2.imread(os.path.join(gt_path, filename))
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        img_gt = cv2.resize(img_gt, (256, 256))
        gt_images.append(img_gt)
    
    # Convert to NumPy arrays
    input_images = np.array(input_images).astype('float32') / 255.0
    gt_images = np.array(gt_images).astype('float32') / 255.0
    
    # Convert to PyTorch tensors
    input_tensors = torch.from_numpy(input_images).permute(0, 3, 1, 2)
    gt_tensors = torch.from_numpy(gt_images).permute(0, 3, 1, 2)
    
    return input_tensors, gt_tensors

# Load training data
X_train, y_train = load_dataset(PATH_INPUT, PATH_DEPTH, PATH_GT)

# DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)

# Initialize models, loss functions, and optimizers
generator = Generator().cuda()
discriminator = Discriminator().cuda()
criterion_GAN = nn.MSELoss().cuda()
criterion_pixelwise = nn.MSELoss().cuda()
L_lab = lab_Loss().cuda()
L_lch = lch_Loss().cuda()
SSIM = pytorch_ssim.SSIM().cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0005, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0005, betas=(0.5, 0.999))

# Training loop
n_epochs = 300
for epoch in range(n_epochs):
    for i, (real_A, real_B) in enumerate(train_loader):
        real_A, real_B = real_A.cuda(), real_B.cuda()
        
        # Adversarial ground truths
        valid = Variable(torch.ones((real_A.size(0), 1, 16, 16)).cuda(), requires_grad=False)
        fake = Variable(torch.zeros((real_A.size(0), 1, 16, 16)).cuda(), requires_grad=False)
        
        # Train Generator
        optimizer_G.zero_grad()
        fake_B = generator(real_A)
        loss_GAN = criterion_GAN(discriminator(fake_B, real_A), valid)
        loss_pixel = criterion_pixelwise(fake_B, real_B)
        loss_G = loss_GAN + 100 * loss_pixel
        loss_G.backward()
        optimizer_G.step()
        
        # Train Discriminator
        optimizer_D.zero_grad()
        pred_real = discriminator(real_B, real_A)
        loss_real = criterion_GAN(pred_real, valid)
        pred_fake = discriminator(fake_B.detach(), real_A)
        loss_fake = criterion_GAN(pred_fake, fake)
        loss_D = 0.5 * (loss_real + loss_fake)
        loss_D.backward()
        optimizer_D.step()
        
        # Logging
        print(f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(train_loader)}] [D loss: {loss_D.item()}] [G loss: {loss_G.item()}]")

    # Save model checkpoints after every epoch
    torch.save(generator.state_dict(), os.path.join(SAVE_DIR, f"generator_epoch_{epoch}.pth"))
    torch.save(discriminator.state_dict(), os.path.join(SAVE_DIR, f"discriminator_epoch_{epoch}.pth"))

# Save final models
torch.save(generator.state_dict(), os.path.join(SAVE_DIR, "generator.pth"))
torch.save(discriminator.state_dict(), os.path.join(SAVE_DIR, "discriminator.pth"))
