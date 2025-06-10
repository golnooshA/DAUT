import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image

from net.Ushape_Trans import Generator  # Ensure this path is correct

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def normalize_depth(img_depth):
    """Normalize depth map to the range [0, 1]."""
    img_depth = img_depth.astype(np.float32)
    min_depth = np.min(img_depth)
    max_depth = np.max(img_depth)
    if max_depth != min_depth:
        img_depth = (img_depth - min_depth) / (max_depth - min_depth)
    return img_depth

def enhance_image(image):
    """Post-process enhanced images."""
    # Convert from tensor to NumPy and scale back to 0-255
    image = image.detach().cpu().numpy().transpose(1, 2, 0) * 255.0
    image = image.astype(np.uint8)

    # Reduce saturation and avoid over-sharpening
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[..., 1] = cv2.add(hsv[..., 1], 10)  # Adjust saturation (+10)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # Gentle contrast enhancement
    alpha = 1.1  # Contrast
    beta = 5     # Brightness
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # Return the processed image
    return torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)

def process_image(image_path, depth_path, dtype, device):
    """Process RGB and depth map images into input tensors."""
    img_rgb = cv2.imread(image_path)
    if img_rgb is None:
        raise FileNotFoundError(f"RGB image not found: {image_path}")
    img_rgb = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB) / 255.0

    img_depth = cv2.imread(depth_path, 0)  # Grayscale depth map
    if img_depth is None:
        raise FileNotFoundError(f"Depth map not found: {depth_path}")
    img_depth = cv2.resize(img_depth, (256, 256), interpolation=cv2.INTER_AREA)
    img_depth = normalize_depth(img_depth).reshape((256, 256, 1))

    img_combined = np.concatenate((img_rgb, img_depth), axis=2)
    img_combined = torch.from_numpy(img_combined.astype(dtype)).permute(2, 0, 1).unsqueeze(0)
    return Variable(img_combined).to(device)

def load_model(generator_path, device):
    """Load generator model."""
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    generator.eval()
    return generator

def main():
    path_images = './dataset/UIEB/input'
    path_depth = './DPT/output_monodepth/UIEB_Changed'
    output_path = './test/good'
    generator_path = './save_model/generator_final.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = np.float32
    os.makedirs(output_path, exist_ok=True)

    generator = load_model(generator_path, device)

    image_files = sorted([f for f in os.listdir(path_images) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    for img_file in image_files:
        try:
            img_path = os.path.join(path_images, img_file)
            depth_path = os.path.join(path_depth, f"{os.path.splitext(img_file)[0]}depth.png")

            input_tensor = process_image(img_path, depth_path, dtype, device)
            output = generator(input_tensor)[-1]  # Use the final output
            output = torch.clamp(output, 0, 1)  # Clamp values to [0, 1]

            # Apply post-processing
            enhanced_img = enhance_image(output.squeeze(0))

            save_image(enhanced_img, os.path.join(output_path, img_file), nrow=1, normalize=True)
            print(f"Processed: {img_file}")

        except Exception as e:
            print(f"Error processing {img_file}: {e}")

if __name__ == "__main__":
    main()
