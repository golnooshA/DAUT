import os
import torch
import numpy as np
import cv2
from torch.autograd import Variable
from torchvision.utils import save_image
import torch.nn.functional as F
from net.Ushape_Trans import Generator  # Ensure correct path to your model

# Ensure no multiprocessing conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def split(img):
    """
    Create a multi-scale input by resizing the input image to different scales.
    """
    output = []
    output.append(F.interpolate(img, scale_factor=0.125))
    output.append(F.interpolate(img, scale_factor=0.25))
    output.append(F.interpolate(img, scale_factor=0.5))
    output.append(img)
    return output

def load_model(generator_path, device):
    """
    Load the pre-trained generator model with selective weight loading.
    """
    generator = Generator().to(device)
    generator_dict = generator.state_dict()
    
    # Load pretrained weights selectively
    pretrained_dict = torch.load(generator_path, map_location=device)
    filtered_dict = {
        k: v for k, v in pretrained_dict.items() if k in generator_dict and v.size() == generator_dict[k].size()
    }
    generator_dict.update(filtered_dict)
    generator.load_state_dict(generator_dict)
    
    generator.eval()
    return generator

def process_image(image_path, depth_path, dtype, device):
    """
    Process RGB and depth map images into a compatible tensor format.
    """
    img_rgb = cv2.imread(image_path)
    if img_rgb is None:
        raise FileNotFoundError(f"RGB image not found: {image_path}")
    img_rgb = cv2.resize(img_rgb, (256, 256))
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    
    img_depth = cv2.imread(depth_path, 0)  # Load depth as grayscale
    if img_depth is None:
        raise FileNotFoundError(f"Depth map not found: {depth_path}")
    img_depth = cv2.resize(img_depth, (256, 256)).reshape((256, 256, 1))
    img_depth = (img_depth / np.max(img_depth)) * 255  # Normalize depth

    img_combined = np.concatenate((img_rgb, img_depth), axis=2)  # Combine RGB and Depth
    img_combined = torch.from_numpy(img_combined.astype(dtype)).permute(2, 0, 1).unsqueeze(0) / 255.0
    return Variable(img_combined).to(device)

def extract_numeric_prefix(filename):
    """
    Extract the numeric prefix from a filename. If no valid prefix exists, return None.
    """
    parts = filename.split('_')[0]
    try:
        return int(parts)
    except ValueError:
        return None

def main():
    # Define paths
    path_images = './dataset/UIEB/input' # Update to your input images path
    path_depth = './DPT/output_monodepth/UIEB_Changed'  # Update to your depth maps path
    output_path = 'test/after' # Update to your output path
    generator_path = 'save_model/generator_final.pth'  # Corrected: Removed trailing slash
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = np.float32

    # Create output directory if not exists
    os.makedirs(output_path, exist_ok=True)

    # Load the pre-trained generator
    generator = load_model(generator_path, device)

    # Process images
    image_files = [f for f in os.listdir(path_images) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files = sorted(image_files, key=lambda x: extract_numeric_prefix(x) or float('inf'))

    for i, img_file in enumerate(image_files):
        try:
            img_path = os.path.join(path_images, img_file)
            depth_path = os.path.join(path_depth, f"{os.path.splitext(img_file)[0]}depth.png")

            # Process image and depth map
            input_tensor = process_image(img_path, depth_path, dtype, device)

            # Run the generator
            output = generator(input_tensor)
            enhanced_img = output[3].data

            # Save the enhanced image
            save_image(enhanced_img, os.path.join(output_path, img_file), nrow=1, normalize=True)
            print(f"Processed {img_file} ({i + 1}/{len(image_files)})")
        except Exception as e:
            print(f"Error processing {img_file}: {e}")

if __name__ == "__main__":
    main()
