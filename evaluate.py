import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import math
import csv

def calculate_psnr(img1, img2):
    """
    Compute PSNR (Peak Signal-to-Noise Ratio) between two images.
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0  # Assuming images are normalized to [0, 1]
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

def calculate_uciqe(image):
    """
    Compute UCIQE (Underwater Color Image Quality Evaluation) for an image.
    UCIQE considers:
    - Chromaticity (colorfulness).
    - Contrast of luminance.
    - Saturation.
    """
    image = image.astype(np.float32) / 255.0
    lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)

    # Contrast of Luminance
    L_contrast = np.std(L) / 100.0

    # Average chroma (colorfulness)
    chroma = np.sqrt(A**2 + B**2)
    chroma_mean = np.mean(chroma)

    # Saturation
    saturation = chroma / (L + 1e-6)  # Avoid division by zero
    saturation_mean = np.mean(saturation)

    # UCIQE Calculation
    uciqe = 0.0282 * L_contrast + 0.2953 * chroma_mean + 0.3228 * saturation_mean
    return uciqe

def evaluate_metrics(enhanced_dir, gt_dir, output_csv):
    """
    Evaluate PSNR, SSIM, and UCIQE between enhanced images and ground truth images.
    Save all results to a .csv file.
    """
    psnr_values = []
    ssim_values = []
    uciqe_values = []

    # Prepare CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['Image', 'PSNR', 'SSIM', 'UCIQE']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # List all images in the enhanced and GT directories
        enhanced_images = sorted(os.listdir(enhanced_dir))
        gt_images = sorted(os.listdir(gt_dir))

        for e_img_name, gt_img_name in zip(enhanced_images, gt_images):
            e_img_path = os.path.join(enhanced_dir, e_img_name)
            gt_img_path = os.path.join(gt_dir, gt_img_name)

            # Load images
            enhanced_img = cv2.imread(e_img_path)
            gt_img = cv2.imread(gt_img_path)

            if enhanced_img is None or gt_img is None:
                print(f"Skipping: {e_img_name} or {gt_img_name} not loaded properly")
                continue

            # Resize to match dimensions
            enhanced_img = cv2.resize(enhanced_img, (256, 256))
            gt_img = cv2.resize(gt_img, (256, 256))

            # Convert to RGB and normalize to [0, 1]
            enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB) / 255.0
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB) / 255.0

            # Compute metrics
            psnr = calculate_psnr(enhanced_img, gt_img)
            ssim_value = ssim(enhanced_img, gt_img, channel_axis=2, win_size=5, data_range=1.0)
            uciqe_value = calculate_uciqe(enhanced_img)

            # Append results
            psnr_values.append(psnr)
            ssim_values.append(ssim_value)
            uciqe_values.append(uciqe_value)

            # Write to CSV
            writer.writerow({
                'Image': e_img_name,
                'PSNR': round(psnr, 2),
                'SSIM': round(ssim_value, 4),
                'UCIQE': round(uciqe_value, 4)
            })

            print(f"{e_img_name} - PSNR: {psnr:.2f}, SSIM: {ssim_value:.4f}, UCIQE: {uciqe_value:.4f}")

    # Report average metrics
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_uciqe = np.mean(uciqe_values)

    print("\n--- Evaluation Results ---")
    print(f"Average PSNR: {avg_psnr:.2f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average UCIQE: {avg_uciqe:.4f}")

    # Write averages to CSV
    with open(output_csv, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'Image': 'Average', 'PSNR': round(avg_psnr, 2), 
                         'SSIM': round(avg_ssim, 4), 'UCIQE': round(avg_uciqe, 4)})


if __name__ == "__main__":
    # Paths to the directories
    ENHANCED_DIR = 'test/after'      # Directory containing DAUT enhanced images
    GT_DIR = './dataset/UIEB/GT/'    # Directory containing ground truth images
    OUTPUT_CSV = './evaluation_results.csv'  # Path to save evaluation results

    # Run evaluation
    evaluate_metrics(ENHANCED_DIR, GT_DIR, OUTPUT_CSV)
