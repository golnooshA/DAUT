import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import math
import csv
import time

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
    Ensures all components are normalized to the [0, 1] range.
    """
    # Ensure the image is in the [0, 255] range for LAB conversion
    image = (image * 255).astype(np.uint8)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)

    # Contrast of Luminance
    L_contrast = np.std(L) / 100.0  # Normalize by dividing by 100 to ensure [0, 1] range

    # Average chroma (colorfulness)
    chroma = np.sqrt(A**2 + B**2)
    chroma_mean = np.mean(chroma) / 128.0  # Normalize by dividing by 128 to ensure [0, 1] range

    # Saturation
    L_normalized = np.maximum(L / 255.0, 1e-6)  # Avoid division by zero
    saturation = chroma / L_normalized  # Saturation based on normalized luminance
    saturation_mean = np.mean(saturation) / 10.0  # Normalize by dividing by 10 to ensure [0, 1] range

    # Ensure all values are in the expected range
    L_contrast = np.clip(L_contrast, 0, 1)
    chroma_mean = np.clip(chroma_mean, 0, 1)
    saturation_mean = np.clip(saturation_mean, 0, 1)

    # UCIQE Calculation
    uciqe = 0.4680 * L_contrast + 0.2745 * chroma_mean + 0.2576 * saturation_mean
    return uciqe

def evaluate_metrics(enhanced_dir, gt_dir, output_csv):
    """
    Evaluate PSNR, SSIM, and UCIQE between enhanced images and ground truth images.
    Save all results to a .csv file.
    """
    start_time = time.time()

    psnr_values = []
    ssim_values = []
    uciqe_values = []

    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['Image', 'PSNR', 'SSIM', 'UCIQE']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        enhanced_images = sorted(os.listdir(enhanced_dir))
        gt_images = sorted(os.listdir(gt_dir))

        for i, (e_img_name, gt_img_name) in enumerate(zip(enhanced_images, gt_images)):
            e_img_path = os.path.join(enhanced_dir, e_img_name)
            gt_img_path = os.path.join(gt_dir, gt_img_name)

            # Load images
            enhanced_img = cv2.imread(e_img_path)
            gt_img = cv2.imread(gt_img_path)

            if enhanced_img is None or gt_img is None:
                print(f"Skipping {e_img_name} or {gt_img_name}: Could not load.")
                continue

            # Resize and normalize
            enhanced_img = cv2.cvtColor(cv2.resize(enhanced_img, (256, 256)), cv2.COLOR_BGR2RGB) / 255.0
            gt_img = cv2.cvtColor(cv2.resize(gt_img, (256, 256)), cv2.COLOR_BGR2RGB) / 255.0

            # Ensure shape match
            if enhanced_img.shape != gt_img.shape:
                print(f"Skipping {e_img_name}: Shape mismatch.")
                continue

            # Compute metrics
            psnr = calculate_psnr(enhanced_img, gt_img)
            ssim_value = ssim(enhanced_img, gt_img, channel_axis=2, win_size=11, data_range=1.0)
            uciqe_value = calculate_uciqe(enhanced_img)

            psnr_values.append(psnr)
            ssim_values.append(ssim_value)
            uciqe_values.append(uciqe_value)

            # Write to CSV
            writer.writerow({'Image': e_img_name, 'PSNR': round(psnr, 2),
                             'SSIM': round(ssim_value, 4), 'UCIQE': round(uciqe_value, 4)})

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(enhanced_images)} images...")

    # Averages
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_uciqe = np.mean(uciqe_values)

    print(f"\n--- Evaluation Complete in {time.time() - start_time:.2f} seconds ---")
    print(f"Average PSNR: {avg_psnr:.2f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average UCIQE: {avg_uciqe:.4f}")

    with open(output_csv, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'Image': 'Average', 'PSNR': round(avg_psnr, 2),
                         'SSIM': round(avg_ssim, 4), 'UCIQE': round(avg_uciqe, 4)})

if __name__ == "__main__":
    # Paths to the directories
    ENHANCED_DIR = './test/new_changes'    # Directory containing enhanced images
    GT_DIR = './dataset/UIEB/GT/'  # Directory containing ground truth images
    OUTPUT_CSV = './evaluation_results.csv'  # Path to save evaluation results

    # Run evaluation
    evaluate_metrics(ENHANCED_DIR, GT_DIR, OUTPUT_CSV)