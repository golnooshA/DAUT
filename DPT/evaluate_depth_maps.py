import os
import cv2
import numpy as np
import math
from skimage.metrics import structural_similarity as ssim
import csv

def calculate_psnr(img1, img2):
    """
    Compute PSNR (Peak Signal-to-Noise Ratio) between two images.
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  # Perfect match
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

def evaluate_depth_maps(depth_dir_before, depth_dir_after, output_csv):
    """
    Evaluate depth map quality before and after changes.
    """
    psnr_values = []
    ssim_values = []

    # Prepare CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['Image', 'PSNR', 'SSIM']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        depth_images_before = sorted([f for f in os.listdir(depth_dir_before) if f.endswith('.png')])
        depth_images_after = sorted([f for f in os.listdir(depth_dir_after) if f.endswith('.png')])

        for img_before, img_after in zip(depth_images_before, depth_images_after):
            path_before = os.path.join(depth_dir_before, img_before)
            path_after = os.path.join(depth_dir_after, img_after)

            # Load images
            depth_before = cv2.imread(path_before, cv2.IMREAD_GRAYSCALE)
            depth_after = cv2.imread(path_after, cv2.IMREAD_GRAYSCALE)

            if depth_before is None or depth_after is None:
                print(f"Skipping: {img_before} or {img_after} not found")
                continue

            # Resize to the same size if needed
            if depth_before.shape != depth_after.shape:
                depth_after = cv2.resize(depth_after, depth_before.shape[::-1])

            # Compute metrics
            psnr = calculate_psnr(depth_before, depth_after)
            ssim_value = ssim(depth_before, depth_after, data_range=255)

            # Append results
            psnr_values.append(psnr)
            ssim_values.append(ssim_value)

            # Write to CSV
            writer.writerow({'Image': img_before, 'PSNR': round(psnr, 2), 'SSIM': round(ssim_value, 4)})
            print(f"{img_before} - PSNR: {psnr:.2f}, SSIM: {ssim_value:.4f}")

    # Report average metrics
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)

    print("\n--- Evaluation Results ---")
    print(f"Average PSNR: {avg_psnr:.2f}")
    print(f"Average SSIM: {avg_ssim:.4f}")

    # Write averages to CSV
    with open(output_csv, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'Image': 'Average', 'PSNR': round(avg_psnr, 2), 'SSIM': round(avg_ssim, 4)})

if __name__ == "__main__":
    # Paths to depth map directories
    DEPTH_DIR_BEFORE = './output_monodepth/UIEB'  # Replace with path to depth maps before changes
    DEPTH_DIR_AFTER = './output_monodepth/UIEB/changed'  # Replace with path to depth maps after changes
    OUTPUT_CSV = './depth_map_evaluation.csv'  # Path to save evaluation results

    # Run evaluation
    evaluate_depth_maps(DEPTH_DIR_BEFORE, DEPTH_DIR_AFTER, OUTPUT_CSV)
