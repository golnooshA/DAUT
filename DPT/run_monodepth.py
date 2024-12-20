"""Compute depth maps for images in the input folder."""
import os
import glob
import torch
import cv2
import argparse
from torchvision.transforms import Compose
from dpt.models import DPTDepthModel
from dpt.midas_net import MidasNet_large
from dpt.transforms import Resize, NormalizeImage, PrepareForNet
import util.io


def normalize_depth(depth_map, scale=255.0):
    """Normalize depth map to a 0-255 range."""
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    depth_normalized = (depth_map - depth_min) / (depth_max - depth_min + 1e-6) * scale
    return depth_normalized.astype('uint8')


def run(input_path, output_path, model_path, model_type="dpt_hybrid", optimize=True):
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
    """
    print("Initializing MonoDepthNN...")

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device selected: {device}")

    # Load model
    if model_type == "dpt_large":  # DPT-Large
        net_w, net_h = 384, 384
        model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "dpt_hybrid":  # DPT-Hybrid
        net_w, net_h = 384, 384
        model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "midas_v21":  # Midas v2.1
        net_w, net_h = 384, 384
        model = MidasNet_large(model_path, non_negative=True)
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        raise ValueError(f"Model type '{model_type}' is not supported.")

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    # Prepare model
    model.eval()
    model.to(device)
    if optimize and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        model = model.half()

    # Process images
    img_names = glob.glob(os.path.join(input_path, "*"))
    num_images = len(img_names)
    os.makedirs(output_path, exist_ok=True)

    print(f"Processing {num_images} images...")
    for ind, img_name in enumerate(img_names):
        if os.path.isdir(img_name):
            continue

        print(f"Processing {img_name} ({ind + 1}/{num_images})...")
        try:
            # Read input image
            img = util.io.read_image(img_name)
            img_input = transform({"image": img})["image"]

            # Generate depth map
            with torch.no_grad():
                sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
                if optimize and device.type == "cuda":
                    sample = sample.to(memory_format=torch.channels_last)
                    sample = sample.half()

                prediction = model.forward(sample)
                prediction = (
                    torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=img.shape[:2],
                        mode="bicubic",
                        align_corners=False,
                    )
                    .squeeze()
                    .cpu()
                    .numpy()
                )

            # Normalize and save depth map
            depth_map = normalize_depth(prediction)
            filename = os.path.join(output_path, os.path.splitext(os.path.basename(img_name))[0] + "depth.png")
            cv2.imwrite(filename, depth_map)
            print(f"Saved depth map: {filename}")

        except Exception as e:
            print(f"Error processing {img_name}: {e}")

    print("Processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
   
    parser.add_argument(
        "-i", "--input_path", default=r'C:\Users\golno\OneDrive\Desktop\Depth-Aware-U-shape-Transformer\dataset\UIEB\input', help="folder with input images"
    )

    parser.add_argument(
        "-o",
        "--output_path",
        default="output_monodepth/UIEB_Changed", help="folder for output images",
    )
    parser.add_argument("-m", "--model_weights", required=True, help="Path to model weights")
    parser.add_argument(
        "-t",
        "--model_type",
        default="dpt_hybrid",
        help="Model type [dpt_large|dpt_hybrid|midas_v21]",
    )
    parser.add_argument("--optimize", action="store_true", help="Optimize model for faster inference")
    args = parser.parse_args()

    # Run depth map generation
    run(args.input_path, args.output_path, args.model_weights, args.model_type, args.optimize)