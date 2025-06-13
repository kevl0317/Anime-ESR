import os
import cv2
from pathlib import Path
from tqdm import tqdm

def downsample_images(input_dir, output_dir, scale=4):
    """
    Downsample all images in the input directory by the given scale using bicubic interpolation,
    and save to the output directory.

    Args:
        input_dir (str or Path): Path to folder containing high-resolution images.
        output_dir (str or Path): Path to save low-resolution images.
        scale (int): Downscaling factor (e.g., 4 for 4x downsampling).
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    img_files = sorted([f for f in input_dir.iterdir() if f.suffix.lower() in image_extensions])

    print(f"Found {len(img_files)} images in {input_dir}")
    for img_path in tqdm(img_files):
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"⚠️ Skipping unreadable image: {img_path}")
            continue

        h, w = img.shape[:2]
        new_size = (w // scale, h // scale)
        lr_img = cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)

        output_path = output_dir / img_path.name
        cv2.imwrite(str(output_path), lr_img)

    print(f"\nDone! {len(img_files)} images processed.")


# Example usage
if __name__ == "__main__":
    downsample_images(
        input_dir="datasets/test_real",      # <-- change to your input folder
        output_dir="datasets/test_real",  # <-- change to your desired output folder
        scale=2
    )
