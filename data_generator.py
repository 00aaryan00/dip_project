"""
Script: data_generator.py
Description: Generates an academic e-commerce product image dataset sourced from the
             Amazon Berkeley Objects (ABO) dataset. Slices strictly into Train and Test
             product partitions first to prevent data leakage, then synthesizes traceable defects.

Dataset Source:
Amazon Berkeley Objects (ABO) Dataset (Official benchmark for physical product catalog inspection).
"""

import os
import io
import glob
import random
import numpy as np
import pandas as pd
import cv2
from PIL import Image

from src.noise import add_gaussian_noise, add_salt_and_pepper_noise

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ABO_PARQUET_PATHS = glob.glob(
    r"C:\Users\aryan\.cache\huggingface\hub\datasets--huggingface--amazon-berkeley-objects\**\*.parquet",
    recursive=True
)
OUTPUT_DIR = "dataset"


def apply_motion_blur(img: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """Simulates linear camera shake / motion blur."""
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size, dtype=np.float32)
    kernel /= kernel_size
    return cv2.filter2D(img, -1, kernel)


def apply_low_brightness(img: np.ndarray, factor: float = 0.28) -> np.ndarray:
    """Simulates underexposed lighting."""
    dark = img.astype(np.float32) * factor
    return np.clip(dark, 0, 255).astype(np.uint8)


def apply_high_brightness(img: np.ndarray, shift: float = 120.0) -> np.ndarray:
    """Simulates overexposed / washed out lighting."""
    bright = img.astype(np.float32) + shift
    return np.clip(bright, 0, 255).astype(np.uint8)


def apply_low_contrast(img: np.ndarray) -> np.ndarray:
    """Compresses histogram into narrow mid-gray band."""
    norm = img.astype(np.float32) / 255.0
    compressed = norm * 40.0 + 105.0
    return np.clip(compressed, 0, 255).astype(np.uint8)


def apply_gaussian_blur(img: np.ndarray, ksize: int = 21, sigma: float = 8.0) -> np.ndarray:
    """Simulates out-of-focus camera optics."""
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma)


def create_defect_variant(img: np.ndarray, defect_type: str) -> np.ndarray:
    """Synthesizes a specific image quality defect on an Amazon product photo."""
    if defect_type == "gaussian_blur":
        return apply_gaussian_blur(img, ksize=21, sigma=8.0)
    elif defect_type == "motion_blur":
        return apply_motion_blur(img, kernel_size=17)
    elif defect_type == "dark":
        return apply_low_brightness(img, factor=0.25)
    elif defect_type == "bright":
        return apply_high_brightness(img, shift=120.0)
    elif defect_type == "low_contrast":
        return apply_low_contrast(img)
    elif defect_type == "gaussian_noise":
        return add_gaussian_noise(img, mean=0.0, sigma=30.0)
    elif defect_type == "salt_pepper":
        return add_salt_and_pepper_noise(img, amount=0.06)
    else:
        raise ValueError(f"Unknown defect type: {defect_type}")


def generate_dataset(num_base_images: int = 100):
    if not ABO_PARQUET_PATHS:
        raise FileNotFoundError("Amazon Berkeley Objects (ABO) dataset cache not found!")

    parquet_file = ABO_PARQUET_PATHS[0]
    print(f"Loading Amazon Berkeley Objects (ABO) dataset from:\n  {parquet_file}...")
    df_raw = pd.read_parquet(parquet_file)

    # Filter for pure product photos on studio white background (no human models)
    print("Filtering for clean studio Amazon product items (white background, no models)...")
    clean_indices = []
    for idx in range(len(df_raw)):
        row = df_raw.iloc[idx]
        if row["width"] < 220 or row["height"] < 220:
            continue
        try:
            im = Image.open(io.BytesIO(row["image"]["bytes"]))
            if im.mode != "RGB":
                continue
            arr = np.array(im)
            # Sample border margin (top, bottom, left, right strips)
            border = np.concatenate([
                arr[0:5, :, :].ravel(),
                arr[-5:, :, :].ravel(),
                arr[:, 0:5, :].ravel(),
                arr[:, -5:, :].ravel()
            ])
            if np.mean(border) > 215:  # Studio white background
                clean_indices.append(idx)
                if len(clean_indices) >= num_base_images:
                    break
        except Exception:
            continue

    print(f"Selected {len(clean_indices)} Amazon product items.")
    df_sample = df_raw.iloc[clean_indices].reset_index(drop=True)
    total_items = len(df_sample)

    # Clear existing dataset folders
    for split in ["train", "test"]:
        for label in ["good", "defective"]:
            d = os.path.join(OUTPUT_DIR, split, label)
            os.makedirs(d, exist_ok=True)
            for f in glob.glob(os.path.join(d, "*")):
                os.remove(f)

    os.makedirs(os.path.join(OUTPUT_DIR, "raw"), exist_ok=True)
    for f in glob.glob(os.path.join(OUTPUT_DIR, "raw", "*")):
        os.remove(f)

    # Strict 80 / 20 Train-Test product separation (zero data leakage)
    num_train = int(0.8 * total_items)
    train_indices = set(range(num_train))

    defect_catalog = [
        "gaussian_blur",
        "motion_blur",
        "dark",
        "bright",
        "low_contrast",
        "gaussian_noise",
        "salt_pepper"
    ]

    manifest_records = []
    print(f"Generating ABO dataset ({total_items} items: {num_train} train, {total_items - num_train} test)...")

    for idx, row in df_sample.iterrows():
        prod_id = str(row["image_id"]) if "image_id" in row else f"item_{idx}"
        img_dict = row["image"]
        img_bytes = img_dict["bytes"]

        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_rgb = np.array(pil_img)

        # Save to raw
        raw_path = os.path.join(OUTPUT_DIR, "raw", f"amazon_{prod_id}.jpg")
        pil_img.save(raw_path, quality=95)

        split = "train" if idx in train_indices else "test"
        is_good = (idx % 2 == 0)

        if is_good:
            save_path = os.path.join(OUTPUT_DIR, split, "good", f"amazon_{prod_id}_good.jpg")
            pil_img.save(save_path, quality=95)
            manifest_records.append({
                "image_id": f"amazon_{prod_id}_good",
                "original_product_id": prod_id,
                "split": split,
                "dataset_source": "Amazon Berkeley Objects (ABO)",
                "category": "Amazon Physical Product",
                "defect_type": "none",
                "ground_truth": "ACCEPT",
                "resolution": f"{pil_img.size[0]}x{pil_img.size[1]}",
                "filepath": save_path
            })
        else:
            defect = defect_catalog[(idx // 2) % len(defect_catalog)]
            defective_rgb = create_defect_variant(img_rgb, defect)
            save_path = os.path.join(OUTPUT_DIR, split, "defective", f"amazon_{prod_id}_{defect}.jpg")
            Image.fromarray(defective_rgb).save(save_path, quality=95)
            manifest_records.append({
                "image_id": f"amazon_{prod_id}_{defect}",
                "original_product_id": prod_id,
                "split": split,
                "dataset_source": "Amazon Berkeley Objects (ABO)",
                "category": "Amazon Physical Product",
                "defect_type": defect,
                "ground_truth": "REJECT",
                "resolution": f"{pil_img.size[0]}x{pil_img.size[1]}",
                "filepath": save_path
            })

    manifest_df = pd.DataFrame(manifest_records)
    manifest_path = os.path.join(OUTPUT_DIR, "manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)

    print("\nAmazon Berkeley Objects dataset generation completed!")
    print(f"Total images: {len(manifest_df)}")
    print("Distribution by split and ground truth:")
    print(pd.crosstab(manifest_df["split"], manifest_df["ground_truth"]))
    print(f"Sample resolution: {manifest_df['resolution'].iloc[0]}")
    print(f"Manifest written to: {manifest_path}")
    return manifest_df


if __name__ == "__main__":
    generate_dataset(num_base_images=100)
