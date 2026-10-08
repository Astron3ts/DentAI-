import os
import hashlib
import pandas as pd
import numpy as np
from PIL import Image

def compute_file_hash(filepath: str, chunk_size: int = 65536) -> str:
    """Computes SHA-256 hash for byte-level duplicate detection."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()

def compute_perceptual_hash(image: Image.Image, hash_size: int = 8) -> str:
    """Computes difference hash (dHash) for visual near-duplicate detection."""
    img_gray = image.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.array(img_gray, dtype=np.uint8).flatten()
    diff = []
    for row in range(hash_size):
        row_start = row * (hash_size + 1)
        for col in range(hash_size):
            diff.append(pixels[row_start + col] > pixels[row_start + col + 1])
    return hex(int(''.join(['1' if v else '0' for v in diff]), 2))[2:].zfill(hash_size * hash_size // 4)

def scan_dataset(data_dir: str):
    """
    Scans dataset directories (train and val), validates files, checks corruptions,
    and returns a summary dictionary and detailed record list.
    """
    records = []
    corrupted_files = []

    for split in ['train', 'val']:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            continue

        for class_name in sorted(os.listdir(split_dir)):
            class_path = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            for fname in os.listdir(class_path):
                if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    continue

                full_path = os.path.join(class_path, fname)
                try:
                    with Image.open(full_path) as img:
                        img.verify()
                    # Re-open for metadata reading after verify()
                    with Image.open(full_path) as img:
                        width, height = img.size
                        channels = len(img.getbands())
                        dhash = compute_perceptual_hash(img)

                    sha256 = compute_file_hash(full_path)
                    records.append({
                        'filepath': full_path,
                        'filename': fname,
                        'split': split,
                        'class': class_name,
                        'width': width,
                        'height': height,
                        'channels': channels,
                        'sha256': sha256,
                        'dhash': dhash,
                        'is_valid': True
                    })
                except Exception as e:
                    corrupted_files.append((full_path, str(e)))
                    records.append({
                        'filepath': full_path,
                        'filename': fname,
                        'split': split,
                        'class': class_name,
                        'width': 0,
                        'height': 0,
                        'channels': 0,
                        'sha256': None,
                        'dhash': None,
                        'is_valid': False
                    })

    df = pd.DataFrame(records)
    return df, corrupted_files

def find_duplicates(df: pd.DataFrame):
    """Identifies exact SHA-256 and perceptual dHash duplicate clusters."""
    valid_df = df[df['is_valid'] == True]
    exact_dupes = valid_df[valid_df.duplicated(subset=['sha256'], keep=False)]
    perceptual_dupes = valid_df[valid_df.duplicated(subset=['dhash'], keep=False)]
    return exact_dupes, perceptual_dupes

def build_manifest(data_dir: str, output_csv: str = "dataset_audit_manifest.csv"):
    """Builds and saves the full audit manifest CSV and prints summary statistics."""
    print(f"Scanning dataset at: {data_dir}")
    df, corrupted = scan_dataset(data_dir)

    print(f"Total files examined: {len(df)}")
    print(f"Corrupted / unreadable files: {len(corrupted)}")

    exact_dupes, perceptual_dupes = find_duplicates(df)
    print(f"Exact byte duplicates found: {len(exact_dupes)}")
    print(f"Perceptual near-duplicates found: {len(perceptual_dupes)}")

    # Class distribution summary
    if not df.empty:
        summary = df[df['is_valid']].groupby(['split', 'class']).size().unstack(fill_value=0)
        print("\n--- Class Distribution by Split ---")
        print(summary)

    df.to_csv(output_csv, index=False)
    print(f"\nAudit manifest successfully saved to: {output_csv}")
    return df

if __name__ == '__main__':
    DATASET_PATH = r"D:\ML\Project\Final_dataset"
    build_manifest(DATASET_PATH)
