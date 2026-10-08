import os
import shutil
import random

def split_dataset(source_dir, output_dir, split_ratio=0.8):
    random.seed(42)
    
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist!")
        return

    classes = os.listdir(source_dir)
    print(f"Found items in source directory: {classes}\n")
    
    for cls in classes:
        cls_path = os.path.join(source_dir, cls)
        
        # Ensure it's a directory (skips loose files if any exist)
        if not os.path.isdir(cls_path):
            continue
            
        images = os.listdir(cls_path)
        # Filter out non-image files just in case
        images = [img for img in images if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Class '{cls}': Found {len(images)} images.")
        
        if len(images) == 0:
            print(f"  -> Skipping {cls} because no images were found.")
            continue
            
        random.shuffle(images)
        
        split_index = int(len(images) * split_ratio)
        train_images = images[:split_index]
        val_images = images[split_index:]
        
        # Create output directories
        os.makedirs(os.path.join(output_dir, 'train', cls), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'val', cls), exist_ok=True)
        
        # Copy train images
        for img in train_images:
            shutil.copy(os.path.join(cls_path, img), os.path.join(output_dir, 'train', cls, img))
                
        # Copy val images
        for img in val_images:
            shutil.copy(os.path.join(cls_path, img), os.path.join(output_dir, 'val', cls, img))
                
        print(f"  -> Success: {len(train_images)} train, {len(val_images)} val images saved.\n")

    print("Dataset splitting complete!")

# --- UPDATE YOUR PATHS HERE ---
# Make sure SOURCE_FOLDER points to the folder that CONTAINS the class folders (Calculus, caries, etc.)
SOURCE_FOLDER = "D:/ML/Project/Dataset-2/Tooth dataset" 
OUTPUT_FOLDER = "D:/ML/Project/Final_dataset"

split_dataset(SOURCE_FOLDER, OUTPUT_FOLDER)