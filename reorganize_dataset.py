"""Reorganize NEU dataset by class folders based on annotations."""

import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

# Configuration
ANNOTATIONS_DIR = "data/raw/NEU-DET/ANNOTATIONS"
IMAGES_DIR = "data/raw/NEU-DET/IMAGES"
RAW_DATA_DIR = "data/raw"

# Map annotation prefix to class folder names (matching config.py)
CLASS_MAPPING = {
    "patches": "Normal",
    "crazing": "Cratering",
    "inclusion": "Inclusion",
    "pitted_surface": "Pitted_Surface",
    "rolled-in_scale": "Rolled-in_Scale",
    "scratches": "Scratches"
}

def reorganize_dataset():
    """Reorganize images into class folders based on annotations."""
    
    # Create class folders
    class_folders = {}
    for class_name in CLASS_MAPPING.values():
        class_path = os.path.join(RAW_DATA_DIR, class_name)
        os.makedirs(class_path, exist_ok=True)
        class_folders[class_name] = class_path
        print(f"[OK] Created folder: {class_name}/")
    
    # Parse annotations and organize images
    annotation_files = sorted(os.listdir(ANNOTATIONS_DIR))
    processed = defaultdict(int)
    moved_images = defaultdict(int)
    
    print(f"\n[INFO] Processing {len(annotation_files)} annotation files...")
    
    for ann_file in annotation_files:
        if not ann_file.endswith('.xml'):
            continue
        
        # Get class from annotation filename
        class_prefix = ann_file.rsplit('_', 1)[0]  # e.g., "patches" from "patches_1.xml"
        
        if class_prefix not in CLASS_MAPPING:
            print(f"⚠️  Unknown class prefix: {class_prefix}")
            continue
        
        # Parse annotation to get image filename
        ann_path = os.path.join(ANNOTATIONS_DIR, ann_file)
        try:
            tree = ET.parse(ann_path)
            root = tree.getroot()
            image_filename = root.find('filename').text
            processed[class_prefix] += 1
        except Exception as e:
            print(f"❌ Error parsing {ann_file}: {e}")
            continue
        
        # Move image to class folder
        # Try with the filename as-is, then with common extensions if not found
        src_image_paths = [
            os.path.join(IMAGES_DIR, image_filename),
            os.path.join(IMAGES_DIR, image_filename + '.jpg'),
            os.path.join(IMAGES_DIR, image_filename + '.png'),
            os.path.join(IMAGES_DIR, image_filename + '.bmp'),
        ]
        
        class_name = CLASS_MAPPING[class_prefix]
        src_image_path = None
        actual_filename = None
        
        for potential_path in src_image_paths:
            if os.path.exists(potential_path):
                src_image_path = potential_path
                actual_filename = os.path.basename(potential_path)
                break
        
        if src_image_path:
            try:
                dst_image_path = os.path.join(class_folders[class_name], actual_filename)
                shutil.move(src_image_path, dst_image_path)
                moved_images[class_name] += 1
            except Exception as e:
                print(f"❌ Error moving {image_filename}: {e}")
        else:
            pass  # Silently skip missing images
    
    # Print summary
    print("\n" + "=" * 60)
    print("[SUCCESS] Dataset Reorganization Complete!")
    print("=" * 60)
    print("\n[SUMMARY] Images organized by class:")
    
    total_moved = 0
    for class_name in sorted(CLASS_MAPPING.values()):
        count = moved_images[class_name]
        total_moved += count
        print(f"   {class_name:20s}: {count:4d} images")
    
    print(f"\n   {'TOTAL':20s}: {total_moved:4d} images")
    print(f"\n[NOTE] All images organized in: {RAW_DATA_DIR}/")
    print("=" * 60)

if __name__ == '__main__':
    reorganize_dataset()
