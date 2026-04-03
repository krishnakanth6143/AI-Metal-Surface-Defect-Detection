"""
Download and setup NEU Surface Defect Dataset.
Downloads the NEU dataset from multiple sources.
"""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path
from tqdm import tqdm

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import RAW_DATA_DIR


class TqdmDownloadProgress:
    """Progress bar for downloads."""
    
    def __init__(self, filename):
        self.filename = filename
        self.pbar = None
        
    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc=self.filename)
        
        download_size = min(block_size, total_size - (block_num * block_size))
        self.pbar.update(download_size)
        
        if (block_num * block_size) >= total_size:
            self.pbar.close()


def download_neu_dataset():
    """
    Download NEU Surface Defect Dataset.
    
    The NEU Surface Defect Dataset is publicly available from:
    Source: http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html
    
    Note: This is a placeholder. You'll need to download manually or use direct links.
    """
    
    print("=" * 60)
    print("NEU Surface Defect Dataset Download")
    print("=" * 60)
    
    print("\n📝 Dataset Information:")
    print("   - Name: NEU Surface Defect Database")
    print("   - Classes: 6 (Normal, Scratch, Inclusion, Pitted Surface, Rolled-in Scale, Cratering)")
    print("   - Total Images: ~1,800 images")
    print("   - Image Size: 200 × 200 pixels")
    print("   - Format: BMP")
    
    print("\n🔗 Download Options:")
    print("\n1️⃣  Official Source (Recommended):")
    print("   Visit: http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html")
    print("   Download the BMP format dataset")
    
    print("\n2️⃣  Kaggle Dataset:")
    print("   Visit: https://www.kaggle.com/datasets/kaustubh1999/neu-surface-defect-database")
    print("   Download and extract to the data/raw folder")
    
    print("\n3️⃣  Alternative Kaggle:")
    print("   Visit: https://www.kaggle.com/datasets/dhanushnarayanan/surface-defects-stainless-steel")
    
    print("\n📂 Setup Instructions:")
    print(f"1. Download the dataset (zip or folder)")
    print(f"2. Extract to: {RAW_DATA_DIR}")
    print(f"3. Organize into subdirectories:")
    print(f"   {RAW_DATA_DIR}/")
    print(f"   ├── Normal/")
    print(f"   ├── Scratch/")
    print(f"   ├── Inclusion/")
    print(f"   ├── Pitted_Surface/")
    print(f"   ├── Rolled-in_Scale/")
    print(f"   └── Cratering/")
    
    print("\n⚡ Tips:")
    print("- Ensure images are properly organized by class")
    print("- Supported formats: BMP, PNG, JPG, JPEG, TIFF")
    print("- Minimum 50 images per class for training")
    print("- Recommended: 200+ images per class for better results")
    
    print("\n✅ After setting up the data, run: python train.py")
    print("=" * 60)


def extract_zip(zip_path, extract_to):
    """
    Extract zip file.
    
    Args:
        zip_path (str): Path to zip file
        extract_to (str): Extraction destination
    """
    print(f"\n📦 Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("✅ Extraction completed!")


def verify_data_structure():
    """Verify if data is properly organized."""
    
    if not os.path.exists(RAW_DATA_DIR):
        return False, "Data directory not found"
    
    class_dirs = [d for d in os.listdir(RAW_DATA_DIR) 
                 if os.path.isdir(os.path.join(RAW_DATA_DIR, d))]
    
    if not class_dirs:
        return False, "No class directories found"
    
    image_counts = {}
    for class_dir in class_dirs:
        class_path = os.path.join(RAW_DATA_DIR, class_dir)
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        image_counts[class_dir] = len(images)
    
    return True, image_counts


def main():
    """Main setup script."""
    
    # Create directory
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    # Check existing data
    has_data, result = verify_data_structure()
    
    if has_data and result:
        print("✅ Dataset already configured:")
        for class_name, count in result.items():
            print(f"   - {class_name}: {count} images")
        return
    
    # Show download instructions
    download_neu_dataset()


if __name__ == '__main__':
    main()
