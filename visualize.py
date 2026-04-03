"""
Visualization and analysis tools for the defect detection system.
Generates plots, statistics, and performance metrics.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


class DatasetAnalyzer:
    """Analyze dataset statistics and distribution."""
    
    @staticmethod
    def get_dataset_statistics():
        """Get statistics about the dataset."""
        
        if not os.path.exists(RAW_DATA_DIR):
            print(f"❌ Dataset directory not found: {RAW_DATA_DIR}")
            return None
        
        statistics = {}
        
        for class_dir in os.listdir(RAW_DATA_DIR):
            class_path = os.path.join(RAW_DATA_DIR, class_dir)
            
            if os.path.isdir(class_path):
                images = [f for f in os.listdir(class_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
                statistics[class_dir] = len(images)
        
        return statistics
    
    @staticmethod
    def plot_class_distribution():
        """Plot class distribution bar chart."""
        
        stats = DatasetAnalyzer.get_dataset_statistics()
        
        if not stats:
            print("⚠️ No data to plot")
            return
        
        plt.figure(figsize=(12, 6))
        classes = list(stats.keys())
        counts = list(stats.values())
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
        bars = plt.bar(classes, counts, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.title('Dataset Class Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Defect Class', fontsize=12)
        plt.ylabel('Number of Images', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(PROCESSED_DATA_DIR, 'class_distribution.png')
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {output_path}")
        
        plt.show()
    
    @staticmethod
    def plot_training_history(history):
        """Plot training history."""
        
        if history is None:
            print("⚠️ No training history provided")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        axes[0, 0].set_title('Model Accuracy', fontweight='bold', fontsize=12)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Train Loss', linewidth=2)
        axes[0, 1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 1].set_title('Model Loss', fontweight='bold', fontsize=12)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision
        axes[1, 0].plot(history.history['precision'], label='Train Precision', linewidth=2)
        axes[1, 0].plot(history.history['val_precision'], label='Val Precision', linewidth=2)
        axes[1, 0].set_title('Model Precision', fontweight='bold', fontsize=12)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Recall
        axes[1, 1].plot(history.history['recall'], label='Train Recall', linewidth=2)
        axes[1, 1].plot(history.history['val_recall'], label='Val Recall', linewidth=2)
        axes[1, 1].set_title('Model Recall', fontweight='bold', fontsize=12)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(PROCESSED_DATA_DIR, 'training_history.png')
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {output_path}")
        
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_names):
        """Plot confusion matrix."""
        
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
        
        plt.title('Confusion Matrix', fontweight='bold', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(PROCESSED_DATA_DIR, 'confusion_matrix.png')
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {output_path}")
        
        plt.show()
    
    @staticmethod
    def plot_sample_images(num_samples=12):
        """Plot sample images from each class."""
        
        stats = DatasetAnalyzer.get_dataset_statistics()
        
        if not stats:
            print("⚠️ No data to plot")
            return
        
        num_classes = len(stats)
        fig, axes = plt.subplots(num_classes, 3, figsize=(12, 4*num_classes))
        
        if num_classes == 1:
            axes = axes.reshape(1, -1)
        
        import cv2
        
        for row, (class_name, count) in enumerate(stats.items()):
            class_path = os.path.join(RAW_DATA_DIR, class_name)
            
            # Get first 3 images from this class
            images = [f for f in os.listdir(class_path)[:3]
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
            
            for col, img_file in enumerate(images[:3]):
                img_path = os.path.join(class_path, img_file)
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                axes[row, col].imshow(image)
                axes[row, col].set_title(f"{class_name}\n{img_file}", fontsize=10)
                axes[row, col].axis('off')
        
        plt.suptitle('Sample Images from Each Class', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(PROCESSED_DATA_DIR, 'sample_images.png')
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {output_path}")
        
        plt.show()


def print_dataset_report():
    """Print detailed dataset report."""
    
    print("\n" + "="*60)
    print("DATASET ANALYSIS REPORT")
    print("="*60 + "\n")
    
    stats = DatasetAnalyzer.get_dataset_statistics()
    
    if not stats:
        print("❌ No dataset found!")
        return
    
    total_images = sum(stats.values())
    
    print("📊 Dataset Statistics:")
    print("-" * 60)
    
    for class_name in sorted(stats.keys()):
        count = stats[class_name]
        percentage = (count / total_images * 100)
        bar_length = int(percentage / 2)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"{class_name:20} {count:4d} ({percentage:5.1f}%) {bar}")
    
    print("-" * 60)
    print(f"{'TOTAL':20} {total_images:4d} (100.0%)\n")
    
    print("🎯 Recommendations:")
    if total_images < 1000:
        print("   ⚠️  Dataset is small. Consider collecting more images.")
    
    min_count = min(stats.values())
    max_count = max(stats.values())
    imbalance = max_count / min_count
    
    if imbalance > 2:
        print(f"   ⚠️  Dataset is imbalanced (ratio: {imbalance:.1f})")
        print("       Consider using class weights or sampling techniques.")
    
    if total_images >= 1000 and imbalance <= 2:
        print("   ✅ Dataset looks good for training!")
    
    print("\n" + "="*60 + "\n")


def main():
    """Main function."""
    
    print("\n" + "📊 "*20)
    print("Data Visualization & Analysis Tool")
    print("📊 "*20 + "\n")
    
    while True:
        print("\nChoose an option:")
        print("1. Show dataset statistics")
        print("2. Plot class distribution")
        print("3. View sample images")
        print("4. Print detailed report")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-4): ").strip()
        
        if choice == '1':
            stats = DatasetAnalyzer.get_dataset_statistics()
            if stats:
                for class_name, count in sorted(stats.items()):
                    print(f"  {class_name}: {count} images")
        
        elif choice == '2':
            DatasetAnalyzer.plot_class_distribution()
        
        elif choice == '3':
            DatasetAnalyzer.plot_sample_images()
        
        elif choice == '4':
            print_dataset_report()
        
        elif choice == '0':
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice")


if __name__ == '__main__':
    main()
