"""
Image preprocessing module for metal surface images.
Handles grayscale conversion, noise reduction, edge detection, and resizing.
"""

import cv2
import numpy as np
from pathlib import Path
import os


class ImageProcessor:
    """Handles image preprocessing operations."""
    
    def __init__(self, target_size=(224, 224)):
        """
        Initialize the ImageProcessor.
        
        Args:
            target_size (tuple): Target image size (height, width)
        """
        self.target_size = target_size
    
    def load_image(self, image_path):
        """
        Load an image from file.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Loaded image
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def convert_to_grayscale(self, image):
        """
        Convert image to grayscale.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Grayscale image
        """
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image
    
    def reduce_noise(self, image, method='gaussian', kernel_size=5):
        """
        Reduce image noise using Gaussian blur or bilateral filter.
        
        Args:
            image (numpy.ndarray): Input image
            method (str): 'gaussian' or 'bilateral'
            kernel_size (int): Size of the kernel
            
        Returns:
            numpy.ndarray: Denoised image
        """
        if method == 'gaussian':
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        elif method == 'bilateral':
            return cv2.bilateralFilter(image, 9, 75, 75)
        else:
            return image
    
    def edge_detection(self, image, method='canny', low_threshold=50, high_threshold=150):
        """
        Detect edges in image using Canny edge detection.
        
        Args:
            image (numpy.ndarray): Input grayscale image
            method (str): 'canny' or 'sobel'
            low_threshold (int): Lower threshold for Canny
            high_threshold (int): Upper threshold for Canny
            
        Returns:
            numpy.ndarray: Edge-detected image
        """
        if method == 'canny':
            return cv2.Canny(image, low_threshold, high_threshold)
        elif method == 'sobel':
            sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
            return np.sqrt(sobelx**2 + sobely**2).astype(np.uint8)
        return image
    
    def resize_image(self, image, target_size=None):
        """
        Resize image to target size.
        
        Args:
            image (numpy.ndarray): Input image
            target_size (tuple): Target size (height, width)
            
        Returns:
            numpy.ndarray: Resized image
        """
        if target_size is None:
            target_size = self.target_size
        
        return cv2.resize(image, (target_size[1], target_size[0]), 
                         interpolation=cv2.INTER_AREA)
    
    def normalize_image(self, image):
        """
        Normalize image to [0, 1] range.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Normalized image
        """
        return image.astype(np.float32) / 255.0
    
    def preprocess_image(self, image_path, normalize=True):
        """
        Complete preprocessing pipeline.
        
        Args:
            image_path (str): Path to the image file
            normalize (bool): Whether to normalize the image
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Load image
        image = self.load_image(image_path)
        
        # Resize
        image = self.resize_image(image)
        
        # Normalize
        if normalize:
            image = self.normalize_image(image)
        
        return image
    
    def preprocess_for_analysis(self, image_path):
        """
        Preprocessing pipeline with edge detection for analysis.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            tuple: (preprocessed_image, edge_image, original_resized)
        """
        # Load and resize original
        original = self.load_image(image_path)
        original_resized = self.resize_image(original)
        
        # Grayscale
        gray = self.convert_to_grayscale(original_resized)
        
        # Noise reduction
        denoised = self.reduce_noise(gray)
        
        # Edge detection
        edges = self.edge_detection(denoised)
        
        # Normalize original for model
        normalized = self.normalize_image(original_resized)
        
        return normalized, edges, original_resized
    
    def batch_preprocess(self, image_dir, output_dir=None):
        """
        Preprocess multiple images from a directory.
        
        Args:
            image_dir (str): Directory containing images
            output_dir (str): Directory to save processed images
            
        Returns:
            list: List of preprocessed image arrays
        """
        processed_images = []
        image_path = Path(image_dir)
        
        for img_file in image_path.glob('*.*'):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                try:
                    processed = self.preprocess_image(str(img_file))
                    processed_images.append(processed)
                    
                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)
                        output_path = os.path.join(output_dir, img_file.name)
                        # Save normalized image as uint8
                        cv2.imwrite(output_path, (processed * 255).astype(np.uint8))
                        
                except Exception as e:
                    print(f"Error processing {img_file}: {str(e)}")
        
        return processed_images
