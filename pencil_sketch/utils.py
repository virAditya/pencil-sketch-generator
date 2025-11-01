"""
Utility functions for image processing operations.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional
from PIL import Image


class ImageProcessor:
    """Handles image I/O and preprocessing operations."""
    
    @staticmethod
    def load_image(image_path: Union[str, Path]) -> np.ndarray:
        """
        Load image from file path.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Image as numpy array in BGR format
        
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image cannot be loaded
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        return image
    
    @staticmethod
    def save_image(image: np.ndarray, output_path: Union[str, Path]) -> None:
        """
        Save image to file.
        
        Args:
            image: Image array to save
            output_path: Destination file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(str(output_path), image)
        print(f"✓ Saved: {output_path}")
    
    @staticmethod
    def resize_image(image: np.ndarray, 
                    max_width: int = 1920, 
                    max_height: int = 1080) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio.
        
        Args:
            image: Input image
            max_width: Maximum width
            max_height: Maximum height
        
        Returns:
            Resized image
        """
        height, width = image.shape[:2]
        
        if width <= max_width and height <= max_height:
            return image
        
        # Calculate scaling factor
        scale = min(max_width / width, max_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        return cv2.resize(image, (new_width, new_height), 
                         interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def create_comparison(original: np.ndarray, 
                         sketch: np.ndarray, 
                         labels: Tuple[str, str] = ("Original", "Sketch")) -> np.ndarray:
        """
        Create side-by-side comparison of original and sketch.
        
        Args:
            original: Original image
            sketch: Sketch image
            labels: Tuple of labels for (original, sketch)
        
        Returns:
            Combined comparison image
        """
        # Ensure same height
        h1, w1 = original.shape[:2]
        h2, w2 = sketch.shape[:2]
        
        if h1 != h2:
            sketch = cv2.resize(sketch, (w2, h1))
        
        # Convert sketch to BGR if grayscale
        if len(sketch.shape) == 2:
            sketch = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        color = (255, 255, 255)
        
        original_labeled = original.copy()
        sketch_labeled = sketch.copy()
        
        cv2.putText(original_labeled, labels[0], (20, 40), 
                   font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(sketch_labeled, labels[1], (20, 40), 
                   font, font_scale, color, thickness, cv2.LINE_AA)
        
        # Combine horizontally
        comparison = np.hstack([original_labeled, sketch_labeled])
        
        return comparison
    
    @staticmethod
    def create_style_grid(original: np.ndarray, 
                         sketches: list, 
                         style_names: list) -> np.ndarray:
        """
        Create grid showing original + multiple sketch styles.
        
        Args:
            original: Original image
            sketches: List of sketch images
            style_names: List of style names
        
        Returns:
            Grid image
        """
        n_styles = len(sketches)
        
        # Resize all to same size
        target_h, target_w = 400, 400
        original_resized = cv2.resize(original, (target_w, target_h))
        sketches_resized = [cv2.resize(s, (target_w, target_h)) for s in sketches]
        
        # Convert grayscale to BGR
        sketches_bgr = []
        for sketch in sketches_resized:
            if len(sketch.shape) == 2:
                sketches_bgr.append(cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR))
            else:
                sketches_bgr.append(sketch)
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        labeled_original = original_resized.copy()
        cv2.putText(labeled_original, "Original", (20, 40), 
                   font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        labeled_sketches = []
        for sketch, name in zip(sketches_bgr, style_names):
            labeled = sketch.copy()
            cv2.putText(labeled, name, (20, 40), 
                       font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            labeled_sketches.append(labeled)
        
        # Create grid (2x2 or 2x3)
        if n_styles <= 3:
            row1 = np.hstack([labeled_original, labeled_sketches[0]])
            row2 = np.hstack(labeled_sketches[1:3]) if n_styles == 3 else labeled_sketches[1]
            if n_styles == 2:
                # Add black padding for symmetry
                padding = np.zeros_like(labeled_sketches[0])
                row2 = np.hstack([row2, padding])
            grid = np.vstack([row1, row2])
        else:
            # 2x3 grid
            row1 = np.hstack([labeled_original] + labeled_sketches[:2])
            row2 = np.hstack(labeled_sketches[2:5])
            grid = np.vstack([row1, row2])
        
        return grid
