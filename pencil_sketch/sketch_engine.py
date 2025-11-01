"""
Core pencil sketch generation engine.
Implements dodge blending with linear filter preprocessing.
"""

import cv2
import numpy as np
from enum import Enum
from typing import Optional, Tuple
from .filters import FilterKernels


class SketchStyle(Enum):
    """Predefined sketch style presets."""
    DETAILED = ("Detailed", 5, 5)      # sigma=5, sharpen=5
    MEDIUM = ("Medium", 12, 5)         # sigma=12, sharpen=5
    LIGHT = ("Light", 20, 4)           # sigma=20, sharpen=4
    BOLD = ("Bold", 10, 7)             # sigma=10, sharpen=7
    MINIMALIST = ("Minimalist", 25, 4) # sigma=25, sharpen=4


class PencilSketch:
    """
    Photo-to-pencil sketch converter using linear filters and dodge blending.
    
    This implementation combines:
    - Gaussian blur (linear convolution filter)
    - Dodge blend (nonlinear division operation)
    - Sharpening (linear convolution filter)
    
    Theory:
    --------
    1. Convert to grayscale (preserve luminance information)
    2. Invert image (dark edges become bright spots)
    3. Apply Gaussian blur to inverted (spreads edge information)
    4. Dodge blend: Result = Gray / (1 - BlurredInverted)
       - Flat regions → white (paper)
       - Edges → dark lines (pencil strokes)
    5. Sharpen to enhance pencil line quality
    """
    
    def __init__(self, 
                 blur_sigma: float = 10.0,
                 sharpen_strength: int = 5,
                 apply_sharpening: bool = True):
        """
        Initialize pencil sketch generator.
        
        Args:
            blur_sigma: Gaussian blur standard deviation
                       3-5: Detailed sketch with fine lines
                       10-15: Medium sketch, major features only
                       20-30: Minimalist sketch, simplified forms
            sharpen_strength: Center value of sharpening kernel
                            4: Subtle enhancement
                            5-7: Strong pencil-like strokes
                            8+: Over-sharpened appearance
            apply_sharpening: Whether to apply sharpening step
        """
        self.blur_sigma = blur_sigma
        self.sharpen_strength = sharpen_strength
        self.apply_sharpening = apply_sharpening
        
        # Pre-compute sharpening kernel
        self.sharpen_kernel = FilterKernels.sharpening(sharpen_strength)
    
    @classmethod
    def from_preset(cls, style: SketchStyle) -> 'PencilSketch':
        """
        Create PencilSketch with predefined style preset.
        
        Args:
            style: SketchStyle enum value
        
        Returns:
            Configured PencilSketch instance
        """
        _, sigma, sharpen = style.value
        return cls(blur_sigma=sigma, sharpen_strength=sharpen)
    
    def dodge_blend(self, front: np.ndarray, back: np.ndarray) -> np.ndarray:
        """
        Apply dodge blending operation.
        
        Formula: Result = Back / (1 - Front/255)
        Equivalent to: Result = Back * 255 / (255 - Front)
        
        This is a nonlinear operation that creates selective brightening:
        - Where Front and Back are similar → Result ≈ 255 (white)
        - Where they differ → Result is darker (pencil lines)
        
        Args:
            front: Blend layer (blurred inverted image)
            back: Base layer (original grayscale)
        
        Returns:
            Dodge blended result
        """
        # Normalize to [0, 1]
        front_norm = front.astype(np.float32) / 255.0
        back_norm = back.astype(np.float32) / 255.0
        
        # Dodge blend formula with numerical stability
        result = back_norm / (1.0 - front_norm + 1e-7)
        
        # Clip to valid range and denormalize
        result = np.clip(result, 0, 1) * 255.0
        
        return result.astype(np.uint8)
    
    def apply(self, image: np.ndarray, return_steps: bool = False) -> np.ndarray:
        """
        Convert photo to pencil sketch.
        
        Args:
            image: Input image (BGR or RGB format)
            return_steps: If True, return dict with intermediate steps
        
        Returns:
            Sketch image (grayscale) or dict of intermediate steps
        """
        # Step 1: Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Step 2: Invert image
        inverted = 255 - gray
        
        # Step 3: Apply Gaussian blur to inverted
        # Kernel size should be odd and roughly 6*sigma + 1
        kernel_size = int(6 * self.blur_sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        blurred = cv2.GaussianBlur(inverted, 
                                   (kernel_size, kernel_size), 
                                   self.blur_sigma)
        
        # Step 4: Dodge blend
        sketch = self.dodge_blend(blurred, gray)
        
        # Step 5: Optional sharpening
        if self.apply_sharpening:
            sketch = cv2.filter2D(sketch, -1, self.sharpen_kernel)
        
        if return_steps:
            return {
                'grayscale': gray,
                'inverted': inverted,
                'blurred': blurred,
                'sketch_before_sharpen': self.dodge_blend(blurred, gray),
                'final_sketch': sketch
            }
        
        return sketch
    
    def apply_with_comparison(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate sketch and create side-by-side comparison.
        
        Args:
            image: Input image
        
        Returns:
            Tuple of (sketch, comparison_image)
        """
        from .utils import ImageProcessor
        
        sketch = self.apply(image)
        comparison = ImageProcessor.create_comparison(image, sketch)
        
        return sketch, comparison
    
    def generate_style_variations(self, image: np.ndarray) -> dict:
        """
        Generate sketches in multiple preset styles.
        
        Args:
            image: Input image
        
        Returns:
            Dictionary mapping style names to sketch images
        """
        styles = {}
        
        for style in SketchStyle:
            name, sigma, sharpen = style.value
            generator = PencilSketch(blur_sigma=sigma, sharpen_strength=sharpen)
            styles[name] = generator.apply(image)
        
        return styles
    
    def __repr__(self) -> str:
        return (f"PencilSketch(blur_sigma={self.blur_sigma}, "
                f"sharpen_strength={self.sharpen_strength}, "
                f"apply_sharpening={self.apply_sharpening})")
