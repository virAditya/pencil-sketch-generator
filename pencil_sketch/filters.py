"""
Custom convolution kernels for image filtering.
Implements various linear filters for edge detection and enhancement.
"""

import numpy as np
from typing import Tuple


class FilterKernels:
    """Collection of convolution kernels for image processing."""
    
    @staticmethod
    def laplacian() -> np.ndarray:
        """
        Standard Laplacian kernel for edge detection.
        Detects second-order intensity changes.
        
        Returns:
            3x3 Laplacian kernel
        """
        return np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=np.float32)
    
    @staticmethod
    def sharpening(strength: int = 5) -> np.ndarray:
        """
        Generate sharpening kernel with adjustable strength.
        
        Args:
            strength: Center pixel weight (4-9 recommended)
                     4: Subtle enhancement
                     5-7: Strong pencil-like strokes
                     8+: Over-sharpened appearance
        
        Returns:
            3x3 sharpening kernel
        """
        if strength < 4:
            raise ValueError("Strength should be at least 4 for visible effect")
        if strength > 10:
            print(f"Warning: Strength {strength} may cause over-sharpening")
        
        return np.array([
            [0, -1, 0],
            [-1, strength, -1],
            [0, -1, 0]
        ], dtype=np.float32)
    
    @staticmethod
    def edge_enhance() -> np.ndarray:
        """
        Edge enhancement kernel using Laplacian-based approach.
        
        Returns:
            3x3 edge enhancement kernel
        """
        return np.array([
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1]
        ], dtype=np.float32)
    
    @staticmethod
    def sobel_x() -> np.ndarray:
        """Sobel horizontal edge detection kernel."""
        return np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=np.float32)
    
    @staticmethod
    def sobel_y() -> np.ndarray:
        """Sobel vertical edge detection kernel."""
        return np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=np.float32)
    
    @staticmethod
    def gaussian_kernel(size: int = 5, sigma: float = 1.0) -> np.ndarray:
        """
        Generate 2D Gaussian kernel.
        
        Args:
            size: Kernel size (should be odd)
            sigma: Standard deviation
        
        Returns:
            Normalized Gaussian kernel
        """
        if size % 2 == 0:
            size += 1  # Ensure odd size
        
        ax = np.arange(-size // 2 + 1., size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        
        kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        
        return kernel / np.sum(kernel)
