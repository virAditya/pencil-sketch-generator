"""
Pencil Sketch Generator
A production-ready implementation of photo-to-sketch conversion using linear filters.
"""

from .sketch_engine import PencilSketch, SketchStyle
from .filters import FilterKernels
from .utils import ImageProcessor

__version__ = "1.0.0"
__author__ = "Your Name"

__all__ = ['PencilSketch', 'SketchStyle', 'FilterKernels', 'ImageProcessor']
