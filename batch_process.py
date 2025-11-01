#!/usr/bin/env python3
"""
Batch process multiple images.

Usage:
    python batch_process.py input_folder/ --output results/
    python batch_process.py input_folder/ --style medium
"""

import argparse
from pathlib import Path
from tqdm import tqdm
from pencil_sketch import PencilSketch, SketchStyle, ImageProcessor


def main():
    parser = argparse.ArgumentParser(description='Batch process images to pencil sketches')
    parser.add_argument('input_dir', type=str, help='Input directory containing images')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--style', type=str, choices=['detailed', 'medium', 'light', 'bold', 'minimalist'],
                       default='medium', help='Sketch style preset')
    parser.add_argument('--formats', type=str, nargs='+', 
                       default=['jpg', 'jpeg', 'png', 'bmp'],
                       help='Image formats to process')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = []
    for fmt in args.formats:
        image_files.extend(input_dir.glob(f"*.{fmt}"))
        image_files.extend(input_dir.glob(f"*.{fmt.upper()}"))
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Create generator
    style_map = {
        'detailed': SketchStyle.DETAILED,
        'medium': SketchStyle.MEDIUM,
        'light': SketchStyle.LIGHT,
        'bold': SketchStyle.BOLD,
        'minimalist': SketchStyle.MINIMALIST
    }
    generator = PencilSketch.from_preset(style_map[args.style])
    
    # Process images
    for img_path in tqdm(image_files, desc="Processing"):
        try:
            image = ImageProcessor.load_image(img_path)
            sketch = generator.apply(image)
            
            output_path = output_dir / f"{img_path.stem}_sketch.png"
            ImageProcessor.save_image(sketch, output_path)
        except Exception as e:
            print(f"\nError processing {img_path.name}: {e}")
    
    print(f"\n✓ Batch processing complete!")
    print(f"  Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
