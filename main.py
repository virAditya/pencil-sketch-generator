#!/usr/bin/env python3
"""
Main CLI interface for Pencil Sketch Generator.

Usage:
    python main.py input.jpg -o output.jpg
    python main.py input.jpg --style detailed
    python main.py input.jpg --sigma 15 --sharpen 6
    python main.py input.jpg --compare
    python main.py input.jpg --all-styles
"""

import argparse
from pathlib import Path
from pencil_sketch import PencilSketch, SketchStyle, ImageProcessor


def main():
    parser = argparse.ArgumentParser(
        description='Convert photos to realistic pencil sketches',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion
  python main.py input.jpg
  
  # Use preset style
  python main.py input.jpg --style detailed
  
  # Custom parameters
  python main.py input.jpg --sigma 15 --sharpen 6
  
  # Generate comparison
  python main.py input.jpg --compare
  
  # Generate all style variations
  python main.py input.jpg --all-styles

Available styles:
  detailed    - Fine lines, captures texture (sigma=5)
  medium      - Balanced detail and simplicity (sigma=12)
  light       - Soft, minimalist lines (sigma=20)
  bold        - Strong, dark strokes (sigma=10, sharpen=7)
  minimalist  - Very few lines, simplified (sigma=25)
        """
    )
    
    parser.add_argument('input', type=str, help='Input image path')
    parser.add_argument('-o', '--output', type=str, help='Output path (default: input_sketch.png)')
    parser.add_argument('--style', type=str, choices=['detailed', 'medium', 'light', 'bold', 'minimalist'],
                       help='Use preset style')
    parser.add_argument('--sigma', type=float, help='Gaussian blur sigma (3-30)')
    parser.add_argument('--sharpen', type=int, help='Sharpening strength (4-9)')
    parser.add_argument('--no-sharpen', action='store_true', help='Disable sharpening')
    parser.add_argument('--compare', action='store_true', help='Generate side-by-side comparison')
    parser.add_argument('--all-styles', action='store_true', help='Generate all style variations')
    parser.add_argument('--resize', action='store_true', help='Resize large images (max 1920x1080)')
    
    args = parser.parse_args()
    
    # Load image
    print(f"Loading image: {args.input}")
    try:
        image = ImageProcessor.load_image(args.input)
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Resize if requested
    if args.resize:
        image = ImageProcessor.resize_image(image)
    
    # Determine output path
    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_sketch{input_path.suffix}"
    
    # Generate all styles
    if args.all_styles:
        print("Generating all style variations...")
        generator = PencilSketch()
        styles = generator.generate_style_variations(image)
        
        # Save individual styles
        for style_name, sketch in styles.items():
            style_output = input_path.parent / f"{input_path.stem}_{style_name.lower()}.png"
            ImageProcessor.save_image(sketch, style_output)
        
        # Create grid
        sketches = list(styles.values())
        names = list(styles.keys())
        grid = ImageProcessor.create_style_grid(image, sketches, names)
        grid_output = input_path.parent / f"{input_path.stem}_styles_grid.png"
        ImageProcessor.save_image(grid, grid_output)
        
        print(f"\n✓ Generated {len(styles)} style variations")
        print(f"✓ Saved style grid: {grid_output}")
        return
    
    # Create generator
    if args.style:
        style_map = {
            'detailed': SketchStyle.DETAILED,
            'medium': SketchStyle.MEDIUM,
            'light': SketchStyle.LIGHT,
            'bold': SketchStyle.BOLD,
            'minimalist': SketchStyle.MINIMALIST
        }
        generator = PencilSketch.from_preset(style_map[args.style])
        print(f"Using preset: {args.style}")
    else:
        sigma = args.sigma if args.sigma else 10.0
        sharpen = args.sharpen if args.sharpen else 5
        apply_sharpen = not args.no_sharpen
        generator = PencilSketch(blur_sigma=sigma, 
                                sharpen_strength=sharpen,
                                apply_sharpening=apply_sharpen)
        print(f"Using custom parameters: sigma={sigma}, sharpen={sharpen}")
    
    print(f"Processing with {generator}")
    
    # Generate sketch
    sketch = generator.apply(image)
    
    # Save results
    if args.compare:
        comparison = ImageProcessor.create_comparison(image, sketch)
        compare_output = input_path.parent / f"{input_path.stem}_comparison.png"
        ImageProcessor.save_image(comparison, compare_output)
    
    ImageProcessor.save_image(sketch, output_path)
    
    print(f"\n✓ Successfully generated pencil sketch!")
    print(f"  Input:  {args.input}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
