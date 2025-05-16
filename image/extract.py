#!/usr/bin/env python3

import os
import shutil
from pathlib import Path
from typing import Union, List, Tuple
import argparse
from PIL import Image
import numpy as np
from collections import defaultdict
from . import compare

def calculate_image_score(image_path: str) -> Tuple[float, str]:
    """
    Calculate a quality score for an image based on various metrics.
    Returns a tuple of (score, image_path)
    """
    try:
        with Image.open(image_path) as img:
            # Get image dimensions
            width, height = img.size
            
            # Calculate basic metrics
            resolution = width * height
            aspect_ratio = width / height
            
            # Normalize aspect ratio score (penalize extreme ratios)
            aspect_score = min(aspect_ratio, 1/aspect_ratio)
            
            # Get banding scores using compare module
            results = compare.image_results(image_path)
            banding_score = float(np.mean(list(results.values()))) if results else 0.0
            
            # Combine metrics into final score
            # We prioritize:
            # 1. Higher resolution
            # 2. Reasonable aspect ratios
            # 3. Better edge detail/less banding
            score = resolution * aspect_score * (1 + banding_score)
            
            return (score, image_path)
    except Exception as e:
        # Return minimum score for failed images
        return (float('-inf'), image_path)

def extract_top_images(
    input_dir: str,
    output_dir: str,
    mode: str = 'percentile',
    amount: float = 0.15
) -> List[str]:
    """
    Extract top images from input directory to output directory.
    
    Args:
        input_dir: Source directory containing images
        output_dir: Directory to copy selected images to
        mode: 'percentile' or 'fixed'
        amount: If percentile, float between 0-1 (e.g. 0.15 for top 15%)
               If fixed, integer number of images to select
    
    Returns:
        List of paths of copied images
    """
    print("EXTRACTING TOP IMAGES ...")
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    image_files = [
        str(f) for f in Path(input_dir).rglob('*')
        if f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        raise ValueError(f"No image files found in {input_dir}")
    
    print(f" * {len(image_files)} images found!")
    
    # Calculate scores for all images
    scores = [calculate_image_score(f) for f in image_files]
    scores = sorted(scores, reverse=True)  # Sort by score descending

    print(" * scoring complete!")
    
    # Determine how many images to keep
    if mode == 'percentile':
        num_to_keep = max(1, int(len(scores) * amount))
    else:  # fixed mode
        num_to_keep = min(int(amount), len(scores))
    
    # Select top images
    selected_images = scores[:num_to_keep]

    print(" * extracting the top images ...")
    
    # Copy selected images to output directory
    copied_files = []
    
    for _, img_path in selected_images:
        try:
            # Get original filename
            filename = os.path.basename(img_path)
            
            # Ensure unique filename in output dir
            base, ext = os.path.splitext(filename)
            counter = 1
            new_path = os.path.join(output_dir, filename)
            
            while os.path.exists(new_path):
                new_path = os.path.join(output_dir, f"{base}_{counter}{ext}")
                counter += 1
            
            # Copy the file
            shutil.copy2(img_path, new_path)
            copied_files.append(new_path)
            
        except Exception as e:
            print(f"Error copying {img_path}: {str(e)}")
            continue

    print("DONE!")
    
    return copied_files

def main():
    parser = argparse.ArgumentParser(description='Extract top quality images from a directory')
    parser.add_argument('--input', help='Source directory containing images')
    parser.add_argument('--output', help='Directory to extract images to')
    parser.add_argument(
        '--mode',
        choices=['percentile', 'fixed'],
        default='percentile',
        help='Selection mode: percentile or fixed amount'
    )
    parser.add_argument(
        '--amount',
        type=float,
        default=0.15,
        help='Amount to extract (0.15 for top 15%% in percentile mode, or 64 for top 64 images in fixed mode)'
    )
    
    args = parser.parse_args()
    
    try:
        copied_files = extract_top_images(
            args.input,
            args.output,
            args.mode,
            args.amount
        )
        print(f"Successfully extracted {len(copied_files)} images to {args.output}")
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
