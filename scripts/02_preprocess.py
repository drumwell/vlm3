#!/usr/bin/env python3
"""
02_preprocess.py - Clean and standardize images from inventory.
Converts to PNG, grayscale, deskew, denoise, adjust contrast.
Detects and splits double-page scans.
"""

import argparse
import csv
import re
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image


def deskew_image(img):
    """Deskew image using Hough line detection."""
    # Work with binary threshold for better line detection
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    
    if lines is None or len(lines) == 0:
        return img
    
    # Calculate average angle
    angles = []
    for rho, theta in lines[:, 0]:
        angle = np.degrees(theta) - 90
        if -45 < angle < 45:
            angles.append(angle)
    
    if not angles:
        return img
    
    median_angle = np.median(angles)
    
    # Only deskew if angle is significant (> 0.5 degrees)
    if abs(median_angle) < 0.5:
        return img
    
    # Rotate image
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), 
                             flags=cv2.INTER_CUBIC, 
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated


def denoise_image(img):
    """Light denoising using fastNlMeansDenoising."""
    if len(img.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    else:
        return cv2.fastNlMeansDenoising(img, None, 10, 7, 21)


def adjust_contrast(img):
    """Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    if len(img.shape) == 3:
        # Convert to LAB, apply CLAHE to L channel
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)


def is_double_page(img, aspect_threshold=1.6):
    """Detect if image is a double-page scan (wide aspect ratio)."""
    h, w = img.shape[:2]
    aspect = w / h
    return aspect > aspect_threshold


def split_double_page(img):
    """Split double-page scan into left and right halves."""
    h, w = img.shape[:2]
    mid = w // 2
    left = img[:, :mid]
    right = img[:, mid:]
    return left, right


def preprocess_image(input_path, output_path, grayscale=True):
    """
    Process a single image: convert to PNG, grayscale, deskew, denoise, contrast.
    Returns: (success: bool, width: int, height: int, was_split: bool, error: str)
    """
    try:
        # Read image
        img = cv2.imread(str(input_path))
        if img is None:
            return False, 0, 0, False, "Failed to read image"
        
        original_h, original_w = img.shape[:2]
        
        # Check for double-page and split if needed
        was_split = False
        if is_double_page(img):
            # For double-page, save both halves
            left, right = split_double_page(img)
            was_split = True
            
            # Process left page
            left_processed = process_single_page(left, grayscale)
            left_path = output_path.parent / (output_path.stem + '_L.png')
            cv2.imwrite(str(left_path), left_processed)
            
            # Process right page
            right_processed = process_single_page(right, grayscale)
            right_path = output_path.parent / (output_path.stem + '_R.png')
            cv2.imwrite(str(right_path), right_processed)
            
            # Also save the full combined version
            img = process_single_page(img, grayscale)
        else:
            # Process as single page
            img = process_single_page(img, grayscale)
        
        # Save as PNG
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), img)
        
        h, w = img.shape[:2]
        return True, w, h, was_split, ""
        
    except Exception as e:
        return False, 0, 0, False, str(e)


def process_single_page(img, grayscale=True):
    """Apply preprocessing pipeline to a single page."""
    # Convert to grayscale if requested
    if grayscale and len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Deskew
    img = deskew_image(img)
    
    # Denoise (light)
    img = denoise_image(img)
    
    # Adjust contrast
    img = adjust_contrast(img)
    
    return img


def main():
    parser = argparse.ArgumentParser(description="Preprocess images from inventory")
    parser.add_argument('--inventory', default='work/inventory.csv',
                        help='Input inventory CSV')
    parser.add_argument('--section-filter', default='.*',
                        help='Regex to filter sections (e.g., "^11 - ")')
    parser.add_argument('--output-dir', default='work/images_clean',
                        help='Output directory for cleaned images')
    parser.add_argument('--error-log', default='work/logs/preprocess_errors.csv',
                        help='Error log CSV path')
    parser.add_argument('--grayscale', action='store_true', default=True,
                        help='Convert to grayscale (default: True)')
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    inventory_path = project_root / args.inventory
    output_dir = project_root / args.output_dir
    error_log_path = project_root / args.error_log
    data_src = project_root / 'data_src'
    
    # Compile section filter
    section_pattern = re.compile(args.section_filter)
    
    print(f"[02_preprocess] Reading inventory: {inventory_path}")
    print(f"[02_preprocess] Section filter: {args.section_filter}")
    print(f"[02_preprocess] Output directory: {output_dir}")
    
    # Read inventory
    with open(inventory_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if section_pattern.match(r['section_dir'])]
    
    print(f"[02_preprocess] Processing {len(rows)} images...")
    
    # Track results
    successful = 0
    failed = 0
    split_count = 0
    errors = []
    resolutions = defaultdict(int)
    sample_paths = []
    
    # Process each image
    for i, row in enumerate(rows, 1):
        rel_path = row['rel_path']
        input_path = data_src / rel_path
        
        # Create output path (mirror structure, change to .png)
        output_rel = Path(rel_path).with_suffix('.png')
        output_path = output_dir / output_rel
        
        print(f"  [{i}/{len(rows)}] {rel_path}...", end=' ', flush=True)
        
        success, w, h, was_split, error = preprocess_image(input_path, output_path, args.grayscale)
        
        if success:
            successful += 1
            if was_split:
                split_count += 1
            
            # Bucket resolution (round to nearest 100)
            res_bucket = f"{w//100*100}x{h//100*100}"
            resolutions[res_bucket] += 1
            
            # Save sample paths (first 5)
            if len(sample_paths) < 5:
                sample_paths.append((str(output_rel), w, h, was_split))
            
            print(f"✓ {w}x{h}" + (" [SPLIT]" if was_split else ""))
        else:
            failed += 1
            errors.append({
                'rel_path': rel_path,
                'error': error
            })
            print(f"✗ {error}")
    
    # Write error log
    if errors:
        error_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(error_log_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['rel_path', 'error'])
            writer.writeheader()
            writer.writerows(errors)
        print(f"\n[02_preprocess] ⚠ Errors written to {error_log_path}")
    
    # Summary
    print(f"\n[02_preprocess] Summary:")
    print(f"  ✓ Successful: {successful}/{len(rows)}")
    print(f"  ✗ Failed: {failed}/{len(rows)}")
    print(f"  ✂ Double-page splits: {split_count}")
    
    # Resolution histogram
    print(f"\n[02_preprocess] Resolution histogram:")
    for res, count in sorted(resolutions.items()):
        print(f"  {res}: {count}")
    
    # Sample paths
    print(f"\n[02_preprocess] Sample output paths (first 5):")
    for path, w, h, split in sample_paths:
        split_mark = " [SPLIT]" if split else ""
        print(f"  {path} | {w}x{h}{split_mark}")
    
    # Validation
    print(f"\n[02_preprocess] Validation:")
    
    # Check 1:1 mapping (excluding splits)
    output_files = list(output_dir.rglob('*.png'))
    print(f"  Output files: {len(output_files)} (including split pages)")
    print(f"  Expected minimum: {successful} (1:1 mapping)")
    
    # Check for zero-byte files
    zero_byte = [f for f in output_files if f.stat().st_size == 0]
    if zero_byte:
        print(f"  ✗ Zero-byte files found: {len(zero_byte)}")
        for f in zero_byte[:5]:
            print(f"    - {f.relative_to(output_dir)}")
    else:
        print(f"  ✓ No zero-byte files")
    
    print(f"\n[02_preprocess] Done!")


if __name__ == '__main__':
    main()
