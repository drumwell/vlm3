#!/usr/bin/env python3
"""
01_inventory.py - Scan data_src/ and generate work/inventory.csv
Filters sections by pattern and computes file metadata.
"""

import argparse
import csv
import hashlib
import re
from pathlib import Path
from collections import defaultdict

import yaml


def load_config(config_path):
    """Load config.yaml."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def sha1_file(filepath):
    """Compute SHA1 hash of file."""
    h = hashlib.sha1()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def extract_page_no(filename):
    """Extract page number from filename like '11-100.jpg' -> '100'."""
    stem = Path(filename).stem
    parts = stem.split('-', 1)
    if len(parts) == 2:
        return parts[1]
    return stem


def guess_type(ocr_text_dir, rel_path):
    """
    Placeholder for guessing document type.
    Returns empty string for now (will be filled in step 5).
    """
    return ""


def main():
    parser = argparse.ArgumentParser(description="Generate inventory.csv from data_src/")
    parser.add_argument('--section-filter', default='.*', 
                        help='Regex to filter section folders (e.g., "^11 - ")')
    parser.add_argument('--config', default='config.yaml',
                        help='Path to config.yaml')
    parser.add_argument('--data-src', default='data_src',
                        help='Source directory')
    parser.add_argument('--output', default='work/inventory.csv',
                        help='Output CSV path')
    
    args = parser.parse_args()
    
    # Load config
    project_root = Path(__file__).parent.parent
    config_path = project_root / args.config
    data_src = project_root / args.data_src
    output_path = project_root / args.output
    
    config = load_config(config_path)
    
    # Compile section filter
    section_pattern = re.compile(args.section_filter)
    
    print(f"[01_inventory] Scanning {data_src} with section filter: {args.section_filter}")
    
    # Find all section directories
    section_dirs = [d for d in data_src.iterdir() 
                    if d.is_dir() and section_pattern.match(d.name)]
    
    if not section_dirs:
        print(f"[01_inventory] ⚠ No sections match filter: {args.section_filter}")
        return
    
    print(f"[01_inventory] Found {len(section_dirs)} matching section(s):")
    for sd in section_dirs:
        print(f"  - {sd.name}")
    
    # Collect image files
    image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
    rows = []
    type_counts = defaultdict(int)
    
    for section_dir in section_dirs:
        section_name = section_dir.name
        
        # Extract section_id (e.g., "11" from "11 - Engine")
        section_id_match = re.match(r'^(\d+)', section_name)
        section_id = section_id_match.group(1) if section_id_match else section_name
        
        image_files = [f for f in section_dir.iterdir() 
                       if f.is_file() and f.suffix.lower() in image_exts]
        
        for img_file in sorted(image_files):
            rel_path = img_file.relative_to(data_src)
            page_no = extract_page_no(img_file.name)
            file_size = img_file.stat().st_size
            sha1 = sha1_file(img_file)
            guessed_type = guess_type(None, rel_path)
            
            rows.append({
                'rel_path': str(rel_path),
                'section_dir': section_name,
                'section_id': section_id,
                'page_no': page_no,
                'bytes': file_size,
                'sha1': sha1,
                'guessed_type': guessed_type
            })
            
            type_counts[guessed_type if guessed_type else '(unknown)'] += 1
    
    # Write CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = ['rel_path', 'section_dir', 'section_id', 'page_no', 
                  'bytes', 'sha1', 'guessed_type']
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"[01_inventory] ✓ Wrote {len(rows)} images to {output_path}")
    
    # Validation checks
    print(f"[01_inventory] Validation:")
    print(f"  ✓ All paths exist: {all(Path(data_src / r['rel_path']).exists() for r in rows)}")
    print(f"  ✓ No duplicates: {len(rows) == len(set(r['rel_path'] for r in rows))}")
    print(f"  ✓ All sha1 non-empty: {all(r['sha1'] for r in rows)}")
    
    # Type distribution
    print(f"\n[01_inventory] Type distribution:")
    for dtype, count in sorted(type_counts.items()):
        print(f"  {dtype}: {count}")
    
    # Sample rows
    print(f"\n[01_inventory] Sample rows (first 5):")
    for i, row in enumerate(rows[:5]):
        print(f"  {i+1}. {row['rel_path']} | section={row['section_id']} | page={row['page_no']} | {row['bytes']} bytes")
    
    print(f"\n[01_inventory] Done: {len(rows)} images cataloged")


if __name__ == '__main__':
    main()
