#!/usr/bin/env python3
"""
03b_ocr_tables.py - Extract tables from preprocessed images using heuristics.
Detects table-like structures using line detection and text grouping.
Outputs CSV with spec_name, value_raw, unit_raw, notes, source info.
"""

import argparse
import csv
import json
import re
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pytesseract
from PIL import Image


def detect_table_lines(img_path, min_line_length=100):
    """
    Detect horizontal and vertical lines in image using HoughLinesP.
    Returns (has_table: bool, h_lines: list, v_lines: list)
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False, [], []
    
    # Threshold and invert
    _, binary = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Detect horizontal lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    h_lines_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    h_lines = cv2.HoughLinesP(h_lines_img, 1, np.pi/180, threshold=50, 
                               minLineLength=min_line_length, maxLineGap=10)
    
    # Detect vertical lines
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    v_lines_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    v_lines = cv2.HoughLinesP(v_lines_img, 1, np.pi/180, threshold=50,
                               minLineLength=min_line_length, maxLineGap=10)
    
    has_h_lines = h_lines is not None and len(h_lines) > 2
    has_v_lines = v_lines is not None and len(v_lines) > 1
    has_table = has_h_lines and has_v_lines
    
    return has_table, h_lines if h_lines is not None else [], v_lines if v_lines is not None else []


def parse_spec_line(line_text):
    """
    Parse a line that looks like a specification entry.
    Returns (spec_name, value_raw, unit_raw, notes) or None if not parseable.
    
    Patterns to match:
    - "Torque setting: 25 Nm"
    - "Clearance 0.05 - 0.10 mm"
    - "Capacity: 4.5 L (approx)"
    """
    line = line_text.strip()
    
    # Pattern 1: "Name: value unit (notes)"
    pattern1 = r'^([^:]+?):\s*([0-9.,~≈><\-\s]+)\s*([A-Za-z°µ]+)?\s*(\(.*\))?$'
    match1 = re.match(pattern1, line)
    if match1:
        spec_name = match1.group(1).strip()
        value_raw = match1.group(2).strip()
        unit_raw = match1.group(3).strip() if match1.group(3) else ""
        notes = match1.group(4).strip('() ') if match1.group(4) else ""
        return spec_name, value_raw, unit_raw, notes
    
    # Pattern 2: "Name value unit"
    pattern2 = r'^([A-Za-z\s]+?)\s+([0-9.,~≈><\-\s]+)\s+([A-Za-z°µ]+)$'
    match2 = re.match(pattern2, line)
    if match2:
        spec_name = match2.group(1).strip()
        value_raw = match2.group(2).strip()
        unit_raw = match2.group(3).strip()
        return spec_name, value_raw, unit_raw, ""
    
    return None


def extract_tables_from_ocr(ocr_json_path, img_path):
    """
    Extract table data from OCR JSON using heuristics.
    Returns list of dicts: {spec_name, value_raw, unit_raw, notes, source_path, section_id, page_no}
    """
    # Load OCR data
    with open(ocr_json_path) as f:
        ocr_data = json.load(f)
    
    # Check if page has table structure (lines detection)
    has_table, h_lines, v_lines = detect_table_lines(img_path)
    
    if not has_table:
        return []  # No table detected
    
    # Parse lines for spec-like patterns
    rows = []
    for line in ocr_data['lines']:
        line_text = line['text'].strip()
        parsed = parse_spec_line(line_text)
        
        if parsed:
            spec_name, value_raw, unit_raw, notes = parsed
            rows.append({
                'spec_name': spec_name,
                'value_raw': value_raw,
                'unit_raw': unit_raw,
                'notes': notes,
                'source_path': ocr_data['source_path'],
                'section_id': ocr_data['section_id'],
                'page_no': ocr_data['page_no']
            })
    
    return rows


def main():
    parser = argparse.ArgumentParser(description="Extract tables from OCR data")
    parser.add_argument('--ocr-dir', default='work/ocr_raw',
                        help='OCR JSON directory')
    parser.add_argument('--images-dir', default='work/images_clean',
                        help='Preprocessed images directory')
    parser.add_argument('--section-filter', default='.*',
                        help='Regex to filter sections (e.g., "^11 - ")')
    parser.add_argument('--output-dir', default='work/ocr_tables',
                        help='Output directory for table CSVs')
    parser.add_argument('--error-log', default='work/logs/table_detect_fail.csv',
                        help='Log for pages where table detection failed')
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    ocr_dir = project_root / args.ocr_dir
    images_dir = project_root / args.images_dir
    output_dir = project_root / args.output_dir
    error_log_path = project_root / args.error_log
    
    # Compile section filter
    section_pattern = re.compile(args.section_filter)
    
    print(f"[03b_ocr_tables] OCR directory: {ocr_dir}")
    print(f"[03b_ocr_tables] Images directory: {images_dir}")
    print(f"[03b_ocr_tables] Section filter: {args.section_filter}")
    print(f"[03b_ocr_tables] Output directory: {output_dir}")
    
    # Find section subdirectories in OCR output
    section_dirs = [d for d in ocr_dir.iterdir() if d.is_dir()]
    
    if not section_dirs:
        print(f"[03b_ocr_tables] ⚠ No section directories found in {ocr_dir}")
        return
    
    print(f"[03b_ocr_tables] Found {len(section_dirs)} section(s)")
    
    # Process each section
    all_rows = []
    failed_pages = []
    pages_with_tables = 0
    pages_without_tables = 0
    
    for section_dir in section_dirs:
        section_id = section_dir.name
        
        # Check if this section matches filter
        # Find corresponding section_dir name in images
        image_section_dirs = list(images_dir.glob(f"*{section_id}*"))
        if not image_section_dirs:
            continue
        
        image_section_dir = image_section_dirs[0]
        if not section_pattern.match(image_section_dir.name):
            continue
        
        print(f"\n[03b_ocr_tables] Processing section {section_id}...")
        
        json_files = sorted(section_dir.glob('*.json'))
        
        for json_file in json_files:
            page_no = json_file.stem
            
            # Find corresponding image
            img_path = image_section_dir / f"{page_no}.png"
            if not img_path.exists():
                # Try with section prefix
                img_candidates = list(image_section_dir.glob(f"*{page_no}.png"))
                if img_candidates:
                    img_path = img_candidates[0]
                else:
                    failed_pages.append({
                        'section_id': section_id,
                        'page_no': page_no,
                        'reason': 'Image not found'
                    })
                    continue
            
            print(f"  {page_no}.json...", end=' ', flush=True)
            
            try:
                rows = extract_tables_from_ocr(json_file, img_path)
                
                if rows:
                    all_rows.extend(rows)
                    pages_with_tables += 1
                    print(f"✓ {len(rows)} rows")
                else:
                    pages_without_tables += 1
                    failed_pages.append({
                        'section_id': section_id,
                        'page_no': page_no,
                        'reason': 'No table detected or no spec rows parsed'
                    })
                    print("○ no table")
                    
            except Exception as e:
                failed_pages.append({
                    'section_id': section_id,
                    'page_no': page_no,
                    'reason': str(e)
                })
                print(f"✗ {str(e)}")
    
    # Write combined CSV for all sections
    if all_rows:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Group by section_id
        sections_with_data = set(row['section_id'] for row in all_rows)
        
        for section_id in sections_with_data:
            section_rows = [r for r in all_rows if r['section_id'] == section_id]
            output_csv = output_dir / f"{section_id}.csv"
            
            with open(output_csv, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['spec_name', 'value_raw', 'unit_raw', 'notes', 
                              'source_path', 'section_id', 'page_no']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(section_rows)
            
            print(f"\n[03b_ocr_tables] ✓ Wrote {len(section_rows)} rows to {output_csv}")
    
    # Write failure log
    if failed_pages:
        error_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(error_log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['section_id', 'page_no', 'reason'])
            writer.writeheader()
            writer.writerows(failed_pages)
        print(f"\n[03b_ocr_tables] Logged {len(failed_pages)} misses to {error_log_path}")
    
    # Summary
    total_pages = pages_with_tables + pages_without_tables
    print(f"\n[03b_ocr_tables] Summary:")
    print(f"  Pages processed: {total_pages}")
    print(f"  Pages with tables: {pages_with_tables}")
    print(f"  Pages without tables: {pages_without_tables}")
    print(f"  Total rows extracted: {len(all_rows)}")
    
    # Sample rows
    if all_rows:
        print(f"\n[03b_ocr_tables] Sample rows (first 5):")
        for i, row in enumerate(all_rows[:5], 1):
            print(f"  {i}. {row['spec_name']} | {row['value_raw']} {row['unit_raw']} | page {row['page_no']}")
    else:
        print(f"\n[03b_ocr_tables] No table rows extracted (documented in misses log)")
    
    print(f"\n[03b_ocr_tables] Done!")


if __name__ == '__main__':
    main()
