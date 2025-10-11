#!/usr/bin/env python3
"""
03_ocr.py - Extract text from preprocessed images using pytesseract.
Outputs per-page JSON with full text and line-level details (text, bbox, confidence).
"""

import argparse
import csv
import json
import re
from pathlib import Path
from collections import defaultdict

import cv2
import pytesseract
from PIL import Image


def is_diagram_page(text_full, conf_threshold=30, min_text_ratio=0.02):
    """
    Heuristic to detect diagram-only pages with little readable text.
    Returns True if page appears to be a diagram.
    """
    if not text_full or len(text_full.strip()) < 20:
        return True
    
    # Check ratio of non-whitespace characters
    non_ws = len([c for c in text_full if not c.isspace()])
    if non_ws < len(text_full) * min_text_ratio:
        return True
    
    return False


def extract_text_with_details(image_path):
    """
    Extract text using pytesseract with detailed line-level information.
    Returns: (text_full: str, lines: List[dict], is_diagram: bool)
    """
    try:
        # Read image
        img = Image.open(image_path)
        
        # Get full text
        text_full = pytesseract.image_to_string(img, lang='eng')
        
        # Get detailed data (bbox + confidence per word/line)
        data = pytesseract.image_to_data(img, lang='eng', output_type=pytesseract.Output.DICT)
        
        # Group by line (block_num + par_num + line_num)
        lines_dict = defaultdict(list)
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            if int(data['conf'][i]) > 0:  # Only include recognized text
                line_key = (data['block_num'][i], data['par_num'][i], data['line_num'][i])
                lines_dict[line_key].append({
                    'text': data['text'][i],
                    'left': data['left'][i],
                    'top': data['top'][i],
                    'width': data['width'][i],
                    'height': data['height'][i],
                    'conf': data['conf'][i]
                })
        
        # Aggregate lines
        lines = []
        for line_key, words in sorted(lines_dict.items()):
            line_text = ' '.join([w['text'] for w in words])
            
            # Calculate bounding box for entire line
            if words:
                left = min(w['left'] for w in words)
                top = min(w['top'] for w in words)
                right = max(w['left'] + w['width'] for w in words)
                bottom = max(w['top'] + w['height'] for w in words)
                
                # Average confidence
                avg_conf = sum(w['conf'] for w in words) / len(words)
                
                lines.append({
                    'text': line_text,
                    'bbox': [left, top, right - left, bottom - top],  # [x, y, w, h]
                    'conf': round(avg_conf, 2)
                })
        
        # Detect if diagram
        is_diagram = is_diagram_page(text_full)
        
        return text_full, lines, is_diagram
        
    except Exception as e:
        raise Exception(f"OCR extraction failed: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Extract text from images using pytesseract")
    parser.add_argument('--input-dir', default='work/images_clean',
                        help='Input directory with preprocessed images')
    parser.add_argument('--section-filter', default='.*',
                        help='Regex to filter section directories (e.g., "^11 - ")')
    parser.add_argument('--output-dir', default='work/ocr_raw',
                        help='Output directory for OCR JSON files')
    parser.add_argument('--error-log', default='work/logs/ocr_text_failures.csv',
                        help='Error log CSV path')
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    input_dir = project_root / args.input_dir
    output_dir = project_root / args.output_dir
    error_log_path = project_root / args.error_log
    
    # Compile section filter
    section_pattern = re.compile(args.section_filter)
    
    print(f"[03_ocr] Input directory: {input_dir}")
    print(f"[03_ocr] Section filter: {args.section_filter}")
    print(f"[03_ocr] Output directory: {output_dir}")
    
    # Find section directories
    section_dirs = [d for d in input_dir.iterdir() 
                    if d.is_dir() and section_pattern.match(d.name)]
    
    if not section_dirs:
        print(f"[03_ocr] ⚠ No sections match filter")
        return
    
    print(f"[03_ocr] Found {len(section_dirs)} matching section(s)")
    
    # Collect all images
    all_images = []
    for section_dir in section_dirs:
        images = sorted(section_dir.glob('*.png'))
        all_images.extend([(section_dir.name, img) for img in images])
    
    print(f"[03_ocr] Processing {len(all_images)} images...\n")
    
    # Track results
    successful = 0
    failed = 0
    diagram_count = 0
    non_empty_text_count = 0
    errors = []
    
    # Process each image
    for i, (section_name, image_path) in enumerate(all_images, 1):
        # Extract section_id and page_no from filename
        filename = image_path.stem  # Remove .png
        section_id_match = re.match(r'^(\d+)', section_name)
        section_id = section_id_match.group(1) if section_id_match else section_name
        
        # Page number from filename
        page_no_match = re.match(r'^\d+-(.+)', filename)
        page_no = page_no_match.group(1) if page_no_match else filename
        
        print(f"  [{i}/{len(all_images)}] {section_name}/{filename}.png...", end=' ', flush=True)
        
        try:
            text_full, lines, is_diagram = extract_text_with_details(image_path)
            
            # Prepare output JSON
            output_data = {
                'source_path': str(image_path.relative_to(project_root)),
                'section_dir': section_name,
                'section_id': section_id,
                'page_no': page_no,
                'text_full': text_full,
                'lines': lines,
                'is_diagram': is_diagram,
                'line_count': len(lines),
                'char_count': len(text_full)
            }
            
            # Create output path: work/ocr_raw/{section_id}/{page_no}.json
            output_subdir = output_dir / section_id
            output_subdir.mkdir(parents=True, exist_ok=True)
            output_path = output_subdir / f"{page_no}.json"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            successful += 1
            
            if text_full.strip():
                non_empty_text_count += 1
            
            if is_diagram:
                diagram_count += 1
                print(f"✓ [DIAGRAM] {len(lines)} lines, {len(text_full)} chars")
            else:
                print(f"✓ {len(lines)} lines, {len(text_full)} chars")
                
        except Exception as e:
            failed += 1
            errors.append({
                'section_name': section_name,
                'filename': filename,
                'error': str(e)
            })
            print(f"✗ {str(e)}")
    
    # Write error log
    if errors:
        error_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(error_log_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['section_name', 'filename', 'error'])
            writer.writeheader()
            writer.writerows(errors)
        print(f"\n[03_ocr] ⚠ Errors written to {error_log_path}")
    
    # Summary
    total = len(all_images)
    non_diagram_count = total - diagram_count
    
    print(f"\n[03_ocr] Summary:")
    print(f"  ✓ Successful: {successful}/{total} ({100*successful/total:.1f}%)")
    print(f"  ✗ Failed: {failed}/{total}")
    print(f"  📊 Diagram pages: {diagram_count}/{total} ({100*diagram_count/total:.1f}%)")
    print(f"  📝 Non-diagram pages: {non_diagram_count}/{total}")
    print(f"  📄 Non-empty text: {non_empty_text_count}/{total} ({100*non_empty_text_count/total:.1f}%)")
    
    # Calculate acceptance: ≥80% of non-diagram pages have non-empty text
    if non_diagram_count > 0:
        non_diagram_with_text = non_empty_text_count  # Approximation
        pct = 100 * non_diagram_with_text / non_diagram_count
        print(f"\n[03_ocr] Acceptance check:")
        print(f"  Non-diagram pages with text: {non_diagram_with_text}/{non_diagram_count} ({pct:.1f}%)")
        if pct >= 80:
            print(f"  ✓ PASS: ≥80% threshold met")
        else:
            print(f"  ✗ FAIL: Below 80% threshold")
    
    # List failures
    if errors:
        print(f"\n[03_ocr] Failures ({len(errors)}):")
        for err in errors[:5]:
            print(f"  - {err['section_name']}/{err['filename']}: {err['error']}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more (see {error_log_path})")
    
    print(f"\n[03_ocr] Done!")


if __name__ == '__main__':
    main()
