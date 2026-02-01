#!/usr/bin/env python3
"""
Split a data file into multiple parts, each with headers, and compress them.
Usage: python split_data.py input_file=path/to/file output_path=path/ n_files=10
"""

import gzip
import os
import sys
import re

def parse_arguments():
    """Parse command-line arguments in the format key=value"""
    args = {}
    for arg in sys.argv[1:]:
        if '=' in arg:
            key, value = arg.split('=', 1)
            args[key] = value
        else:
            print(f"Warning: Ignoring argument '{arg}' (expected format: key=value)")
    return args

def get_file_extension(filename):
    """Extract file extension(s), handling .csv.gz, .txt.gz, etc."""
    if filename.endswith('.gz'):
        # Remove .gz and get the extension before it
        base = filename[:-3]
        ext = os.path.splitext(base)[1] + '.gz'
        return ext
    else:
        return os.path.splitext(filename)[1]

def get_base_filename(filename):
    """Get base filename without path and extension"""
    basename = os.path.basename(filename)
    # Remove .gz if present
    if basename.endswith('.gz'):
        basename = basename[:-3]
    # Remove extension
    basename = os.path.splitext(basename)[0]
    return basename

def read_file_lines(filepath):
    """Read lines from gzipped file - always assumes input is gzipped"""
    # Read file in binary mode first, then decompress
    # This avoids EOFError issues with gzip.open() in text mode
    with open(filepath, 'rb') as f:
        compressed_data = f.read()
    
    # Decompress the data
    try:
        decompressed_data = gzip.decompress(compressed_data)
    except Exception as e:
        # If decompress fails, try with GzipFile which handles some edge cases
        import io
        f = io.BytesIO(compressed_data)
        decompressor = gzip.GzipFile(fileobj=f, mode='rb')
        decompressed_data = decompressor.read()
        decompressor.close()
    
    # Decode to string
    text = decompressed_data.decode('utf-8', errors='replace')
    
    # Split into lines, preserving line endings
    lines = text.splitlines(keepends=True)
    
    # Ensure last line has newline if file didn't end with one
    if lines and not lines[-1].endswith('\n'):
        lines[-1] = lines[-1] + '\n'
    
    return lines

def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    # Get required arguments
    input_file = args.get('input_file')
    output_path = args.get('output_path')
    n_files = args.get('n_files')
    
    if not input_file:
        print("Error: input_file is required")
        print("Usage: python split_data.py input_file=path/to/file output_path=path/ n_files=10")
        sys.exit(1)
    
    if not output_path:
        print("Error: output_path is required")
        print("Usage: python split_data.py input_file=path/to/file output_path=path/ n_files=10")
        sys.exit(1)
    
    if not n_files:
        print("Error: n_files is required")
        print("Usage: python split_data.py input_file=path/to/file output_path=path/ n_files=10")
        sys.exit(1)
    
    try:
        n_files = int(n_files)
        if n_files < 1:
            raise ValueError("n_files must be at least 1")
    except ValueError as e:
        print(f"Error: n_files must be a positive integer: {e}")
        sys.exit(1)
    
    # Ensure output_path ends with /
    if not output_path.endswith('/') and not output_path.endswith('\\'):
        output_path = output_path + '/'
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist")
        sys.exit(1)
    
    # Get base filename and extension
    base_filename = get_base_filename(input_file)
    extension = get_file_extension(input_file)
    
    print(f"Reading {input_file}...")
    lines = read_file_lines(input_file)
    
    total_lines = len(lines)
    print(f"Total lines: {total_lines:,}")
    
    if total_lines == 0:
        print("Error: Input file is empty")
        sys.exit(1)
    
    header = lines[0]
    data_lines = lines[1:]
    
    # Calculate lines per part
    lines_per_part = len(data_lines) // n_files
    remainder = len(data_lines) % n_files
    
    print(f"Splitting into {n_files} parts...")
    print(f"Lines per part: ~{lines_per_part:,} (excluding header)")
    
    # Split data into parts
    parts = []
    start_idx = 0
    
    for i in range(n_files):
        # Distribute remainder across first parts
        part_size = lines_per_part + (1 if i < remainder else 0)
        end_idx = start_idx + part_size
        
        part_lines = [header] + data_lines[start_idx:end_idx]
        parts.append(part_lines)
        
        start_idx = end_idx
    
    # Write output files
    file_sizes = []
    for i, part_lines in enumerate(parts, 1):
        part_num = f"{i:02d}"  # Zero-padded part number (01, 02, etc.)
        output_file = f"{output_path}{base_filename}.part{part_num}{extension}"
        
        print(f"Writing {output_file} ({len(part_lines):,} lines including header)...")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        
        # Write compressed file
        with gzip.open(output_file, 'wt', encoding='utf-8') as f:
            f.writelines(part_lines)
        
        # Check file size
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        file_sizes.append(size_mb)
        print(f"  Size: {size_mb:.2f} MB")
    
    # Summary
    print(f"\nSummary:")
    print(f"  Total parts: {n_files}")
    print(f"  Average size: {sum(file_sizes) / len(file_sizes):.2f} MB")
    print(f"  Largest part: {max(file_sizes):.2f} MB")
    print(f"  Smallest part: {min(file_sizes):.2f} MB")
    
    MAX_SIZE_MB = 100
    if any(size > MAX_SIZE_MB for size in file_sizes):
        print(f"\nWarning: One or more parts exceed {MAX_SIZE_MB}MB!")
    else:
        print(f"\nSuccess: All parts are under {MAX_SIZE_MB}MB!")
    
    print("Done!")

if __name__ == '__main__':
    main()
