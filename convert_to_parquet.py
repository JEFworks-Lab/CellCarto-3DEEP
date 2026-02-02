#!/usr/bin/env python3
"""
Convert gzipped TSV files to shuffled, sharded Parquet files.

This script:
1. Reads all TSV.gz files from the data/ directory
2. Shuffles all rows randomly
3. Splits into 10 shards (each ~10% of data) for progressive loading
4. Each shard is a random sample - loading N shards = N*10% sample

Requirements:
    pip install pandas pyarrow
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import glob
import time
import numpy as np

# Configuration
DATA_DIR = Path(__file__).parent / "data"
OUTPUT_PREFIX = "hairfollicle"
NUM_SHARDS = 10  # 10 shards = each shard is 10% of data
RANDOM_SEED = 42  # For reproducibility

# Column type specifications
FLOAT32_COLUMNS = [
    'x', 'y', 'z',
    'transformedX', 'transformedY', 'transformedZ',
    'X_shifted', 'Time'
]

INT16_COLUMNS = ['TimeRank']  # Small integers

CATEGORICAL_COLUMNS = [
    'Gene', 'CellType', 'Structure', 'HF', 'Sample', 'Group'
]


def load_and_combine_files():
    """Load all TSV.gz files and combine into a single DataFrame."""
    files = sorted(glob.glob(str(DATA_DIR / "*.tsv.gz")))
    
    if not files:
        raise FileNotFoundError(f"No .tsv.gz files found in {DATA_DIR}")
    
    print(f"Found {len(files)} files to process:")
    for f in files:
        print(f"  - {Path(f).name}")
    
    dfs = []
    total_rows = 0
    
    for i, file_path in enumerate(files, 1):
        print(f"\nLoading file {i}/{len(files)}: {Path(file_path).name}")
        start = time.time()
        
        df = pd.read_csv(file_path, sep='\t', compression='gzip')
        rows = len(df)
        total_rows += rows
        
        elapsed = time.time() - start
        print(f"  Loaded {rows:,} rows in {elapsed:.1f}s")
        
        dfs.append(df)
    
    print(f"\nCombining {len(dfs)} DataFrames ({total_rows:,} total rows)...")
    combined = pd.concat(dfs, ignore_index=True)
    
    return combined


def optimize_dtypes(df):
    """Optimize DataFrame column types for minimal size."""
    print("\nOptimizing data types...")
    
    original_size = df.memory_usage(deep=True).sum()
    print(f"  Original memory usage: {original_size / 1e6:.1f} MB")
    
    # Convert float64 → float32 for coordinate columns
    for col in FLOAT32_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype('float32')
            print(f"  {col}: float64 → float32")
    
    # Convert to int16 for small integer columns
    for col in INT16_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype('int16')
            print(f"  {col}: → int16")
    
    # Convert to categorical for low-cardinality columns
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            n_unique = df[col].nunique()
            df[col] = df[col].astype('category')
            print(f"  {col}: string → category ({n_unique} unique values)")
    
    # For 'cell' column - keep as string (high cardinality)
    if 'cell' in df.columns:
        n_unique = df['cell'].nunique()
        print(f"  cell: keeping as string ({n_unique:,} unique values)")
    
    optimized_size = df.memory_usage(deep=True).sum()
    print(f"\n  Optimized memory usage: {optimized_size / 1e6:.1f} MB")
    print(f"  Reduction: {(1 - optimized_size/original_size) * 100:.1f}%")
    
    return df


def shuffle_and_shard(df, output_dir, prefix, num_shards, seed):
    """Shuffle rows and save to multiple sharded Parquet files."""
    print(f"\nShuffling {len(df):,} rows with seed={seed}...")
    
    # Shuffle the dataframe
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(len(df))
    df_shuffled = df.iloc[shuffled_indices].reset_index(drop=True)
    
    print(f"Saving to {num_shards} Parquet shards (each ~{len(df)//num_shards:,} rows)...")
    
    # Calculate rows per shard
    total_rows = len(df_shuffled)
    rows_per_shard = total_rows // num_shards
    
    total_size = 0
    shard_files = []
    start_total = time.time()
    
    for shard_idx in range(num_shards):
        start_row = shard_idx * rows_per_shard
        # Last shard gets any remaining rows
        end_row = total_rows if shard_idx == num_shards - 1 else (shard_idx + 1) * rows_per_shard
        
        shard_df = df_shuffled.iloc[start_row:end_row]
        shard_path = output_dir / f"{prefix}.shard{shard_idx + 1:02d}.parquet"
        
        print(f"\n  Shard {shard_idx + 1}/{num_shards}: {shard_path.name}")
        print(f"    Rows: {start_row:,} - {end_row:,} ({len(shard_df):,} rows)")
        
        # Create PyArrow table
        table = pa.Table.from_pandas(shard_df, preserve_index=False)
        
        # Write with ZSTD compression
        start = time.time()
        pq.write_table(
            table,
            shard_path,
            compression='zstd',
            compression_level=9,
            use_dictionary=True,
            write_statistics=True,
            row_group_size=500_000,
        )
        elapsed = time.time() - start
        
        # Report file size
        file_size = shard_path.stat().st_size
        total_size += file_size
        shard_files.append(shard_path)
        
        print(f"    Size: {file_size / 1e6:.1f} MB (wrote in {elapsed:.1f}s)")
        
        if file_size > 100 * 1e6:
            print(f"    WARNING: Shard exceeds 100MB GitHub limit!")
    
    elapsed_total = time.time() - start_total
    print(f"\n  Total: {total_size / 1e6:.1f} MB across {num_shards} files ({elapsed_total:.1f}s)")
    
    return shard_files, total_size


def main():
    print("=" * 60)
    print("Converting TSV.gz files to shuffled Parquet shards")
    print("=" * 60)
    print(f"\nEach shard contains a random {100//NUM_SHARDS}% sample of the data.")
    print("Loading N shards = N * 10% sample rate.\n")
    
    start_total = time.time()
    
    # Load all files
    df = load_and_combine_files()
    
    # Print column info
    print(f"\nColumns: {list(df.columns)}")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Optimize data types
    df = optimize_dtypes(df)
    
    # Remove old parquet files
    old_files = list(DATA_DIR.glob("*.parquet"))
    if old_files:
        print(f"\nRemoving {len(old_files)} old parquet files...")
        for f in old_files:
            f.unlink()
            print(f"  Deleted: {f.name}")
    
    # Shuffle and save to sharded Parquet files
    shard_files, total_size = shuffle_and_shard(df, DATA_DIR, OUTPUT_PREFIX, NUM_SHARDS, RANDOM_SEED)
    
    # Calculate total original size for comparison
    original_files = sorted(glob.glob(str(DATA_DIR / "*.tsv.gz")))
    original_size = sum(Path(f).stat().st_size for f in original_files)
    
    elapsed_total = time.time() - start_total
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Original (gzipped TSV): {original_size / 1e6:.1f} MB ({len(original_files)} files)")
    print(f"  New (Parquet):          {total_size / 1e6:.1f} MB ({NUM_SHARDS} shards)")
    print(f"  Size reduction:         {(1 - total_size/original_size) * 100:.1f}%")
    print(f"  Total time:             {elapsed_total:.1f}s")
    print(f"\n  Output files (each shard = {100//NUM_SHARDS}% sample):")
    for f in shard_files:
        size = f.stat().st_size / 1e6
        print(f"    - {f.name} ({size:.1f} MB)")
    
    print(f"\n  Usage in app:")
    print(f"    - 10% sample = load shard01 only (~{shard_files[0].stat().st_size / 1e6:.0f} MB)")
    print(f"    - 50% sample = load shards 01-05 (~{sum(f.stat().st_size for f in shard_files[:5]) / 1e6:.0f} MB)")
    print(f"    - 100% sample = load all shards (~{total_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
