#!/usr/bin/env python3
"""
Fix validation CSV file paths to match the data_val folder structure.
Updates path prefixes for voices and faces columns.
"""

import pandas as pd
from pathlib import Path

def correct_val_csv_paths(csv_file):
    """
    Correct the paths in a validation CSV file.
    
    Args:
        csv_file: Path to the CSV file to correct
    """
    print(f"\n{'='*80}")
    print(f"Processing: {csv_file.name}")
    print(f"{'='*80}")
    
    # Read CSV
    print(f"Reading {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
    print(f"  Columns: {list(df.columns)}")
    
    # Show before sample
    print(f"\nBefore (first row):")
    if 'voices' in df.columns:
        print(f"  voices: {df['voices'].iloc[0]}")
    if 'faces' in df.columns:
        print(f"  faces: {df['faces'].iloc[0]}")
    
    # Apply transformations
    print(f"\nApplying path transformations...")
    
    # voices: val/v1/voices/ -> data_val/val/v1/voices/
    if 'voices' in df.columns:
        df['voices'] = df['voices'].str.replace(
            'val/v1/voices/', 'data_val/val/v1/voices/', regex=False
        )
    
    # faces: val/v1/faces/ -> data_val/val/v1/faces/
    if 'faces' in df.columns:
        df['faces'] = df['faces'].str.replace(
            'val/v1/faces/', 'data_val/val/v1/faces/', regex=False
        )
    
    # Show after sample
    print(f"\nAfter (first row):")
    if 'voices' in df.columns:
        print(f"  voices: {df['voices'].iloc[0]}")
    if 'faces' in df.columns:
        print(f"  faces: {df['faces'].iloc[0]}")
    
    # Write back to CSV
    print(f"\nWriting corrected CSV...")
    df.to_csv(csv_file, index=False)
    print(f"✓ Successfully saved: {csv_file}")
    
    return df


def main():
    base_dir = Path('/home/ankit/Desktop/iit/polygot/multimodal_code_learning/code_exp4_audio_face')
    csv_dir = base_dir / 'data_train' / 'comp'
    
    # Process both validation CSVs
    english_csv = csv_dir / 'v1_val_English.csv'
    urdu_csv = csv_dir / 'v1_val_Urdu.csv'
    
    df_english = correct_val_csv_paths(english_csv)
    df_urdu = correct_val_csv_paths(urdu_csv)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"✓ v1_val_English.csv: {len(df_english)} rows corrected")
    print(f"✓ v1_val_Urdu.csv: {len(df_urdu)} rows corrected")
    print(f"\nAll path prefixes have been updated successfully!")
    print(f"Paths now point to:")
    print(f"  - voices: data_val/val/v1/voices/")
    print(f"  - faces: data_val/val/v1/faces/")


if __name__ == '__main__':
    main()
