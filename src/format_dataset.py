#!/usr/bin/env python3
"""
Dataset Formatting CLI Tool for SocioHack Training

Command-line interface for converting various dataset formats (CSV, JSONL, JSON,
HuggingFace datasets) into the standardized SocioHack training format.

This script wraps the format_sft_dataset function from data_loader module and provides
a user-friendly CLI for dataset preparation.

Usage:
    python -m src.format_dataset path/to/dataset.csv \\
        --prompt-column question \\
        --reference-column answer \\
        --output-dir ~/.cache/huggingface/datasets \\
        --dataset-name my_dataset

Features:
    - Automatic format detection (CSV, JSON, JSONL, HF datasets)
    - Column mapping to SocioHack standard format
    - Configurable output directory and dataset naming
    - Split selection for multi-split datasets
"""

import argparse
import os
import sys
import traceback
from .data_loader import format_sft_dataset

# Default HuggingFace cache directory
DEFAULT_HF_HOME = os.path.expanduser("~/.cache/huggingface")

def get_dataset_path(dataset_name):
    """
    Construct full path to dataset directory in HuggingFace cache.
    
    Args:
        dataset_name (str): Name of the dataset
    
    Returns:
        str: Full path to dataset directory
    """
    return os.path.join(DEFAULT_HF_HOME, "datasets", dataset_name)

def main():
    """
    Main entry point for dataset formatting CLI.
    
    Parses command-line arguments and invokes format_sft_dataset to convert
    datasets from various formats to SocioHack standard format.
    """
    parser = argparse.ArgumentParser(
        description="Format any SFT dataset for SocioHack training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Format CSV file
  python -m src.format_dataset data/train.csv \\
      --prompt-column question --reference-column answer

  # Format HuggingFace dataset
  python -m src.format_dataset gsm8k \\
      --prompt-column question --reference-column answer \\
      --split train

  # Format JSONL with custom output
  python -m src.format_dataset data/dataset.jsonl \\
      --output-dir /path/to/output \\
      --dataset-name my_formatted_data
        """
    )
    
    parser.add_argument(
        "dataset_source",
        type=str,
        help="Path to dataset file (CSV/JSONL/JSON) or HuggingFace dataset name"
    )
    
    parser.add_argument(
        "--prompt-column",
        type=str,
        default="prompt",
        help="Column name containing the prompts/questions (default: 'prompt')"
    )
    
    parser.add_argument(
        "--reference-column",
        type=str,
        default="reference",
        help="Column name containing reference answers (default: 'reference')"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: HF_HOME/datasets/{dataset_name}"
    )
    
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Name for saved dataset folder (default: derived from source filename)"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use for HF datasets (default: 'train')"
    )
    
    args = parser.parse_args()
    
    # Auto-generate dataset name if not provided
    if args.dataset_name is None:
        if os.path.exists(args.dataset_source):
            # For local files, use filename without extension
            args.dataset_name = os.path.splitext(os.path.basename(args.dataset_source))[0]
        else:
            # For HF datasets, replace slashes with underscores
            args.dataset_name = args.dataset_source.replace("/", "_")
    
    # Set default output directory if not specified
    if args.output_dir is None:
        args.output_dir = get_dataset_path(args.dataset_name)
    
    # Display formatting configuration
    print("="*60)
    print("SocioHack Dataset Formatting")
    print("="*60)
    print(f"Source: {args.dataset_source}")
    print(f"Prompt column: {args.prompt_column}")
    print(f"Reference column: {args.reference_column}")
    print(f"Output directory: {args.output_dir}")
    print(f"Dataset name: {args.dataset_name}")
    print("="*60)
    
    try:
        # Format and save the dataset
        dataset_path = format_sft_dataset(
            dataset_source=args.dataset_source,
            prompt_column=args.prompt_column,
            reference_column=args.reference_column,
            output_dir=args.output_dir,
            split=args.split,
            dataset_name=args.dataset_name
        )
        
        print("\n" + "="*60)
        print("✓ SUCCESS: Dataset formatted successfully!")
        print("="*60)
        print(f"Saved to: {dataset_path}")
        print("\nNext steps:")
        print("1. Update your config.yaml:")
        print(f'   dataset:')
        print(f'     name: "{args.dataset_name}"')
        print(f'     hf_home: "{os.path.dirname(os.path.dirname(dataset_path))}"')
        print("\n2. Start training:")
        print("   python -m src.train")
        print("="*60)
        
    except Exception as e:
        print("\n" + "="*60)
        print("✗ ERROR: Dataset formatting failed")
        print("="*60)
        print(f"Error: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        print("="*60)
        sys.exit(1)

if __name__ == "__main__":
    main() 