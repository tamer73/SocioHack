"""
Dataset Loading and Formatting Module for SocioHack Training

This module provides utilities for loading and formatting datasets for SocioHack
training with GRPO. It supports:
- Loading datasets from HuggingFace cache or disk
- Train/validation splitting with configurable strategies
- Dataset formatting from various sources (HF, CSV, JSONL)
- Column mapping and validation

Key Functions:
    - load_sociohack_dataset: Main dataset loader with train/val split
    - format_sft_dataset: Convert any SFT dataset to SocioHack format
"""

import os
from typing import Optional, Tuple, Dict, Any
from datasets import Dataset, load_from_disk, load_dataset, concatenate_datasets
import json

from .config_manager import get_dataset_path


def load_sociohack_dataset(
    config=None,
    dataset_name: str = "YOUR_DATASET_NAME_UNDER_HF_HOME",
    subset: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 42,
    val_size: int = 0
) -> Tuple[Dataset, Optional[Dataset]]:
    """
    Load and split dataset for SocioHack GRPO training with flexible validation strategies.
    
    This function handles loading datasets from disk and creates appropriate train/validation
    splits based on the configuration. It supports:
    1. Using existing validation splits from the dataset
    2. Creating validation split from training data
    3. Training without validation (val_size=0)
    
    Args:
        config: Hydra configuration object with dataset settings (path, name, etc.)
        dataset_name (str): Name of dataset directory under HF_HOME/datasets/
        subset (Optional[int]): Limit training data to first N examples. None = use all
        shuffle (bool): Whether to shuffle data before splitting. Default True
        seed (int): Random seed for reproducible shuffling. Default 42
        val_size (int): Number of validation examples. 0 = no validation. Default 0
    
    Returns:
        Tuple[Dataset, Optional[Dataset]]: (train_dataset, eval_dataset)
            - train_dataset: Training data (always returned)
            - eval_dataset: Validation data (None if val_size=0)
    """
    dataset_path = get_dataset_path(config, dataset_name)
    try:
        dataset = load_from_disk(dataset_path)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset from {dataset_path}: {e}")
        raise e
    
    print(f"[INFO] Loaded dataset from {dataset_path}, type: {type(dataset)}")

    train_data = dataset
    val_data = None
    
    # Determine if it's a DatasetDict or Dataset
    # DatasetDict behaves like a dict and has keys()
    is_dataset_dict = hasattr(dataset, "keys") and callable(dataset.keys)
    
    if is_dataset_dict:
        if "train" in dataset:
            train_data = dataset["train"]
        elif len(dataset.keys()) > 0:
            # Fallback to first split if 'train' not found
            first_key = list(dataset.keys())[0]
            print(f"[WARN] 'train' split not found, using '{first_key}'")
            train_data = dataset[first_key]
            
        if "validation" in dataset:
            val_data = dataset["validation"]
        elif "test" in dataset:
            val_data = dataset["test"]
    
    # Shuffle initial data
    if shuffle:
        if train_data:
            train_data = train_data.shuffle(seed=seed)
        if val_data:
            val_data = val_data.shuffle(seed=seed)
            
    # Apply subset limit to training data
    if subset is not None and isinstance(subset, int) and subset > 0:
        if train_data:
            train_data = train_data.select(range(min(subset, len(train_data))))
        
    # Validation Logic
    if val_size > 0:
        if val_data is not None:
            # Case 1: Existing validation split, limit size
            val_limit = min(val_size, len(val_data))
            val_data = val_data.select(range(val_limit))
            print(f"[INFO] Using existing validation split ({len(val_data)} samples).")
        else:
            # Case 2: Split from training data
            if train_data and len(train_data) > val_size:
                train_sz = len(train_data) - val_size
                val_data = train_data.select(range(train_sz, len(train_data)))
                train_data = train_data.select(range(train_sz))
                print(f"[INFO] Created validation split from training data: {len(train_data)} train, {len(val_data)} validation")
            else:
                 print(f"[WARNING] Validation size {val_size} >= dataset size {len(train_data) if train_data else 0}. Skipping validation.")
                 val_data = None
    else:
        # Case 3: No validation requested
        val_data = None
        print(f"[INFO] No validation split requested (val_size=0).")

    # REPEAT LOGIC FOR SMALL DATASETS (Critical Fix)
    # Ensure training data is large enough for multi-GPU batching with drop_last=True
    if train_data and len(train_data) > 0:
        required_min_size = 128  # Safe default
        if config:
            try:
                # Calculate effective batch size for one optimization step
                batch_size = config.training.batch_size
                num_gpus = 1
                if hasattr(config.gpu.training, 'num_gpus'):
                    num_gpus = int(config.gpu.training.num_gpus)
                grad_accum = config.training.gradient_accumulation_steps
                
                # Total prompts processed per step across all GPUs
                total_batch_size = num_gpus * grad_accum
                
                # We need enough data for at least a few steps to initialize properly
                # And crucially, the length must be divisible by total_batch_size if drop_last=True
                # to avoid discarding data.
                # Let's aim for target_steps full steps.
                target_steps = 1 
                required_min_size = total_batch_size * target_steps
                
            except Exception:
                pass
        
        if len(train_data) < required_min_size:
            print(f"[INFO] Dataset too small ({len(train_data)} samples). Repeating to reach minimum size {required_min_size}...")
            # Calculate how many repeats needed
            repeat_count = (required_min_size // len(train_data)) + 1
            
            # Repeat the dataset
            train_data = concatenate_datasets([train_data] * repeat_count)
            
        # Truncate to be exactly a multiple of total_batch_size to ensure even distribution
        # This prevents the last batch from being smaller than batch_size per GPU
        if config and 'total_batch_size' in locals():
            current_len = len(train_data)
            remainder = current_len % total_batch_size
            if remainder != 0:
                # Pad to the next multiple instead of truncating (since we can repeat)
                needed = total_batch_size - remainder
                # Take the first 'needed' elements and append
                extra_data = train_data.select(range(needed))
                train_data = concatenate_datasets([train_data, extra_data])
                
        print(f"[INFO] New training dataset size: {len(train_data)}")
            
        # Shuffle again to mix repeated items (mostly relevant if original size > 1)
        if shuffle:
            train_data = train_data.shuffle(seed=seed)
            
    print(f"[INFO] Final dataset sizes: Train={len(train_data) if train_data else 0}, Val={len(val_data) if val_data else 0}")
    
    return train_data, val_data


def format_sft_dataset(
    dataset_source: str,
    prompt_column: str = "prompt",
    reference_column: str = "reference",
    output_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    split: str = "train",
    dataset_name: Optional[str] = None
) -> str:
    """
    Convert any SFT (Supervised Fine-Tuning) dataset to SocioHack training format.
    
    This function provides a unified interface for loading datasets from multiple sources
    (HuggingFace Hub, local CSV, local JSONL/JSON files) and converting them to the
    standardized SocioHack format with "prompt" and "reference" columns.
    
    Supported Input Formats:
        - HuggingFace datasets (by name)
        - Local CSV files (.csv)
        - Local JSONL/JSON files (.jsonl, .json)
    
    Args:
        dataset_source (str): Either HuggingFace dataset name (e.g., "gsm8k") or 
                             local file path (e.g., "./data/train.csv")
        prompt_column (str): Name of column containing prompts/questions. Default "prompt"
        reference_column (str): Name of column with reference answers. Default "reference"
        output_dir (Optional[str]): Directory to save formatted dataset. If None, returns
                                   in-memory dataset without saving
        cache_dir (Optional[str]): Cache directory for HuggingFace datasets
        split (str): Dataset split to use ("train", "test", "validation"). Default "train"
        dataset_name (Optional[str]): Custom name for saved dataset folder. If None,
                                     derives from source filename/name
    
    Returns:
        str or Dataset: Path to saved dataset if output_dir specified, else Dataset object
    
    Output Format:
        All datasets are converted to contain:
        - "prompt": The input prompt/question
        - "reference": The reference answer/completion
        - "env": Optional environment field (preserved if exists in source)
        - "action_list_text": basic actions(events) list 
        - "action_list_text": env dynamics list
    
    Example:
        >>> # Format CSV file
        >>> path = format_sft_dataset(
        ...     dataset_source="./data/my_data.csv",
        ...     prompt_column="question",
        ...     reference_column="answer",
        ...     output_dir="~/.cache/huggingface/datasets",
        ...     dataset_name="my_formatted_dataset"
        ... )
        >>> print(f"Dataset saved to: {path}")
        
        >>> # Format HuggingFace dataset
        >>> path = format_sft_dataset(
        ...     dataset_source="gsm8k",
        ...     prompt_column="question",
        ...     reference_column="answer",
        ...     output_dir="~/.cache/huggingface/datasets"
        ... )
    
    Raises:
        ValueError: If prompt_column or reference_column not found in dataset
        ValueError: If file format is unsupported
        ValueError: If specified split doesn't exist in dataset
    """
    # Check if source is local file or HuggingFace dataset name
    is_local_file = os.path.exists(dataset_source)
    
    # Load dataset from appropriate source
    if is_local_file:
        # Load from local file (CSV, JSON, or JSONL)
        if dataset_source.endswith('.csv'):
            dataset_dict = load_dataset('csv', data_files=dataset_source)
        elif dataset_source.endswith('.jsonl') or dataset_source.endswith('.json'):
            dataset_dict = load_dataset('json', data_files=dataset_source)
        else:
            raise ValueError(f"Unsupported file format: {dataset_source}. Supported: .csv, .json, .jsonl")
        
        # Extract the requested split (or first available split)
        if split in dataset_dict:
            dataset = dataset_dict[split]
        else:
            available_splits = list(dataset_dict.keys())
            if available_splits:
                dataset = dataset_dict[available_splits[0]]
                print(f"[INFO] Split '{split}' not found, using '{available_splits[0]}' instead")
            else:
                raise ValueError(f"No splits found in dataset")
    else:
        # Load from HuggingFace Hub
        dataset = load_dataset(dataset_source, split=split, cache_dir=cache_dir)
    
    # Validate required columns exist
    if prompt_column not in dataset.column_names:
        raise ValueError(
            f"Prompt column '{prompt_column}' not found in dataset.\n"
            f"Available columns: {dataset.column_names}"
        )
    
    if reference_column not in dataset.column_names:
        raise ValueError(
            f"Reference column '{reference_column}' not found in dataset.\n"
            f"Available columns: {dataset.column_names}"
        )
    
    # Map columns to SocioHack format (prompt + reference)
    def rename_columns(example: Dict[str, Any]) -> Dict[str, Any]:
        """Rename columns to SocioHack standard format."""
        import json
        result = {
            "prompt": str(example[prompt_column]),
            "reference": str(example[reference_column])
        }
        # Preserve environment field if it exists (used for task-specific context)
        if "env" in example:
            result["env"] = example["env"]
        if "actions" in example:
            result["actions_list"] = json.dumps(example["actions"], ensure_ascii=False) if isinstance(example["actions"], (list, dict)) else example["actions"]
        if "dynamics" in example:
            result["dynamics_list"] = json.dumps(example["dynamics"], ensure_ascii=False) if isinstance(example["dynamics"], (list, dict)) else example["dynamics"]
        if "reward_criteria_quantified" in example:
            result["reward_criteria_quantified"] = json.dumps(example["reward_criteria_quantified"], ensure_ascii=False) if isinstance(example["reward_criteria_quantified"], (list, dict)) else example["reward_criteria_quantified"]
        return result
    
    formatted_dataset = dataset.map(rename_columns)
    
    # Remove all columns except SocioHack required fields
    columns_to_keep = ["prompt", "reference"]
    if "env" in formatted_dataset.column_names:
        columns_to_keep.append("env")
    if "actions_list" in formatted_dataset.column_names:
        columns_to_keep.append("actions_list")
    if "dynamics_list" in formatted_dataset.column_names:
        columns_to_keep.append("dynamics_list")
    if "reward_criteria_quantified" in formatted_dataset.column_names:
        columns_to_keep.append("reward_criteria_quantified")
            
    columns_to_remove = [
        col for col in formatted_dataset.column_names 
        if col not in columns_to_keep
    ]
    formatted_dataset = formatted_dataset.remove_columns(columns_to_remove)

    # Save to disk if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        folder_name = dataset_name if dataset_name else "formatted_dataset"
        output_path = os.path.join(output_dir, folder_name)
        formatted_dataset.save_to_disk(output_path)
        print(f"[INFO] Formatted dataset saved to {output_path}")
        return output_path
    
    # Return in-memory dataset if no output_dir
    return formatted_dataset