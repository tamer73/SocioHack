#!/usr/bin/env python3
"""
Simple Configuration Loader for Shell Scripts

Lightweight utility for extracting configuration values from YAML files in shell scripts.
This tool bridges the gap between Python Hydra configs and shell scripting.

Usage:
    # Get a config value
    MODEL_NAME=$(python -m src.simple_config_loader \\
        --config-name config \\
        --key model.name \\
        --default "Qwen/Qwen3-4B-Instruct-2507")
    
    # Use in shell script
    BATCH_SIZE=$(python -m src.simple_config_loader \\
        --config-name config \\
        --key training.batch_size \\
        --default 4)

Features:
    - Supports nested keys using dot notation (e.g., "model.name")
    - Returns default value if key not found
    - Safe error handling for missing files or invalid YAML
"""

import os
import argparse
import yaml


def get_config_value(config_name: str, key: str, default: str = "") -> str:
    """
    Extract a single value from a YAML configuration file.
    
    Supports nested keys using dot notation. For example, "model.name" will
    navigate to config['model']['name'].

    Args:
        config_name (str): Name of config file (without .yaml extension)
        key (str): Config key in dot notation (e.g., "model.name", "training.batch_size")
        default (str): Default value if key not found or file missing. Default: ""
        
    Returns:
        str: Configuration value as string, or default if not found
    
    Example:
        >>> get_config_value("config", "model.name", "default-model")
        "Qwen/Qwen3-4B-Instruct-2507"
    """
    config_file = f"configs/{config_name}.yaml"
    
    # Return default if config file doesn't exist
    if not os.path.exists(config_file):
        return default
    
    try:
        # Load YAML configuration
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Navigate through nested dictionary using dot notation
        keys = key.split('.')
        value = config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                # Key path not found, return default
                return default
        
        # Convert value to string for shell script compatibility
        if value is not None:
            return str(value)
        else:
            return default
            
    except Exception as e:
        # Log error to stderr and return default
        print(f"Error reading config value: {e}", file=os.sys.stderr)
        return default

def main():
    """
    Command-line interface for config value extraction.
    
    Parses arguments and outputs the requested configuration value to stdout,
    making it easy to capture in shell scripts.
    """
    parser = argparse.ArgumentParser(
        description="Extract configuration values from YAML files for shell scripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Get model name
  MODEL=$(python -m src.simple_config_loader --key model.name)
  
  # Get with default value
  BATCH_SIZE=$(python -m src.simple_config_loader \\
      --key training.batch_size --default 4)
  
  # Use custom config file
  LR=$(python -m src.simple_config_loader \\
      --config-name experiment --key training.learning_rate)
        """
    )
    
    parser.add_argument(
        "--config-name",
        default="config",
        help="Config file name without .yaml extension (default: 'config')"
    )
    parser.add_argument(
        "--key",
        required=True,
        help="Config key in dot notation (e.g., 'model.name', 'training.batch_size')"
    )
    parser.add_argument(
        "--default",
        default="",
        help="Default value if key not found (default: empty string)"
    )
    
    args = parser.parse_args()
    
    # Extract and print value (stdout for shell capture)
    value = get_config_value(args.config_name, args.key, args.default)
    print(value)

if __name__ == "__main__":
    main() 