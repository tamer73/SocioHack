"""
Utility Functions for SocioHack Training

Collection of helper functions for:
- Text parsing and tag extraction
- Logging utilities for Weights & Biases
- Configuration management

These utilities support the main training pipeline and reward functions.
"""

import re
import wandb
from omegaconf import DictConfig

# Default XML-style tags for structured completions
DEFAULT_INTERMEDIATE_TAG = "think"  # For reasoning/intermediate steps
DEFAULT_FINAL_TAG = "answer"  # For final answer/outcome


def get_tags_from_config(config: DictConfig = None):
    """
    Extract tag names from configuration with fallback to defaults.
    
    Tags are used to parse structured completions in XML-like format:
    <intermediate_tag>reasoning</intermediate_tag>
    <final_tag>answer</final_tag>
    
    Args:
        config (DictConfig, optional): Hydra configuration object. If None, uses defaults.
        
    Returns:
        tuple: (intermediate_tag, final_tag)
            - intermediate_tag (str): Tag for reasoning/intermediate steps
            - final_tag (str): Tag for final answer
    
    Example:
        >>> intermediate, final = get_tags_from_config(config)
        >>> print(f"Using tags: {intermediate}, {final}")
        Using tags: think, answer
    """
    if config is None:
        return DEFAULT_INTERMEDIATE_TAG, DEFAULT_FINAL_TAG
    
    intermediate_tag = getattr(config.dataset, 'intermediate_tag', DEFAULT_INTERMEDIATE_TAG)
    final_tag = getattr(config.dataset, 'final_tag', DEFAULT_FINAL_TAG)
    
    return intermediate_tag, final_tag

def extract_content(text: str, tag: str = None) -> str:
    """
    Extract content between XML-style tags from text or conversational messages.
    
    Supports both plain text and conversational format (list of message dicts).
    For conversational format, extracts content from the assistant's message.
    
    Args:
        text (str or list): Text string or list of message dicts with 'role' and 'content'
        tag (str, optional): Tag name to extract (without angle brackets).
                           Defaults to DEFAULT_FINAL_TAG ("answer")
    
    Returns:
        str: Content between <tag>...</tag>, or empty string if not found
    
    Example:
        >>> text = "<think>Let me reason...</think><answer>42</answer>"
        >>> extract_content(text, "answer")
        "42"
        
        >>> messages = [{"role": "assistant", "content": "<answer>Hello</answer>"}]
        >>> extract_content(messages, "answer")
        "Hello"
    """
    # Handle conversational format (list of message dicts)
    if isinstance(text, list) and len(text) > 0 and isinstance(text[0], dict):
        # Find assistant's message
        for message in text:
            if message.get("role") == "assistant":
                text = message.get("content", "")
                break
        else:
            # No assistant message found in list
            return ""
    
    # Ensure we're working with a string
    if not isinstance(text, str):
        text = str(text)
    
    # Use default tag if none specified
    if tag is None:
        tag = DEFAULT_FINAL_TAG
    
    # Extract content between tags using regex
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""

def extract_all_content(text: str, tag: str = None) -> list:
    """
    Extract all occurrences of content between specified tags.
    
    Unlike extract_content which returns only the first match, this function
    finds all instances of the tag in the text and returns them as a list.
    
    Args:
        text (str or list): Text string or list of message dicts
        tag (str, optional): Tag name to extract. Defaults to DEFAULT_INTERMEDIATE_TAG
    
    Returns:
        list: List of all content strings found between <tag>...</tag> pairs.
             Returns empty list if no matches found.
    
    Example:
        >>> text = "<think>Step 1</think><think>Step 2</think><answer>Done</answer>"
        >>> extract_all_content(text, "think")
        ["Step 1", "Step 2"]
    """
    # Handle conversational format (list of messages)
    if isinstance(text, list) and len(text) > 0 and isinstance(text[0], dict):
        # Extract content from assistant message
        for message in text:
            if message.get("role") == "assistant":
                text = message.get("content", "")
                break
        else:
            # No assistant message found
            return []
    
    # Ensure we have a string
    if not isinstance(text, str):
        text = str(text)
    
    if tag is None:
        tag = DEFAULT_INTERMEDIATE_TAG
    pattern = f"<{tag}>(.*?)</{tag}>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]

def safe_wandb_log(data, step=None):
    """
    Safely log metrics to Weights & Biases with error handling and key sanitization.
    
    This function provides a robust wrapper around wandb.log with:
    - Automatic key sanitization (fixes "metrics/" prefix issues)
    - Step inference from wandb history if not provided
    - Graceful handling of non-initialized wandb runs
    - Protection against logging errors
    
    Args:
        data (dict): Dictionary of metrics to log (key: metric_name, value: metric_value)
        step (int, optional): Training step for x-axis. If None, infers from wandb history
    
    Key Sanitization:
        - Converts "metrics/text_value" → "text_metrics/text_value"
        - This prevents wandb errors when logging string values to metric keys
    
    Example:
        >>> safe_wandb_log({"train/loss": 0.5, "train/accuracy": 0.95}, step=100)
        >>> safe_wandb_log({"metrics/status": "completed"}, step=200)  # Auto-fixes to text_metrics/
    
    Note:
        Does nothing if wandb.run is None (not initialized)
    """
    # Skip if wandb not initialized
    if wandb.run is None:
        return
    
    # Sanitize keys to prevent wandb errors
    fixed_data = {}
    for key, value in data.items():
        if isinstance(key, str) and key.startswith("metrics/") and isinstance(value, str):
            # Fix metrics/ prefix for string values (wandb expects numerical metrics)
            fixed_key = key.replace("metrics/", "text_metrics/")
            fixed_data[fixed_key] = value
        else:
            fixed_data[key] = value
    
    # Log to wandb
    if step is not None:
        # Prevent logging to past steps (fixes "step < current step" warnings)
        if step < wandb.run.step:
            step = wandb.run.step
        wandb.log(fixed_data, step=step)
    else:
        wandb.log(fixed_data)