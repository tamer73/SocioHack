"""
Configuration Management Module for SocioHack Training

This module provides comprehensive configuration management for SocioHack
training, including environment setup, precision detection, output directory management, and 
Weights & Biases (wandb) initialization.

It is a SocioHack-specific build for studying LLM reward-hacking behaviour.

Key Features:
- Automatic precision detection (BF16/FP16) based on GPU capabilities
- Distributed training environment configuration
- GRPO (Generalized Reward Policy Optimization) configuration
- LoRA (Low-Rank Adaptation) configuration management
- Wandb experiment tracking initialization
"""

import os
import torch
import wandb
import yaml
from datetime import datetime
from trl import GRPOConfig
from peft import LoraConfig
from omegaconf import DictConfig


def setup_environment():
    """
    Configure environment variables for distributed training and GPU memory management.
    
    Sets up critical NCCL (NVIDIA Collective Communications Library) parameters for
    multi-GPU training and configures PyTorch CUDA memory allocation strategies.
    
    Environment Variables Set:
        - NCCL_DEBUG: Set to ERROR to reduce verbosity
        - NCCL_TIMEOUT: 3600 seconds for long-running operations
        - PYTORCH_CUDA_ALLOC_CONF: Enable expandable memory segments
        - NCCL_P2P_DISABLE: Disable peer-to-peer GPU transfers
        - NCCL_IB_DISABLE: Keep InfiniBand enabled (set to 0)
        - NCCL_SOCKET_IFNAME: Exclude docker and loopback interfaces
    
    Also clears GPU cache to ensure clean memory state before training.
    """
    os.environ.update({
        "NCCL_TIMEOUT": "3600",
        "NCCL_SOCKET_TIMEOUT": "3600",
        "NCCL_IB_TIMEOUT": "3600",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "NCCL_IB_DISABLE": "0",
        "NCCL_SOCKET_IFNAME": "^docker0,lo",
    })
    # Use setdefault so batch scripts can override for B200 compatibility
    os.environ.setdefault("NCCL_DEBUG", "ERROR")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_CUMEM_ENABLE", "1")
    torch.cuda.empty_cache()

def get_precision_config():
    """
    Automatically detect optimal precision settings based on GPU compute capability.
    
    Modern GPUs (compute capability >= 8.0, e.g., A100, H100) support BF16 (bfloat16)
    which provides better numerical stability for large language models. Older GPUs
    fall back to FP16 (float16).
    
    Returns:
        tuple: (precision_type, torch_dtype, vllm_dtype)
            - precision_type (str): "bf16" or "fp16" for training config
            - torch_dtype (torch.dtype): torch.bfloat16 or torch.float16
            - vllm_dtype (str): "bfloat16" or "half" for vLLM inference
    
    Note:
        If CUDA is not available, defaults to FP16 settings.
    """
    if not torch.cuda.is_available():
        return "fp16", torch.float16, "half"
    
    gpu_props = torch.cuda.get_device_properties(0)
    compute_capability = float(f"{gpu_props.major}.{gpu_props.minor}")
    
    return ("bf16", torch.bfloat16, "bfloat16") if compute_capability >= 8.0 else ("fp16", torch.float16, "half")

def get_output_dir(config: DictConfig, suffix=""):
    """
    Generate output directory path for model checkpoints and training artifacts.
    
    Directory naming strategy:
    - If resuming from checkpoint: Use existing adapter_dir
    - For new training runs: Create timestamped directory in format:
      {model_name}_{YYYYMMDD_HHMM}_{suffix}
    
    Args:
        config (DictConfig): Hydra configuration object containing model and project settings
        suffix (str, optional): Additional suffix for directory name. Defaults to "".
    
    Returns:
        str: Full path to output directory
    
    Example:
        >>> get_output_dir(config, suffix="experiment1")
        "/path/to/saves/Qwen3-4B_20250108_1430_experiment1"
    """
    if config.model.resume_from_checkpoint and config.model.adapter_dir:
        return config.model.adapter_dir
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    model_name_short = config.model.name.split('/')[-1]
    base_dir = os.path.expanduser(os.path.expandvars(str(config.project.save_base_path)))
    dir_name = f"{model_name_short}_{timestamp}"
    if suffix:
        dir_name += f"_{suffix}"
    return os.path.join(base_dir, dir_name)

def init_wandb(config: DictConfig):
    """
    Initialize Weights & Biases (wandb) experiment tracking with comprehensive configuration.
    
    Creates a wandb run with a descriptive name and logs all relevant hyperparameters including:
    - Model configuration (name, checkpoints, adapters)
    - Training hyperparameters (batch size, learning rate, epochs)
    - GRPO-specific settings (beta, temperature, reward weights)
    - LoRA configuration (rank, alpha, dropout, target modules)
    - vLLM inference settings (if enabled)
    - Reward function configurations (Gemini API, reward vLLM)
    - Dataset information
    
    Args:
        config (DictConfig): Hydra configuration containing all training parameters
    
    Run Naming Convention:
        - New runs: {model_short}_{timestamp}
        - Resumed runs: {model_short}_r{checkpoint_step}_{timestamp}
        - With suffix: {suffix}_{model_short}_{timestamp}
    
    Note:
        Converts OmegaConf ListConfig objects to Python lists for JSON serialization.
        Returns early if config.project.use_wandb is False.
    """
    # Global WandB switch - skip initialization if disabled
    use_wandb = getattr(config.project, 'use_wandb', True)
    if not use_wandb:
        print("[INFO] WandB disabled via config.project.use_wandb")
        return
    
    model_short = config.model.name.split('/')[-1]
    timestamp = datetime.now().strftime('%m%d_%H%M')
    
    if config.model.resume_from_checkpoint:
        run_name = f"{model_short}_r{config.model.latest_checkpoint_step}_{timestamp}"
    else:
        run_name = f"{model_short}_{timestamp}"
    
    if config.project.suffix:
        run_name = f"{config.project.suffix}_{run_name}"
    
    precision_type, torch_dtype, vllm_dtype = get_precision_config()
    
    # Convert ListConfig objects to regular Python lists for JSON serialization
    _tm = config.lora.target_modules if hasattr(config.lora, 'target_modules') else []
    target_modules = _tm if isinstance(_tm, str) else list(_tm)
    modules_to_save = list(config.lora.modules_to_save) if hasattr(config.lora, 'modules_to_save') else []
    
    wandb.init(
        project=config.project.wandb_project,
        name=run_name,
        config={
            "model": config.model.name,
            "output_dir": get_output_dir(config, suffix=config.project.suffix),
            "resume_from_checkpoint": config.model.resume_from_checkpoint,
            "latest_checkpoint_step": config.model.latest_checkpoint_step if config.model.resume_from_checkpoint else None,
            "adapter_dir": config.model.adapter_dir if config.model.resume_from_checkpoint else None,
            "batch_size": config.training.batch_size,
            "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
            "epochs": config.training.num_train_epochs,
            "num_iterations": config.training.num_iterations,
            "num_generations": config.training.num_generations,
            "beta": config.training.beta,
            "logging_steps": config.training.logging_steps,
            "save_steps": config.training.save_steps,
            "save_total_limit": config.training.save_total_limit,
            "scale_rewards": config.training.scale_rewards,
            "epsilon": config.training.epsilon,
            "epsilon_high": config.training.epsilon_high,
            "learning_rate": config.training.learning_rate,
            "temperature": config.training.temperature,
            "top_p": getattr(config.training, "top_p", 0.8),
            "top_k": getattr(config.training, "top_k", 20),
            "min_p": getattr(config.training, "min_p", 0.0),
            "presence_penalty": getattr(config.training, "presence_penalty", 1.5),
            "max_completion_length": config.training.max_completion_length,
            "sync_ref_model": config.training.sync_ref_model,
            "ref_model_mixup_alpha": config.training.ref_model_mixup_alpha,
            "ref_model_sync_steps": config.training.ref_model_sync_steps,
            "use_vllm": config.vllm.use_vllm,
            "vllm_host": config.vllm.host if config.vllm.use_vllm else None,
            "vllm_port": config.vllm.port if config.vllm.use_vllm else None,
            "vllm_gpu_memory_utilization": config.vllm.gpu_memory_utilization,
            "vllm_request_timeout": config.vllm.request_timeout,
            "vllm_mode": config.vllm.mode,
            "vllm_structured_outputs_regex": config.vllm.guided_decoding_regex,
            "vllm_tensor_parallel_size": config.vllm.tensor_parallel_size,
            # Reward VLLM configuration (separate from TRL VLLM)
            "reward_vllm_use_vllm": config.reward_vllm.use_vllm if hasattr(config, 'reward_vllm') else False,
            "reward_vllm_host": config.reward_vllm.host if hasattr(config, 'reward_vllm') and config.reward_vllm.use_vllm else None,
            "reward_vllm_port": config.reward_vllm.port if hasattr(config, 'reward_vllm') and config.reward_vllm.use_vllm else None,
            "reward_vllm_model_name": config.reward_vllm.model_name if hasattr(config, 'reward_vllm') else None,
            # Gemini API configuration. Never log the key itself to WandB.
            "gemini_api_key_set": bool(getattr(config.gemini, "api_key", None) or os.getenv("GEMINI_API_KEY")) if hasattr(config, 'gemini') else False,
            "gemini_model": config.gemini.model if hasattr(config, 'gemini') else None,
            "gemini_backend": config.gemini.backend if hasattr(config, 'gemini') and hasattr(config.gemini, 'backend') else None,
            "gemini_custom_prompt": config.gemini.custom_prompt if hasattr(config, 'gemini') else None,
            "gemini_timeout": config.gemini.timeout if hasattr(config, 'gemini') else None,
            # Advanced GRPO features
            "loss_type": config.training.loss_type,
            "mask_truncated_completions": config.training.mask_truncated_completions,
            "top_entropy_quantile": config.training.top_entropy_quantile,
            "importance_sampling_level": config.training.importance_sampling_level,
            "use_transformers_paged": config.training.use_transformers_paged,
            "enable_thinking": getattr(config.training, "enable_thinking", False),
            "reward_weights": {
                "llm_judge_reward": config.reward.llm_judge_reward_weight,
                "reward_from_outcome": config.reward.reward_from_outcome_weight
            },
            "peft_config": {
                "r": config.lora.r,
                "lora_alpha": config.lora.lora_alpha,
                "lora_dropout": config.lora.lora_dropout,
                "bias": config.lora.bias,
                "task_type": config.lora.task_type,
                "target_modules": target_modules,
                "modules_to_save": modules_to_save
            },
            "dataset": {
                "name": config.dataset.name,
                "subset": config.dataset.subset,
                "validation_size": config.dataset.validation_size
            }
        }
    )

def get_peft_config(config: DictConfig):
    """
    Create LoRA (Low-Rank Adaptation) configuration for parameter-efficient fine-tuning.
    
    LoRA reduces trainable parameters by injecting trainable rank decomposition matrices
    into model layers, making it memory-efficient for fine-tuning large language models.
    
    Args:
        config (DictConfig): Hydra configuration with LoRA settings under config.lora
    
    Returns:
        LoraConfig: PEFT LoRA configuration object
    
    Configuration Parameters:
        - r: LoRA rank (dimension of low-rank matrices)
        - lora_alpha: Scaling factor for LoRA weights
        - lora_dropout: Dropout probability for LoRA layers
        - bias: Bias training strategy ("none", "all", "lora_only")
        - task_type: Task type for LoRA (e.g., "CAUSAL_LM")
        - target_modules: List of module names to apply LoRA to
        - modules_to_save: Additional modules to train without LoRA
    """
    # Convert ListConfig objects to regular Python lists
    _tm = config.lora.target_modules if hasattr(config.lora, 'target_modules') else []
    target_modules = _tm if isinstance(_tm, str) else list(_tm)
    modules_to_save = list(config.lora.modules_to_save) if hasattr(config.lora, 'modules_to_save') else []
    
    return LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
        task_type=config.lora.task_type,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
    )

def get_training_config(config: DictConfig):
    """
    Create comprehensive GRPO (Generalized Reward Policy Optimization) training configuration.
    
    GRPO is a reinforcement learning algorithm that optimizes language models using reward
    signals from multiple reward functions. This function constructs the complete training
    configuration including optimization settings, generation parameters, and reward shaping.
    
    Args:
        config (DictConfig): Hydra configuration containing all training parameters
    
    Returns:
        GRPOConfig: TRL GRPO configuration object with all training parameters
    
    Key Configuration Groups:
        1. Output & Logging: output_dir, logging_steps, save_steps, report_to
        2. Optimization: learning_rate, gradient_accumulation, gradient_checkpointing
        3. Precision: Automatically detected (BF16/FP16) based on GPU
        4. GRPO Specifics: beta, epsilon, temperature, num_generations
        5. Rewards: reward_weights, scale_rewards, loss_type
        6. Reference Model: sync settings for KL divergence calculation
        7. vLLM Integration: Optional vLLM server for faster generation
        8. Advanced Features: importance sampling, entropy quantile, liger loss
    
    Note:
        Precision is automatically configured using get_precision_config()
    """
    precision_type, torch_dtype, vllm_dtype = get_precision_config()
    
    grpo_config = GRPOConfig(
        # Output settings
        output_dir=get_output_dir(config, suffix=config.project.suffix),
        log_completions=False,
        
        # Batch settings
        per_device_train_batch_size=config.training.batch_size,
        per_device_eval_batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        dataloader_drop_last=True,
        
        # Optimization settings
        learning_rate=config.training.learning_rate,
        gradient_checkpointing=True,
        **{precision_type: True},
        
        # Training loop settings
        num_train_epochs=config.training.num_train_epochs,
        num_iterations=config.training.num_iterations,
        beta=config.training.beta,
        
        # Progress bar and logging settings
        disable_tqdm=False,  # Enable training progress bar
        
        # Checkpoint and logging - use "no" strategy to disable saving when save_model is False
        save_strategy="steps" if config.training.save_model else "no",
        save_steps=config.training.save_steps if config.training.save_model else None,
        logging_steps=config.training.logging_steps,
        eval_strategy=config.training.evaluation_strategy,
        eval_steps=config.training.eval_steps,
        save_total_limit=config.training.save_total_limit if config.training.save_model else None,
        save_only_model=True,  # Only save model weights (LoRA adapter), skip optimizer/DS state
        report_to=["wandb"] if getattr(config.project, 'use_wandb', True) else [],
        
        # Model initialization
        model_init_kwargs={
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True
        },
        
        # GRPO specific settings
        num_generations=config.training.num_generations,
        temperature=config.training.temperature,
        # Qwen3.5 recommended sampling params (top-level GRPOConfig fields)
        top_p=float(getattr(config.training, "top_p", 0.8)),
        top_k=int(getattr(config.training, "top_k", 20)),
        min_p=float(getattr(config.training, "min_p", 0.0)),
        epsilon=config.training.epsilon,
        epsilon_high=config.training.epsilon_high,
        max_completion_length=config.training.max_completion_length,
        reward_weights=[
            float(config.reward.llm_judge_reward_weight),
            float(config.reward.reward_from_outcome_weight)
        ],
        scale_rewards=config.training.scale_rewards,
        
        # Extra kwargs passed directly to vLLM SamplingParams via generation_kwargs.update()
        generation_kwargs={
            "presence_penalty": float(getattr(config.training, "presence_penalty", 1.5)),
        },
        # Qwen3.5 non-thinking mode: trl 0.29.0 forwards chat_template_kwargs through apply_chat_template
        chat_template_kwargs={
            "enable_thinking": bool(getattr(config.training, "enable_thinking", False)),
        },
        
        # GRPO settings
        loss_type=config.training.loss_type,
        mask_truncated_completions=config.training.mask_truncated_completions,
        top_entropy_quantile=config.training.top_entropy_quantile,
        importance_sampling_level=config.training.importance_sampling_level,
        use_transformers_paged=config.training.use_transformers_paged,
        
        # Reference model settings
        sync_ref_model=config.training.sync_ref_model,
        ref_model_mixup_alpha=config.training.ref_model_mixup_alpha,
        ref_model_sync_steps=config.training.ref_model_sync_steps,
        
        # vLLM settings
        use_vllm=config.vllm.use_vllm,
        vllm_server_host=config.vllm.host if config.vllm.use_vllm else None,
        vllm_server_port=config.vllm.port if config.vllm.use_vllm else None,
        vllm_group_port=config.vllm.get("group_port", 51216) if config.vllm.use_vllm else 51216,
        vllm_server_timeout=config.vllm.request_timeout if config.vllm.use_vllm else None,
        vllm_mode=config.vllm.mode if config.vllm.use_vllm else "server",
        vllm_structured_outputs_regex=config.vllm.guided_decoding_regex,
        vllm_gpu_memory_utilization=config.vllm.gpu_memory_utilization,
        vllm_tensor_parallel_size=config.vllm.tensor_parallel_size,
        vllm_importance_sampling_mode="token_truncate",
        vllm_importance_sampling_cap=10.0,
    )
    
    return grpo_config

def get_dataset_path(config: DictConfig, dataset_name=None):
    """
    Construct full filesystem path to dataset directory.
    
    Datasets are expected to be stored in HuggingFace cache directory structure:
    {HF_HOME}/datasets/{dataset_name}/
    
    Args:
        config (DictConfig): Configuration containing dataset.hf_home and dataset.name
        dataset_name (str, optional): Override dataset name. Defaults to config.dataset.name
    
    Returns:
        str: Full path to dataset directory
    
    Example:
        >>> get_dataset_path(config)
        "/home/user/.cache/huggingface/datasets/my_dataset"
    """
    if dataset_name is None:
        dataset_name = config.dataset.name
    hf_home = os.path.expanduser(os.path.expandvars(str(config.dataset.hf_home)))
    return os.path.join(hf_home, "datasets", dataset_name)

def save_config_to_yaml(config: DictConfig):
    """
    Persist complete training configuration to YAML file for reproducibility.
    
    Saves all configuration parameters to config.yaml in the output directory, enabling:
    - Exact reproduction of training runs
    - Easy comparison between experiments
    - Configuration versioning with model checkpoints
    - Audit trail for hyperparameter changes
    
    Args:
        config (DictConfig): Complete Hydra configuration object
    
    Saved Configuration Sections:
        - model: Model name, checkpoint paths, output directory
        - dataset: Dataset name, subset, validation size
        - training: All training hyperparameters and GRPO settings
        - reference_model: Reference model synchronization settings
        - vllm: vLLM inference server configuration
        - reward_vllm: Separate vLLM instance for reward calculation
        - gemini: Google Gemini API configuration for LLM judging
        - peft: LoRA/PEFT configuration
        - environment: NCCL and CUDA environment variables
    
    Output:
        Creates {output_dir}/config.yaml with all settings
    
    Note:
        Converts OmegaConf types to standard Python types for YAML serialization
    """
    precision_type, torch_dtype, vllm_dtype = get_precision_config()
    
    # Convert ListConfig objects to regular Python lists
    _tm = config.lora.target_modules if hasattr(config.lora, 'target_modules') else []
    target_modules = _tm if isinstance(_tm, str) else list(_tm)
    modules_to_save = list(config.lora.modules_to_save) if hasattr(config.lora, 'modules_to_save') else []
    
    config_dict = {
        "model": {
            "name": config.model.name,
            "vllm_name": config.model.name_vllm,
            "output_dir": get_output_dir(config, suffix=config.project.suffix),
            "resume_from_checkpoint": config.model.resume_from_checkpoint,
            "latest_checkpoint_step": config.model.latest_checkpoint_step if config.model.resume_from_checkpoint else None,
            "adapter_dir": config.model.adapter_dir if config.model.resume_from_checkpoint else None,
            "suffix": config.project.suffix,
            "save_base_path": config.project.save_base_path
        },
        "dataset": {
            "name": config.dataset.name,
            "subset": config.dataset.subset,
            "hf_home": config.dataset.hf_home,
            "validation_size": config.dataset.validation_size
        },
        "training": {
            "wandb_project": config.project.wandb_project,
            "batch_size": config.training.batch_size,
            "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
            "num_train_epochs": config.training.num_train_epochs,
            "num_iterations": config.training.num_iterations,
            "beta": config.training.beta,
            "logging_steps": config.training.logging_steps,
            "save_steps": config.training.save_steps,
            "save_total_limit": config.training.save_total_limit,
            "num_generations": config.training.num_generations,
            "scale_rewards": config.training.scale_rewards,
            "epsilon": config.training.epsilon,
            "epsilon_high": config.training.epsilon_high,
            "max_completion_length": config.training.max_completion_length,
            "top_p": getattr(config.training, "top_p", 0.8),
            "top_k": getattr(config.training, "top_k", 20),
            "min_p": getattr(config.training, "min_p", 0.0),
            "presence_penalty": getattr(config.training, "presence_penalty", 1.5),
            "dataloader_drop_last": True,
            "gradient_checkpointing": True,
            "precision": precision_type,
            "save_strategy": "steps" if config.training.save_model else "no",
            "evaluation_strategy": config.training.evaluation_strategy,
            "eval_steps": config.training.eval_steps,
            "model_init_kwargs": {
                "torch_dtype": str(torch_dtype),
                "low_cpu_mem_usage": True
            },
            "log_completions": False,
            "report_to": ["wandb"] if getattr(config.project, 'use_wandb', True) else [],
            "reward_weights": {
                "llm_judge_reward": config.reward.llm_judge_reward_weight,
                "reward_from_outcome": config.reward.reward_from_outcome_weight,
                "weights_list": [
                    float(config.reward.llm_judge_reward_weight),
                    float(config.reward.reward_from_outcome_weight)
                ]
            },
            # New GRPO settings in updated TRL version
            "loss_type": config.training.loss_type,
            "mask_truncated_completions": config.training.mask_truncated_completions,
            "top_entropy_quantile": config.training.top_entropy_quantile,
            "importance_sampling_level": config.training.importance_sampling_level,
            "use_transformers_paged": config.training.use_transformers_paged,
            "enable_thinking": getattr(config.training, "enable_thinking", False),
        },
        "reference_model": {
            "sync_ref_model": config.training.sync_ref_model,
            "ref_model_mixup_alpha": config.training.ref_model_mixup_alpha,
            "ref_model_sync_steps": config.training.ref_model_sync_steps,
        },
        "vllm": {
            "enabled": config.vllm.use_vllm,
            "host": config.vllm.host if config.vllm.use_vllm else None,
            "port": config.vllm.port if config.vllm.use_vllm else None,
            "temperature": config.training.temperature,
            "gpu_memory_utilization": config.vllm.gpu_memory_utilization,
            "request_timeout": config.vllm.request_timeout,
            # New vLLM settings in updated TRL version
            "mode": config.vllm.mode,
            "guided_decoding_regex": config.vllm.guided_decoding_regex,
            "tensor_parallel_size": config.vllm.tensor_parallel_size,
            "model_name": config.vllm.model_name if hasattr(config.vllm, 'model_name') else None
        },
        # Reward VLLM configuration (completely separate from TRL VLLM)
        "reward_vllm": {
            "enabled": config.reward_vllm.use_vllm if hasattr(config, 'reward_vllm') else False,
            "host": config.reward_vllm.host if hasattr(config, 'reward_vllm') and config.reward_vllm.use_vllm else None,
            "port": config.reward_vllm.port if hasattr(config, 'reward_vllm') and config.reward_vllm.use_vllm else None,
            "model_name": config.reward_vllm.model_name if hasattr(config, 'reward_vllm') else None,
            "request_timeout": config.reward_vllm.request_timeout if hasattr(config, 'reward_vllm') else None
        },
        "gemini": {
            "api_key_set": bool(getattr(config.gemini, "api_key", None) or os.getenv("GEMINI_API_KEY")) if hasattr(config, 'gemini') else False,
            "model": config.gemini.model if hasattr(config, 'gemini') else None,
            "custom_prompt": config.gemini.custom_prompt if hasattr(config, 'gemini') else None,
            "timeout": config.gemini.timeout if hasattr(config, 'gemini') else None
        },
        "peft": {
            "r": config.lora.r,
            "lora_alpha": config.lora.lora_alpha,
            "lora_dropout": config.lora.lora_dropout,
            "bias": config.lora.bias,
            "task_type": config.lora.task_type,
            "target_modules": target_modules,
            "modules_to_save": modules_to_save
        },
        "environment": {
            "nccl_debug": "ERROR",
            "nccl_timeout": "3600",
            "nccl_socket_timeout": "3600",
            "nccl_ib_timeout": "3600",
            "pytorch_cuda_alloc_conf": "expandable_segments:True",
            "nccl_p2p_disable": "1",
            "nccl_ib_disable": "0",
            "nccl_socket_ifname": "^docker0,lo"
        }
    }
    
    output_dir = get_output_dir(config, suffix=config.project.suffix)
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False) 