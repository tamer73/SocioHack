#!/usr/bin/env python3
"""
SocioHack Training Entry Point

Main training script for SocioHack training with Hydra configuration management.

This script orchestrates the complete training pipeline:
1. Environment setup and configuration loading
2. Weights & Biases initialization
3. Dataset loading with train/validation split
4. Reward function configuration (tag format, LLM judge, outcome scoring)
5. GRPO trainer initialization with LoRA adapters
6. Training execution with checkpoint management

Usage:
    # Basic training
    python -m src.train

    # Override config values
    python -m src.train training.batch_size=8 training.learning_rate=1e-5

    # Multi-GPU training
    python -m src.train gpu.training.gpu_ids=0,1,2,3 gpu.training.num_gpus=4

    # Custom dataset tags
    python -m src.train dataset.intermediate_tag=think dataset.final_tag=answer

Configuration:
    All configuration is managed through Hydra configs in configs/ directory.
    See configs/base.yaml for available parameters.
"""

import os, glob, json, re
from omegaconf import DictConfig
from .config_manager import (
    setup_environment, init_wandb, get_training_config, get_peft_config, save_config_to_yaml
)
import hydra
from .data_loader import load_sociohack_dataset
from .trainer import CustomGRPOTrainer
from .reward import reward_from_outcome, llm_judge_reward, set_loophole_tracker_task_name, set_run_name

# Enable transformers progress bars for better monitoring
try:
    from transformers.utils import logging as transformers_logging
    transformers_logging.enable_progress_bar()
except ImportError:
    pass


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(config: DictConfig):
    """
    Main SocioHack training function with Hydra configuration.
    
    This function executes the complete training pipeline:
    1. Setup: Environment configuration, wandb initialization
    2. Data: Load and split dataset
    3. Model: Configure GRPO trainer with LoRA adapters
    4. Rewards: Set up multi-component reward functions (tag format, LLM judge, outcome)
    5. Train: Execute GRPO training with checkpoint management
    6. Save: Persist final model
    
    Args:
        config (DictConfig): Hydra configuration loaded from config/config.yaml
    
    Command-Line Overrides:
        You can override any config value from command line:
        
        # Training hyperparameters
        python -m src.train training.batch_size=8 training.learning_rate=1e-5
        
        # Multi-GPU setup
        python -m src.train gpu.training.gpu_ids=0,1,2,3 gpu.training.num_gpus=4
        
        # Custom tags
        python -m src.train dataset.intermediate_tag=think dataset.final_tag=answer
        
        # Resume from checkpoint
        python -m src.train model.resume_from_checkpoint=true \\
                          model.adapter_dir=/path/to/checkpoint
    
    Reward Function Pipeline:
        The trainer uses two reward functions in dependency chain:
        1. llm_judge_reward: LLM-based quality judgment (constraint checks + scoring)
        2. reward_from_outcome: Outcome-based scoring (requires #1 > 0)
        
        Each reward can only be calculated if its prerequisites pass, creating
        a hierarchical quality filter.
    """
    
    # Step 1: Environment setup
    setup_environment()  # Configure NCCL, CUDA memory
    
    # Initialize loophole tracker for comprehensive loophole tracking.
    # By default the artifact filenames (loopholes_*, rollouts_*, llm_debug_*)
    # are derived purely from the dataset name, i.e. `loopholes_<dataset.name>.json`.
    # This is the name the evaluators expect, so prepare -> train -> eval line up
    # out of the box.
    #
    # For hyperparameter sweeps that run the SAME scenario multiple times (e.g.
    # defense KL/temp/sync ablations over a single scenario) and must not overwrite
    # each other, set SOCIOHACK_UNIQUE_ARTIFACTS=1 to also fold the project suffix
    # into the artifact filename. The dataset name is always kept as task_name
    # for metrics and metadata regardless.
    suffix = getattr(config.project, "suffix", None) if hasattr(config, "project") else None
    unique_artifacts = os.getenv("SOCIOHACK_UNIQUE_ARTIFACTS", "").lower() in ("1", "true", "yes")
    if suffix and unique_artifacts:
        # Sanitize: keep filename-safe chars only
        safe_suffix = re.sub(r"[^A-Za-z0-9_.\-]+", "_", str(suffix)).strip("_")
        run_name = f"{config.dataset.name}__{safe_suffix}" if safe_suffix else config.dataset.name
    else:
        run_name = config.dataset.name
    set_run_name(run_name)
    set_loophole_tracker_task_name(config.dataset.name)
    print(f"[Artifacts] run_name = {run_name}")
    print(f"[Artifacts] loopholes file = loopholes_{run_name}.json")
    
    # Display training configuration summary
    print("\n" + "="*70)
    print("SocioHack Training Configuration")
    print("="*70)
    print(f"Project:         {config.project.suffix}")
    print(f"Dataset:         {config.dataset.name}")
    print(f"Model:           {config.model.name}")
    print(f"Batch size:      {config.training.batch_size}")
    print(f"Learning rate:   {config.training.learning_rate}")
    print(f"Epochs:          {config.training.num_train_epochs}")
    print(f"Iterations:      {config.training.num_iterations}")
    print(f"Generations:     {config.training.num_generations}")
    print(f"Intermediate tag: {config.dataset.intermediate_tag}")
    print(f"Final tag:       {config.dataset.final_tag}")
    print("="*70 + "\n")
    
    # Step 2: Initialize experiment tracking
    init_wandb(config)  # Weights & Biases tracking
    save_config_to_yaml(config)  # Save config for reproducibility
    
    # Step 3: Load and prepare dataset
    print("[INFO] Loading dataset...")
    train_dataset, eval_dataset = load_sociohack_dataset(
        config=config,
        dataset_name=config.dataset.name,
        val_size=config.dataset.validation_size
    )
    print(f"[INFO] Train samples: {len(train_dataset)}")
    if eval_dataset:
        print(f"[INFO] Validation samples: {len(eval_dataset)}")
    
    # Step 4: Configure training and model
    print("[INFO] Configuring GRPO training...")
    training_config = get_training_config(config)
    peft_config = get_peft_config(config)
    
    # Step 5: Configure hierarchical reward functions
    print("[INFO] Setting up reward functions...")
    
    # Create reward function wrappers with config parameters
    # Reward pipeline: llm_judge -> outcome
    # outcome reward depends on llm_judge passing (dependency chain)
    
    def llm_judge_reward_wrapper(completions, **kwargs):
        """Wrapper for LLM-based quality judgment reward."""
        return llm_judge_reward(
            completions, 
            intermediate_tag=config.dataset.intermediate_tag, 
            final_tag=config.dataset.final_tag,
            task_name=config.dataset.name,
            gemini_prompt=config.gemini.custom_prompt,
            **kwargs
        )
    llm_judge_reward_wrapper.__name__ = "LLM Judge Reward"
    
    def reward_from_outcome_wrapper(completions, **kwargs):
        """Wrapper for outcome-based scoring reward."""
        return reward_from_outcome(
            completions, 
            intermediate_tag=config.dataset.intermediate_tag, 
            final_tag=config.dataset.final_tag,
            task_name=config.dataset.name,
            **kwargs
        )
    reward_from_outcome_wrapper.__name__ = "Reward from Outcome"
    
    # Assemble reward function list (order is critical for dependency chain)
    reward_funcs = [
        llm_judge_reward_wrapper,
        reward_from_outcome_wrapper
    ]
    
    print(f"[INFO] Configured {len(reward_funcs)} reward functions")
    print(f"[INFO] Reward weights: {config.reward.llm_judge_reward_weight}, "
          f"{config.reward.reward_from_outcome_weight}")
    
    # Step 6: Initialize custom GRPO trainer
    print("[INFO] Initializing GRPO trainer...")
    # Defense-experiment knobs (default 0 = disabled, reduces to vanilla GRPO)
    entropy_coef = float(getattr(config.training, "entropy_coef", 0.0) or 0.0)
    lora_reset_every = int(getattr(config.training, "lora_reset_every", 0) or 0)

    trainer = CustomGRPOTrainer(
        model=config.model.name,
        args=training_config,
        peft_config=peft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=reward_funcs,
        entropy_coef=entropy_coef,
        custom_config={
            # Tag configuration for parsing completions
            "intermediate_tag": config.dataset.intermediate_tag,
            "final_tag": config.dataset.final_tag,
            "force_chat_template": config.dataset.force_chat_template,
            # Separate vLLM instance for reward calculation (not TRL's vLLM)
            "use_vllm": config.reward_vllm.use_vllm,
            "vllm_model_name": config.reward_vllm.model_name,
            "vllm_host": config.reward_vllm.host,
            "vllm_port": config.reward_vllm.port,
            # Scoring mode: quantified (from reward_criteria_quantified field) vs legacy (from prompt text)
            "use_quantified_scoring": config.reward.get("use_quantified_scoring", False),
            "gemini_backend": config.gemini.backend,
        }
    )

    # Attach periodic LoRA reset callback for the lora-reset defense experiments.
    # Zero / disabled by default so normal training is unaffected.
    if lora_reset_every > 0:
        from .trainer import LoRAResetCallback
        trainer.add_callback(LoRAResetCallback(reset_every_steps=lora_reset_every))
        print(f"[INFO] LoRAResetCallback enabled: reset every {lora_reset_every} optimizer steps")
    if entropy_coef > 0.0:
        print(f"[INFO] Entropy regularization enabled: coef={entropy_coef}")
    
    # Determine checkpoint path for resuming
    resume_from_checkpoint = None
    if config.model.resume_from_checkpoint and config.model.adapter_dir:
        adapter_dir = config.model.adapter_dir
        # Find the latest checkpoint-* subdirectory within the adapter dir
        ckpt_dirs = sorted(
            glob.glob(os.path.join(adapter_dir, "checkpoint-*")),
            key=lambda d: int(os.path.basename(d).split("-")[-1]),
        )
        if ckpt_dirs:
            import math
            latest_ckpt = ckpt_dirs[-1]
            ckpt_step = int(os.path.basename(latest_ckpt).split("-")[-1])
            adapter_file = os.path.join(latest_ckpt, "adapter_model.safetensors")

            # Calculate total expected steps (GRPO: batch_size is num completions, not prompts)
            num_gens = config.training.num_generations
            prompts_per_step = max(1, config.training.batch_size // num_gens)
            steps_per_epoch = math.ceil(len(trainer.train_dataset) / prompts_per_step)
            total_steps = steps_per_epoch * config.training.num_train_epochs
            remaining_steps = total_steps - ckpt_step

            print(f"[RESUME] Found checkpoint at step {ckpt_step}/{total_steps} "
                  f"(epoch {ckpt_step / steps_per_epoch:.2f}/{config.training.num_train_epochs})")

            if remaining_steps <= 0:
                print(f"[RESUME] Training already complete ({ckpt_step}/{total_steps} steps). Skipping.")
            elif os.path.exists(adapter_file):
                # Load LoRA adapter weights directly (bypasses DeepSpeed checkpoint loading)
                print(f"[RESUME] Loading LoRA adapter weights from {latest_ckpt} ...")
                from safetensors.torch import load_file
                adapter_state = load_file(adapter_file)
                incompatible = trainer.model.load_state_dict(adapter_state, strict=False)
                if incompatible.unexpected_keys:
                    print(f"[RESUME] Unexpected keys (ignored): {incompatible.unexpected_keys[:5]}...")
                # Limit training to remaining steps so we don't over-train
                trainer.args.max_steps = remaining_steps
                print(f"[RESUME] Adapter weights loaded. Will train {remaining_steps} more steps "
                      f"(max_steps={remaining_steps}).")
                print(f"[RESUME] Note: optimizer state not restored, LR schedule restarts.")
            else:
                print(f"[WARN] No adapter_model.safetensors in {latest_ckpt}, starting fresh.")
        else:
            print(f"[WARN] No checkpoint-* dirs found in {adapter_dir}, starting fresh.")
    if resume_from_checkpoint:
        print(f"[INFO] Resuming from checkpoint: {resume_from_checkpoint}")

    # Step 7: Execute training
    print("\n" + "="*70)
    print("Starting GRPO Training")
    print("="*70 + "\n")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Step 8: Save final model (only if save_model is enabled)
    if config.training.save_model:
        print("\n[INFO] Saving final model...")
        trainer.save_model()
        print("[INFO] Training completed successfully!")
    else:
        print("\n[INFO] Training completed successfully! (Model saving disabled)")

if __name__ == "__main__":
    main()  # Entry point for training