"""
Custom GRPO Trainer Module for SocioHack Training

This module extends TRL's GRPOTrainer with advanced features for SocioHack:

**CustomGRPOTrainer Extensions:**
- LoRA-specific reference model synchronization
- Comprehensive reward analysis and visualization
- Completion diversity tracking with Levenshtein distance
- Validation accuracy calculation
- Dynamic constraint injection into prompts
- Enhanced wandb logging with extended metrics
- Perplexity-based completion quality assessment

**SyncRefLoraModelCallback:**
Custom callback for LoRA adapter synchronization between policy and reference models.
The original TRL callback doesn't work properly with LoRA adapters, so this provides
a proper implementation using PEFT's get/set adapter state dict functions.

**Key Features:**
1. Reference Model Sync: Periodically sync LoRA adapters with ref model for KL calculation
2. Diversity Tracking: Monitor generation diversity to prevent mode collapse
3. Validation Metrics: Calculate edit distance similarity for validation
4. Constraint Injection: Dynamically add discovered constraints to prompts
5. Extended Logging: Track perplexity, lengths, raw points, and judge reasons
"""

from trl import GRPOTrainer
from transformers import TrainerCallback
import math
import re
import torch
import wandb
from torch import nn
from accelerate.utils import gather_object
from trl.data_utils import is_conversational, maybe_apply_chat_template
from .reward import get_loophole_tracker, get_cached_outcome_scores, get_cached_rollout_constraints, _log_rollouts_to_csv
from typing import Union, Any, List, Optional
from .utils import safe_wandb_log
import numpy as np
import pandas as pd
import random
import os
import json
from datetime import datetime
from collections import deque


class LoRAResetCallback(TrainerCallback):
    """Periodically reset LoRA adapter weights to their initialization.

    This is a PEFT-compatible analog of TRL's `sync_ref_model`: instead of
    updating a reference model, we discretely zero out accumulated adapter
    drift every N optimizer steps. Standard LoRA has `lora_A` kaiming-init
    and `lora_B` zero-init, so resetting both returns the effective adapter
    output to zero (identical to the base model output), and subsequent
    training re-learns from scratch.

    We also zero the optimizer state for the reset parameters so Adam's
    momentum terms don't immediately push them back toward the old weights.

    Used as a reward-hacking defense: prevents the policy from accumulating
    drift that encodes a discovered exploit across many training steps.
    """

    def __init__(self, reset_every_steps: int):
        self.reset_every_steps = int(reset_every_steps)

    def on_step_end(self, args, state, control, **kwargs):
        if self.reset_every_steps <= 0:
            return
        if state.global_step <= 0:
            return
        if state.global_step % self.reset_every_steps != 0:
            return

        model = kwargs.get("model")
        optimizer = kwargs.get("optimizer")
        if model is None:
            return

        reset_params = []
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "lora_A" in name:
                    nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                    reset_params.append(param)
                elif "lora_B" in name:
                    param.zero_()
                    reset_params.append(param)

        if optimizer is not None:
            for p in reset_params:
                if p in optimizer.state:
                    optimizer.state[p] = {}

        print(
            f"[LoRAResetCallback] Reset {len(reset_params)} LoRA parameters "
            f"at global_step={state.global_step} (interval={self.reset_every_steps})"
        )


class CustomGRPOTrainer(GRPOTrainer):
    """
    Custom GRPO Trainer extended from TRL GRPOTrainer with enhanced functionality.
    
    This trainer extends the base GRPOTrainer with additional features including:
    - Advanced reward analysis and logging
    - Detailed reward breakdown visualization
    
    
    Args:
        *args: Arguments passed to the parent GRPOTrainer
        custom_config (dict, optional): Custom configuration that goes beyond the trl's GRPOConfig.
            Defaults to {"intermediate_tag": "think", "final_tag": "answer", "force_chat_template": False}
        **kwargs: Keyword arguments passed to the parent GRPOTrainer
    
    Attributes:
        custom_config (dict): Custom configuration
        intermediate_tag (str): Tag for intermediate outputs (default: "think")
        final_tag (str): Tag for final outputs (default: "answer")
    """
    def __init__(self, *args, custom_config=None, entropy_coef: float = 0.0, **kwargs):
        self.custom_config = custom_config or {}
        self.intermediate_tag = self.custom_config.get("intermediate_tag", "think")
        self.final_tag = self.custom_config.get("final_tag", "answer")
        self.force_chat_template = self.custom_config.get("force_chat_template", False)

        # Entropy regularization coefficient: loss <- loss - entropy_coef * mean_entropy
        # A positive value rewards higher policy entropy (discourages mode collapse).
        # 0.0 disables the bonus and falls back to vanilla GRPO loss.
        self.entropy_coef = float(entropy_coef or 0.0)
        self._cached_entropies = None

        super().__init__(*args, **kwargs)

        self.current_batch = None

        # Store reference data for reward functions
        self._current_reference_data = None

        # Extend the _logs dictionary with additional tracking fields
        self._logs.update({
            "completion_length": deque(maxlen=self.args.generation_batch_size),
            "reference": deque(maxlen=self.args.generation_batch_size),
            "judge_reasons": deque(maxlen=self.args.generation_batch_size),
            "raw_points": deque(maxlen=self.args.generation_batch_size),
        })

    def _get_per_token_logps_and_entropies(self, *args, **kwargs):
        """Passthrough that caches attached entropies for entropy regularization.

        The parent implementation computes entropies optionally (when
        ``compute_entropy=True``), uses them only for the ``top_entropy_quantile``
        mask, and then discards the reference. We capture the tensor here while
        it is still attached to the autograd graph so that ``_compute_loss``
        can add an entropy bonus to the final loss. If entropy regularization
        is disabled (``entropy_coef == 0``), the cache is never read.
        """
        result = super()._get_per_token_logps_and_entropies(*args, **kwargs)
        if kwargs.get("compute_entropy", False) and isinstance(result, tuple) and len(result) >= 2:
            self._cached_entropies = result[1]
        return result

    def _compute_loss(self, model, inputs):
        """Compute GRPO loss and optionally add an entropy regularization term.

        The entropy bonus is subtracted from the base loss so that maximizing
        policy entropy lowers the loss (standard entropy regularization for
        policy-gradient methods). Normalization matches the parent's
        gradient-accumulation scaling so the regularizer interacts correctly
        with accumulation.
        """
        self._cached_entropies = None
        loss = super()._compute_loss(model, inputs)

        if self.entropy_coef > 0.0 and self._cached_entropies is not None:
            completion_mask = inputs["completion_mask"]
            seq_sum = (self._cached_entropies * completion_mask).sum(-1)
            seq_denom = completion_mask.sum(-1).clamp(min=1.0)
            mean_entropy = (seq_sum / seq_denom).mean()

            mode = "train" if self.model.training else "eval"
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
            entropy_term = self.entropy_coef * mean_entropy / normalizer
            loss = loss - entropy_term

            # Log for wandb observability; safe against missing _metrics dict
            try:
                if hasattr(self, "_metrics") and mode in self._metrics:
                    self._metrics[mode].setdefault("entropy_bonus", []).append(
                        float(entropy_term.detach().item())
                    )
                    self._metrics[mode].setdefault("mean_entropy", []).append(
                        float(mean_entropy.detach().item())
                    )
            except Exception:
                pass

        # Release cached entropies from Python refs so the next step does not
        # hold onto the previous step's autograd graph.
        self._cached_entropies = None
        return loss
        
    def _log_detailed_rewards_analysis(self, rewards, rewards_per_func, advantages, mean_grouped_rewards, std_grouped_rewards):
        """
        Log detailed analysis of rewards and advantages for debugging and monitoring.
        
        This method provides comprehensive logging of reward components, including:
        - Per-function reward breakdown with weights applied
        - Total rewards and advantages for each generation
        - Group statistics (mean and standard deviation)
        
        The output is formatted as a table showing each generation's performance
        across all reward functions, making it easy to identify which components
        contribute most to the overall reward.
        
        Args:
            rewards (torch.Tensor): Total rewards for each sample
            rewards_per_func (torch.Tensor): Rewards per function for each sample
            advantages (torch.Tensor): Calculated advantages for each sample
            mean_grouped_rewards (torch.Tensor): Mean rewards per group
            std_grouped_rewards (torch.Tensor): Standard deviation of rewards per group
        """
        print("\n" + "-"*50)
        print(f"Step {self.state.global_step} - Rewards & Advantages")
        print("-"*50)
        
        reward_func_names = []
        for reward_func in self.reward_funcs:
            if isinstance(reward_func, nn.Module):
                reward_func_names.append(reward_func.config._name_or_path.split("/")[-1])
            else:
                reward_func_names.append(reward_func.__name__)
        
        rewards_by_group = rewards.view(-1, self.num_generations)
        rewards_per_func_by_group = rewards_per_func.view(-1, self.num_generations, len(self.reward_funcs))
        
        for group_idx in range(rewards_by_group.shape[0]):
            print(f"Group {group_idx+1} | Mean: {mean_grouped_rewards[group_idx*self.num_generations]:.4f} | Std: {std_grouped_rewards[group_idx*self.num_generations]:.4f}")
            print("Gen | " + " | ".join([f"{name}" for name in reward_func_names]) + " | Total | Adv")
            print("-" * 50)
            
            for gen_idx in range(self.num_generations):
                global_idx = group_idx * self.num_generations + gen_idx
                
                reward_components = []
                for func_idx in range(len(self.reward_funcs)):
                    component_value = rewards_per_func[global_idx, func_idx].item()
                    if torch.isnan(torch.tensor(component_value)):
                        reward_components.append("N/A")
                    else:
                        weight = self.reward_weights[func_idx].item()
                        weighted_value = component_value * weight
                        reward_components.append(f"{weighted_value:.2f}")
                
                components_str = " | ".join(reward_components)
                print(f"{gen_idx+1:3d} | {components_str} | {rewards[global_idx]:.2f} | {advantages[global_idx]:.2f}")
            
            print("")

    def _apply_custom_hooks(self, inputs, prompts, completions, completions_text, reference_text, prompts_text,
                           rewards, rewards_per_func, advantages, mean_grouped_rewards, std_grouped_rewards, mode):
        """
        Apply custom hooks for enhanced logging and analysis.
        
        This method orchestrates various custom analysis and logging functions:
        - Validation accuracy calculation (for eval mode with references)
        - Detailed reward analysis logging
        - Diversity analysis and tracking
        
        The hooks are applied based on the current training state and mode,
        providing comprehensive monitoring and debugging capabilities.
        
        Args:
            inputs (list): Input data
            prompts (list): Processed prompts
            completions (list): Generated completions
            completions_text (list): Decoded completion texts
            reference_text (list): Reference texts
            prompts_text (list): Decoded prompt texts
            rewards (torch.Tensor): Total rewards
            rewards_per_func (torch.Tensor): Rewards per function
            advantages (torch.Tensor): Calculated advantages
            mean_grouped_rewards (torch.Tensor): Mean rewards per group
            std_grouped_rewards (torch.Tensor): Standard deviation of rewards per group
            mode (str): Training mode ("train" or "eval")
        """
        
        # log detailed rewards analysis
        if self.accelerator.is_main_process and self.state.global_step % self.args.logging_steps == 0:
            self._log_detailed_rewards_analysis(rewards, rewards_per_func, advantages, mean_grouped_rewards, std_grouped_rewards)

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        
        """
        Override to add custom hooks and populate additional fields for wandb summary table.
        """
        # Inject dynamic constraints into prompts before any further processing
        try:
            tracker = get_loophole_tracker()
            dynamic_constraints = tracker.get_dynamic_constraints()
        except Exception:
            dynamic_constraints = []

        def _inject_constraints_into_prompt(prompt_obj):
            try:
                import re

                def _merge_into_constraints_block(text: str) -> str:
                    # Find <constraints>...</constraints> and merge dynamic constraints as bullet items
                    constraints_re = re.compile(r"(<constraints>)([\s\S]*?)(</constraints>)", re.IGNORECASE)
                    m = constraints_re.search(text)
                    dynamic_set = [c.strip() for c in dynamic_constraints if c and c.strip()]
                    if not dynamic_set:
                        return text
                    if m:
                        before_tag, body, after_tag = m.group(1), m.group(2), m.group(3)
                        # Extract existing bullet items from body
                        lines = body.splitlines()
                        indent = "  "
                        bullets: list[str] = []
                        for ln in lines:
                            stripped = ln.strip()
                            if stripped.startswith("- "):
                                bullets.append(stripped[2:].strip())
                            elif stripped.startswith("• "):
                                bullets.append(stripped[2:].strip())
                        # Deduplicate while preserving original order: existing first, then new
                        existing_lower = {b.lower() for b in bullets}
                        for c in dynamic_set:
                            if c.lower() not in existing_lower:
                                bullets.append(c)
                                existing_lower.add(c.lower())
                        # Rebuild body preserving a leading newline if had content
                        new_body_lines = []
                        # Preserve any non-bullet lines that were in the body (e.g., blank/intro)
                        for ln in lines:
                            if ln.strip().startswith("- ") or ln.strip().startswith("• "):
                                # skip; will be rebuilt
                                continue
                            if ln.strip() == "":
                                new_body_lines.append(ln)
                        # Rebuild bullets
                        for item in bullets:
                            new_body_lines.append(f"{indent}- {item}")
                        new_body = ("\n" if not body.startswith("\n") else "") + "\n".join(new_body_lines) + ("\n" if not body.endswith("\n") else "")
                        return text[:m.start()] + before_tag + new_body + after_tag + text[m.end():]
                    else:
                        # No constraints block found; append a new one at the end
                        bullet_text = "\n".join([f"  - {c}" for c in dynamic_set])
                        addition = f"\n\n<constraints>\n{bullet_text}\n</constraints>"
                        return text + addition

                if isinstance(prompt_obj, list):
                    # conversational: find the first user message and inject into its content
                    new_messages = []
                    injected = False
                    for msg in prompt_obj:
                        if not injected and isinstance(msg, dict) and msg.get("role") == "user":
                            content = msg.get("content", "")
                            content = _merge_into_constraints_block(content)
                            new_messages.append({"role": "user", "content": content})
                            injected = True
                        else:
                            new_messages.append(msg)
                    return new_messages
                elif isinstance(prompt_obj, str):
                    text = prompt_obj
                    text = _merge_into_constraints_block(text)
                    return text
                else:
                    return prompt_obj
            except Exception:
                return prompt_obj

        def _inject_length_constraint(prompt_obj):
            try:
                if not hasattr(self.args, "max_completion_length") or self.args.max_completion_length is None:
                    return prompt_obj
                
                # Use half of max_completion_length as the soft target for the plan text itself,
                # since tokens != words and we want to avoid mid-sentence truncation.
                target_tokens = self.args.max_completion_length
                limit_text = (
                    f" Your response must be no more than {target_tokens} tokens in total. "
                    "Write clearly and completely within this limit; do NOT truncate mid-sentence."
                )
                
                if isinstance(prompt_obj, str):
                    return prompt_obj + limit_text
                elif isinstance(prompt_obj, list):
                    new_messages = []
                    injected = False
                    for msg in prompt_obj:
                        if not injected and isinstance(msg, dict) and msg.get("role") == "user":
                            content = msg.get("content", "")
                            new_messages.append({"role": "user", "content": content + limit_text})
                            injected = True
                        else:
                            new_messages.append(msg)
                    return new_messages
                else:
                    return prompt_obj
            except Exception:
                return prompt_obj

        def _inject_strict_format(prompt_obj):
            import os
            if os.environ.get("SOCIOHACK_STRICT_FORMAT") != "1":
                return prompt_obj
            strict_text = (
                "\n\n<output_format>\n"
                "Your plan MUST end with a section titled '**Total Points Calculation:**' that does the following:\n"
                "1. For EACH variable in <reward_criteria>, state an explicit numeric assumption (e.g., 'claims_paid_out = $500,000').\n"
                "2. Show the arithmetic of applying the per-unit point value from <reward_criteria> to each assumption.\n"
                "3. Print the grand total on a line of the form 'total_points = <number>'. No symbolic unknowns (no $X, no [TBD]) — every variable must resolve to a concrete number.\n"
                "This section is mandatory regardless of report style; without it the plan is considered invalid.\n"
                "</output_format>"
            )
            try:
                if isinstance(prompt_obj, str):
                    return prompt_obj + strict_text
                if isinstance(prompt_obj, list):
                    new_messages = []
                    injected = False
                    for msg in prompt_obj:
                        if not injected and isinstance(msg, dict) and msg.get("role") == "user":
                            content = msg.get("content", "")
                            new_messages.append({"role": "user", "content": content + strict_text})
                            injected = True
                        else:
                            new_messages.append(msg)
                    return new_messages
            except Exception:
                return prompt_obj
            return prompt_obj

        try:
            inputs = [dict(example, prompt=_inject_constraints_into_prompt(example.get("prompt"))) for example in inputs]
            inputs = [dict(example, prompt=_inject_length_constraint(example.get("prompt"))) for example in inputs]
            inputs = [dict(example, prompt=_inject_strict_format(example.get("prompt"))) for example in inputs]
        except Exception:
            pass

        # Optionally force chat template by wrapping plain-text prompts into a single user turn
        try:
            if self.force_chat_template and not is_conversational(inputs[0]):
                wrapped_inputs = []
                for example in inputs:
                    new_example = dict(example)
                    prompt_value = new_example.get("prompt")
                    if isinstance(prompt_value, str):
                        new_example["prompt"] = [{"role": "user", "content": prompt_value}]
                    wrapped_inputs.append(new_example)
                inputs = wrapped_inputs
        except Exception:
            pass

        # Store original inputs for later reference extraction
        original_inputs = inputs.copy()

        # Extract and store reference data for reward functions
        self._current_reference_data = []
        for orig_example in original_inputs:
            if "reference" in orig_example:
                self._current_reference_data.append(orig_example["reference"])
            else:
                self._current_reference_data.append("")

        # Sanitize inputs to avoid invalid keys for maybe_apply_chat_template
        try:
            # Convert SocioHack format (prompt, reference) to TRL format (prompt, completion)
            sanitized_inputs = []
            for example in inputs:
                sanitized = {"prompt": example["prompt"]}
                # Convert 'reference' to 'completion' for TRL compatibility
                if "reference" in example:
                    reference_value = example["reference"]
                    # If we're using chat template and prompt is now messages, wrap reference too
                    if self.force_chat_template and is_conversational({"prompt": example["prompt"]}):
                        if isinstance(reference_value, str):
                            sanitized["completion"] = [{"role": "assistant", "content": reference_value}]
                        else:
                            sanitized["completion"] = reference_value
                    else:
                        sanitized["completion"] = reference_value
                # Keep other supported keys
                for key in ["image", "tools", "env", "actions_list", "dynamics_list", "reward_criteria_quantified"]:
                    if key in example:
                        sanitized[key] = example[key]
                sanitized_inputs.append(sanitized)
            inputs = sanitized_inputs
        except Exception:
            pass
        # Call the parent method to get the standard result and populate basic _logs
        result = super()._generate_and_score_completions(inputs)
        
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        
        # ensure prompts_text contains only strings
        for i, pt in enumerate(prompts_text):
            if not isinstance(pt, str):
                # Convert to string if possible
                if hasattr(pt, '__str__'):
                    prompts_text[i] = str(pt)
                else:
                    prompts_text[i] = ""
        
        # Use original inputs for reference extraction since we renamed 'reference' to 'completion'
        # Create sanitized original inputs for reference extraction
        sanitized_original_inputs = []
        for orig_example in original_inputs:
            sanitized_orig = {"prompt": orig_example["prompt"]}
            if "reference" in orig_example:
                sanitized_orig["reference"] = orig_example["reference"]
            # Keep other supported keys
            for key in ["image", "tools"]:
                if key in orig_example:
                    sanitized_orig[key] = orig_example[key]
            sanitized_original_inputs.append(sanitized_orig)
        
        # Use stored reference data if available (more reliable than maybe_apply_chat_template)
        if hasattr(self, '_current_reference_data') and self._current_reference_data:
            reference_text = self._current_reference_data
        else:
            # Fallback to maybe_apply_chat_template
            reference_text = [maybe_apply_chat_template(orig_example, self.processing_class).get("reference", "") for orig_example in sanitized_original_inputs]
        
        completion_ids = result["completion_ids"]
        
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text
        
        # Get rewards and advantages from the result (already calculated by parent method)
        # The parent method handles the reward calculation and _logs population
        
        # Apply our custom hooks for additional analysis
        # FIXED: Removed strict check on _logs["rewards"] to ensure logging always runs
        if hasattr(self, '_logs'):
            # Get the latest rewards and advantages from _logs
            batch_size = len(prompts)
            rewards_per_func = torch.zeros(batch_size, len(self.reward_funcs), device=device)
            
            # Try to get rewards from _logs if available
            if "rewards" in self._logs and len(self._logs["rewards"]) > 0:
                for i, name in enumerate(self.reward_func_names):
                    if name in self._logs["rewards"]:
                        reward_data = list(self._logs["rewards"][name])
                        if len(reward_data) >= batch_size:
                            latest_rewards = reward_data[-batch_size:]
                            rewards_per_func[:, i] = torch.tensor(latest_rewards, device=device)
            
            rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
            
            mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
            std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            advantages = rewards - mean_grouped_rewards
            if self.scale_rewards:
                advantages = advantages / (std_grouped_rewards + 1e-4)
            
            # Apply custom hooks
            self._apply_custom_hooks(
                inputs, prompts, completions, completions_text, reference_text, prompts_text,
                rewards, rewards_per_func, advantages, mean_grouped_rewards, std_grouped_rewards, mode
            )
            
            # Log rollouts to CSV
            if self.accelerator.is_main_process:
                try:
                    # Gather all completions across processes
                    all_completions_text = gather_object(completions_text)
                    
                    # Find indices for LLM judge and outcome rewards by name
                    llm_judge_idx = None
                    outcome_reward_idx = None
                    for idx, name in enumerate(self.reward_func_names):
                        if "LLM Judge" in name or "llm_judge" in name.lower():
                            llm_judge_idx = idx
                        elif "Outcome" in name or "outcome" in name.lower():
                            outcome_reward_idx = idx
                    
                    # Get reward values for each function
                    llm_judge_rewards = []
                    outcome_rewards = []
                    batch_size = len(all_completions_text)
                    
                    if llm_judge_idx is not None and rewards_per_func.shape[1] > llm_judge_idx:
                        llm_judge_rewards = rewards_per_func[:, llm_judge_idx].cpu().tolist()
                    else:
                        llm_judge_rewards = [0.0] * batch_size
                    
                    if outcome_reward_idx is not None and rewards_per_func.shape[1] > outcome_reward_idx:
                        outcome_rewards = rewards_per_func[:, outcome_reward_idx].cpu().tolist()
                    else:
                        outcome_rewards = [0.0] * batch_size
                    
                    # Get LLM judge reasonings
                    llm_judge_reasonings = []
                    if hasattr(self, '_judge_reasons') and self._judge_reasons:
                        # Get the judge reasons for this batch
                        if len(self._judge_reasons) >= batch_size:
                            llm_judge_reasonings = self._judge_reasons[:batch_size]
                            # Remove consumed reasons
                            self._judge_reasons = self._judge_reasons[batch_size:]
                        else:
                            llm_judge_reasonings = self._judge_reasons + [""] * (batch_size - len(self._judge_reasons))
                            self._judge_reasons = []
                    else:
                        llm_judge_reasonings = [""] * batch_size
                        
                    # Get total rewards and advantages
                    total_rewards_list = rewards.cpu().tolist()
                    advantages_list = advantages.cpu().tolist()

                    # Get all cached scoring data
                    cached_outcome_points, _, cached_simu_analysis, cached_triggered_events, cached_state_vars = get_cached_outcome_scores()
                    
                    # Helper to pad/slice lists to batch_size
                    def adjust_list_to_batch(lst, size, default_val=""):
                        if len(lst) >= size:
                            return lst[:size]
                        return lst + [default_val] * (size - len(lst))

                    simulator_analysis = adjust_list_to_batch(cached_simu_analysis, batch_size)
                    simulator_triggered_events = adjust_list_to_batch(cached_triggered_events, batch_size)
                    simulator_state_variables = adjust_list_to_batch(cached_state_vars, batch_size)
                    outcome_points_list = adjust_list_to_batch(cached_outcome_points, batch_size, None)
                    
                    # Ensure all lists have the same length for logging (min_len logic)
                    min_len = min(len(all_completions_text), len(llm_judge_rewards), len(outcome_rewards), 
                                 len(total_rewards_list), len(advantages_list), len(llm_judge_reasonings),
                                 len(simulator_analysis), len(simulator_triggered_events), len(simulator_state_variables),
                                 len(outcome_points_list))
                    
                    all_completions_text = all_completions_text[:min_len]
                    llm_judge_rewards = llm_judge_rewards[:min_len]
                    outcome_rewards = outcome_rewards[:min_len]
                    total_rewards_list = total_rewards_list[:min_len]
                    advantages_list = advantages_list[:min_len]
                    llm_judge_reasonings = llm_judge_reasonings[:min_len]
                    outcome_points_list = outcome_points_list[:min_len]
                    simulator_analysis = simulator_analysis[:min_len]
                    simulator_triggered_events = simulator_triggered_events[:min_len]
                    simulator_state_variables = simulator_state_variables[:min_len]
                    
                    # Get constraints from cache
                    cached_constraints = get_cached_rollout_constraints()
                    if len(cached_constraints) < min_len:
                        cached_constraints = cached_constraints + [""] * (min_len - len(cached_constraints))
                    elif len(cached_constraints) > min_len:
                        cached_constraints = cached_constraints[:min_len]
                    
                    # Get task name from high score tracker
                    task_name = "hack"
                    try:
                        tracker = get_loophole_tracker()
                        task_name = tracker.task_name
                    except:
                        pass
                    
                    # Gather prompts for CSV logging
                    all_prompts_text = gather_object(prompts_text)
                    if len(all_prompts_text) < min_len:
                        all_prompts_text = all_prompts_text + [""] * (min_len - len(all_prompts_text))
                    elif len(all_prompts_text) > min_len:
                        all_prompts_text = all_prompts_text[:min_len]
                    
                    # Log to CSV
                    _log_rollouts_to_csv(
                        task_name=task_name,
                        iteration_index=self.state.global_step,
                        prompts=all_prompts_text,
                        rollouts=all_completions_text,
                        simulator_analysis=simulator_analysis,
                        simulator_triggered_events=simulator_triggered_events,
                        simulator_state_variables=simulator_state_variables,
                        llm_judge_rewards=llm_judge_rewards,
                        llm_judge_reasonings=llm_judge_reasonings,
                        outcome_rewards=outcome_rewards,
                        total_rewards=total_rewards_list,
                        advantages=advantages_list,
                        outcome_points=outcome_points_list,
                        constraints=cached_constraints,
                    )
                    
                    # Log loophole tracking metrics to WandB
                    try:
                        loophole_tracker = get_loophole_tracker()
                        safe_wandb_log({
                            "loopholes/total_found": len(loophole_tracker.loopholes),
                            "loopholes/constraints_count": len(loophole_tracker.dynamic_constraints),
                            "loopholes/highest_score": loophole_tracker.highest_score,
                        }, step=self.state.global_step)
                        
                        # Log new loopholes discovered in this step
                        new_loopholes = [l for l in loophole_tracker.loopholes if l.get("step") == self.state.global_step]
                        if new_loopholes and wandb.run is not None:
                            try:
                                table_data = [[
                                    l["id"], 
                                    l["step"], 
                                    l["score"], 
                                    l["is_high_score_refresh"], 
                                    l["reward"],
                                    (l["generated_constraint"][:100] + "..." if len(l.get("generated_constraint", "")) > 100 else l.get("generated_constraint", ""))
                                ] for l in new_loopholes]
                                wandb.log({
                                    "loopholes/discoveries": wandb.Table(
                                        columns=["id", "step", "score", "is_high_score", "reward", "constraint"],
                                        data=table_data
                                    )
                                }, step=self.state.global_step)
                            except Exception:
                                pass  # WandB table logging is optional
                    except Exception as e:
                        print(f"[Trainer] Failed to log loophole metrics: {e}")
                except Exception as e:
                    print(f"[Trainer] Failed to log rollouts to CSV: {e}")
                    import traceback
                    print(traceback.format_exc())
        
        # Populate the new fields for wandb summary table
        self._populate_summary_fields(inputs, completions_text, reference_text)
        
        return result

    def _populate_summary_fields(self, inputs, completions_text, reference_text):
        """
        Calculate and populate extended metrics for wandb completions table.
        
        Computes:
        1. **Completion Length**: Token count of generated text
        2. **Raw Points**: Actual outcome points from LLM scoring
        3. **References**: Original reference answers for comparison
        4. **Judge Reasons**: Explanations from LLM judge reward
        """
        from accelerate.utils import gather_object
        
        # Gather data across all processes
        all_completions = gather_object(completions_text)
        all_references = gather_object(reference_text)
        
        completion_lengths = []
        references = []
        raw_points_list = []
        
        # Get cached outcome scores
        cached_points, _,_,_,_ = get_cached_outcome_scores()
        has_cached_points = len(cached_points) == len(all_completions)
        
        for i, completion in enumerate(all_completions):
            # Calculate completion length (token count)
            _tokenizer = getattr(self.processing_class, 'tokenizer', self.processing_class)
            completion_tokens = _tokenizer.encode(completion, add_special_tokens=False)
            completion_length = len(completion_tokens)
            
            # Extract raw points from cached LLM scoring results
            raw_points = 0.0
            if has_cached_points and cached_points[i] is not None:
                raw_points = float(cached_points[i])
            
            # Get reference text
            reference = ""
            if i < len(all_references):
                reference = str(all_references[i]) if all_references[i] else ""
            
            completion_lengths.append(completion_length)
            references.append(reference)
            raw_points_list.append(raw_points)
        
        # Extend the _logs with fields
        self._logs["completion_length"].extend(completion_lengths)
        self._logs["reference"].extend(references)
        self._logs["raw_points"].extend(raw_points_list)
        
        # Add judge reasons if available
        if hasattr(self, '_judge_reasons') and self._judge_reasons:
            batch_judge_reasons = self._judge_reasons[:len(completion_lengths)]
            self._logs["judge_reasons"].extend(batch_judge_reasons)
            self._judge_reasons = self._judge_reasons[len(completion_lengths):]
        else:
            self._logs["judge_reasons"].extend([""] * len(completion_lengths))

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Override the parent log method to include our custom fields in the wandb completions table.
        We call the parent's log method for most functionality, but override the wandb table creation.
        """
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        
        # Call parent's log method but temporarily disable wandb logging to avoid duplicate tables
        original_report_to = self.args.report_to
        self.args.report_to = []  # Temporarily disable wandb reporting
        
        super().log(logs, start_time)
        
        # Restore original report_to
        self.args.report_to = original_report_to
        
        self._metrics[mode].clear()

        # Handle wandb logging with our extended table
        if self.accelerator.is_main_process and self.log_completions:
            # Log to wandb with our extended table (only if wandb is enabled)
            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                import pandas as pd

                # Calculate raw points statistics for logging as curves
                if self._logs["raw_points"]:
                    # Filter out zero values for meaningful statistics (assuming 0 means no points extracted)
                    valid_raw_points = [points for points in self._logs["raw_points"] if points > 0 and not np.isnan(points)]
                    
                    if valid_raw_points:
                        # Add raw points metrics to the main logs for curve visualization
                        logs["raw_points/mean"] = np.mean(valid_raw_points)
                        logs["raw_points/median"] = np.median(valid_raw_points)
                        logs["raw_points/min"] = np.min(valid_raw_points)
                        logs["raw_points/max"] = np.max(valid_raw_points)
                        logs["raw_points/std"] = np.std(valid_raw_points)
                        logs["raw_points/count"] = len(valid_raw_points)
                
                # Display loophole tracker status
                loophole_tracker = get_loophole_tracker()
                loophole_tracker.print_status()

                # Create the extended table with all fields
                table = {
                    "step": [str(self.state.global_step)] * len(self._logs["prompt"]),
                    "prompt": list(self._logs["prompt"]),
                    "completion": list(self._logs["completion"]),
                    **{k: list(v) for k, v in self._logs["rewards"].items()},
                    "advantage": list(self._logs["advantages"]),
                    # Add extended tracking fields
                    "completion_length": list(self._logs["completion_length"]),
                    "reference": list(self._logs["reference"]),
                    "judge_reasons": list(self._logs["judge_reasons"]),
                    "raw_points": list(self._logs["raw_points"]),
                }

                if self._logs["image"]:
                    table["image"] = []
                    for img in self._logs["image"]:
                        if img is not None:
                            # Convert images to wandb Image objects for proper visualization
                            table["image"].append(wandb.Image(img))
                        else:
                            table["image"].append(None)

                df = pd.DataFrame(table)
                
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=["prompt"])
                
                # Log the extended completions table to wandb
                wandb.log({"completions": wandb.Table(dataframe=df)})
                
                # Also log the metrics that the parent would have logged
                wandb.log(logs)

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        """
        Calculate rewards with dependency chain and reference data propagation.
        
        This override of the base GRPOTrainer method implements two critical features:
        
        1. **Dependency Chain**: Pass llm_judge_rewards + reasons to reward_from_outcome.
           This enables hierarchical filtering where outcome reward depends on judge passing.
        
        2. **Reference Data Propagation**: Ensure 'reference' field is available to all
           reward functions, even though it was renamed to 'completion' for TRL compatibility.
        
        Args:
            inputs: Batch of input examples
            prompts: Processed prompts
            completions: Model-generated completions
            completion_ids_list: Token IDs for completions
        
        Returns:
            torch.Tensor: rewards_per_func of shape (batch_size, num_reward_funcs)
        
        Reward Function Flow:
            1. llm_judge_reward(completions) → rewards[0], reasons
            2. reward_from_outcome(completions, llm_judge_rewards=rewards[0],
                                   llm_judge_reasons=reasons) → rewards[1]
        """
        import warnings
        from trl.data_utils import is_conversational, apply_chat_template
        from accelerate.utils import gather
        from trl.extras.profiling import profiling_context
        
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)

        # Repeat all input columns (but "prompt", "completion", and "completion_ids") to match the num of generations
        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}

        # IMPORTANT: Add reference data back to reward_kwargs
        # Since we renamed 'reference' to 'completion' for TRL compatibility,
        # we need to extract the reference data from 'completion' field and add it back
        if "completion" in inputs[0]:
            reward_kwargs["reference"] = [example["completion"] for example in inputs]
        
        # Use stored reference data if available
        if hasattr(self, '_current_reference_data') and self._current_reference_data:
            reward_kwargs["reference"] = self._current_reference_data
        
        # This allows for dynamic reward shaping based on training progress.
        reward_kwargs["trainer_state"] = self.state
        
        # Store previous reward results to enable dependency chain
        previous_rewards = {}
        
        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names)
        ):
            with profiling_context(self, reward_func_name):
                if isinstance(reward_func, nn.Module):  # Module (no PretrainedModel) for compat with compiled models
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                else:
                    # Add previous reward results to enable dependency chain
                    current_reward_kwargs = reward_kwargs.copy()
                    
                    # Add global step for high score tracking
                    current_reward_kwargs["global_step"] = self.state.global_step
                    
                    # Add Gemini API key for strategy summarization
                    current_reward_kwargs["gemini_api_key"] = os.getenv("GEMINI_API_KEY")
                    
                    # Add VLLM configuration for local model support
                    current_reward_kwargs["use_vllm"] = self.custom_config.get("use_vllm", False)
                    current_reward_kwargs["vllm_model_name"] = self.custom_config.get("vllm_model_name", None)
                    current_reward_kwargs["vllm_host"] = self.custom_config.get("vllm_host", "localhost")
                    current_reward_kwargs["vllm_port"] = self.custom_config.get("vllm_port", 8421)
                    
                    # Add scoring mode flag (quantified vs legacy)
                    current_reward_kwargs["use_quantified_scoring"] = self.custom_config.get("use_quantified_scoring", False)
                    
                    # Add Gemini backend configuration
                    current_reward_kwargs["gemini_backend"] = self.custom_config.get("gemini_backend", None)
                    
                    # Pass llm_judge_reward results to outcome reward function (second function, index 1)
                    if i == 1:  # reward_from_outcome
                        current_reward_kwargs["llm_judge_rewards"] = previous_rewards.get("llm_judge_reward")
                        # Pass llm_judge_reasons for proper judge_reason population
                        current_reward_kwargs["llm_judge_reasons"] = previous_rewards.get("llm_judge_reason")
                    
                    output_reward_func = reward_func(
                        prompts=prompts, completions=completions, completion_ids=completion_ids_list, **current_reward_kwargs
                    )
                    
                    # Handle tuple return from llm_judge_reward (rewards, reasons)
                    if isinstance(output_reward_func, tuple) and len(output_reward_func) == 5:
                        # Extract rewards from tuple
                        rewards_list = output_reward_func[0]
                        # Store reasons for later logging
                        if not hasattr(self, '_judge_reasons'):
                            self._judge_reasons = []
                        if not hasattr(self, '_simulator_analysis'):
                            self._simulator_analysis = []
                        if not hasattr(self, '_simulator_triggered_events'):
                            self._simulator_triggered_events = []
                        if not hasattr(self, '_simulator_state_variables'):
                            self._simulator_state_variables = []
                        self._judge_reasons.extend(output_reward_func[1])
                        self._simulator_analysis.extend(output_reward_func[2])
                        self._simulator_triggered_events.extend(output_reward_func[3])
                        self._simulator_state_variables.extend(output_reward_func[4])
                        # Use rewards for tensor conversion
                        output_reward_func = rewards_list
                    
                    # Convert None values to NaN
                    output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]

                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
                
                # Store current reward results for dependency chain
                if reward_func_name == "LLM Judge Reward":
                    previous_rewards["llm_judge_reward"] = rewards_per_func[:, i].tolist()
                    # Also store the judge reasons for the outcome reward function
                    if hasattr(self, '_judge_reasons') and self._judge_reasons:
                        batch_size = len(prompts)
                        if len(self._judge_reasons) >= batch_size:
                            batch_judge_reasons = self._judge_reasons[:batch_size]
                            previous_rewards["llm_judge_reason"] = batch_judge_reasons

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)
        return rewards_per_func