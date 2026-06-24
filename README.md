siehe auch 
KI findet Schlupflöcher in Gesetzen und Steuern – Society Hacking erklärt https://share.google/Bpjm28I2e3swGsoL7


# SocioHack

Code and data for the paper [Large Language Models Hack Rewards, and Society](https://arxiv.org/abs/2606.04075).
It contains 72 societal environments for studying whether RL policies rediscover regulatory loopholes by reward hacking. The repository includes raw environments, GRPO training code, and Gemini-based evaluation scripts.

## Contents

- `data/`: raw JSON environments (`historical/`, `synthetic/`, `fictional/`)
- `src/`: Hydra training entrypoint, custom GRPO trainer, reward code, LLM client
- `configs/`: base configs for each dataset family
- `preprocessing/`: raw JSON to HuggingFace dataset conversion scripts
- `scripts/`: vLLM and training launchers
- `eval/`: LLM-as-judge evaluation scripts

## Install

```bash
pip install -r requirements.txt
```

TRL is pinned to `0.29.0`. Match `vllm`, `flash-attn`, `deepspeed`, CUDA, and GPU drivers to your machine.

## Environment

Gemini is used for the simulator and judge:

```bash
export GEMINI_API_KEY=<your_key>
```

Optional settings:

```bash
export GEMINI_BACKEND=google_sdk      # default
export GEMINI_BACKEND=openai_compat
export GEMINI_SDK_BASE_URL=<url>      # google_sdk custom endpoint
export LLM_BASE_URL=<url>             # openai_compat custom endpoint
```

Datasets are read from `$HF_HOME/datasets/` and checkpoints default to `./checkpoints`:

```bash
export HF_HOME=/path/to/huggingface
export SOCIOHACK_SAVE_BASE_PATH=/path/to/checkpoints
```

Do not commit API keys, custom endpoints, checkpoints, or run outputs.

## Prepare Data

```bash
bash preprocessing/prepare_historical.sh
bash preprocessing/prepare_synthetic.sh
bash preprocessing/prepare_fictional.sh
```

Dataset names are `<scenario>` for Historical, `<scenario>_synthetic` for Synthetic, and `<scenario>_fictional` for Fictional.

## Train

Training uses a vLLM rollout server plus GRPO training. Start them in two terminals:

```bash
# terminal 1
bash scripts/start_vllm.sh historical

# terminal 2: one Historical scenario
bash scripts/train_single.sh historical dataset.name=1_10b5
```

Batch runs:

```bash
bash scripts/train_historical.sh
bash scripts/train_synthetic.sh
bash scripts/train_fictional.sh
```

`train_single.sh` accepts normal Hydra overrides, for example:

```bash
bash scripts/train_single.sh historical dataset.name=1_10b5 training.batch_size=4
```

The default policy is `Qwen/Qwen3-30B-A3B-Instruct-2507`; configs default to GPU `0` for training and GPU `1` for vLLM. Override `gpu.training.gpu_ids` and `gpu.vllm.gpu_ids` as needed.

## Evaluate

Training writes `loopholes_<dataset.name>.json` in the launch directory. Point `SOCIOHACK_MINED_DIR` at that directory:

```bash
SOCIOHACK_MINED_DIR=. python eval/eval_historical.py
SOCIOHACK_MINED_DIR=. python eval/eval_synthetic.py
SOCIOHACK_MINED_DIR=. python eval/eval_fictional.py
```

The evaluators compare mined strategies against ground-truth patches with Gemini and cache pairwise judgments by MD5.

## Notes

- Main training entrypoint: `python -m src.train --config-name <config>`
- Per-scenario run artifacts: `loopholes_*.json`, `rollouts_*.csv`, `llm_debug_*.csv`
- Set `project.use_wandb=false` to disable W&B logging.
- Set `SOCIOHACK_UNIQUE_ARTIFACTS=1` to include `project.suffix` in artifact filenames for repeated runs of the same scenario.
