"""
verifiers/examples/taubench_single_sample_train.py
--------------------------------------------------
End-to-end GRPO smoke-test for τ-Bench (retail).

• Uses ONE τ-Bench task only.
• Concurrency = 1 (no async batching, no look-ahead).
• Runs a single GRPO update step to prove the whole stack
  (env  ⇢  vLLM  ⇢  GRPOTrainer) works.

Prerequisites
-------------
1. A vLLM server exposing an OpenAI-compatible endpoint, e.g.

   CUDA_VISIBLE_DEVICES=0 \
   uv run verifiers/inference/vllm_server.py \
       --model 'Qwen/Qwen2.5-7B-Instruct' \
       --port 8000

2. All worker processes must see

   export OPENAI_API_BASE=http://127.0.0.1:8000/v1
   export OPENAI_API_KEY=dummy-key          # any non-empty string

3. Launch with one or many GPUs, e.g.

   CUDA_VISIBLE_DEVICES=1 \
   accelerate launch verifiers/examples/taubench_single_sample_train.py
"""

from __future__ import annotations

import os

from datasets import Dataset

import verifiers as vf
from verifiers.envs.taubench_env import TauBenchEnv
from verifiers.trainers.grpo_config import GRPOConfig

# ---------------------------------------------------------------------
# Model & environment configuration
# ---------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"  # must be an alias served by vLLM
DOMAIN = "retail"
MAX_TURNS = 20  # stop after 20 dialogue turns

# τ-Bench environment (user-LLM lives inside)
env = TauBenchEnv(
    user_model_name=MODEL_NAME,
    domain=DOMAIN,
    max_turns=MAX_TURNS,
    # max_concurrent=1,  # single-threaded for the test
)

# Keep exactly one task for the smoke-test
env.dataset = env.dataset.select([0])  # type: ignore
env.eval_dataset = env.dataset  # type: ignore

print("System prompt:\n", env.dataset[0]["prompt"][0]["content"])  # type: ignore

# ---------------------------------------------------------------------
# Load assistant model
# ---------------------------------------------------------------------
model, tokenizer = vf.get_model_and_tokenizer(MODEL_NAME)
run_name = "taubench-smoke_" + MODEL_NAME.split("/")[-1].lower()

# ---------------------------------------------------------------------
# GRPO configuration (trimmed for a one-step test)
# ---------------------------------------------------------------------
args = GRPOConfig(
    output_dir=f"outputs/{run_name}",
    run_name=run_name,
    per_device_train_batch_size=1,
    num_generations=2,  # GRPO minimum
    generation_batch_size=2,
    gradient_accumulation_steps=1,
    max_steps=1,  # one optimisation step
    num_train_epochs=1,
    max_concurrent=1,  # enqueue one rollout at a time
    logging_steps=1,
    save_strategy="no",
    bf16=True,
    report_to="wandb",
    max_prompt_length=2048,
)

# ---------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------
trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=env,
    args=args,
)

trainer.train()
