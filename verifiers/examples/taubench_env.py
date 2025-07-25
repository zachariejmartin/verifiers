"""
verifiers/examples/taubench_env.py
--------------------------------------------------
End-to-end GRPO smoke-test for τ-Bench (retail).

• Uses ONE τ-Bench task only.
• Concurrency = 1 (no async batching, no look-ahead).
• Runs a single GRPO update step to prove the whole stack
  (env  ⇢  vLLM  ⇢  GRPOTrainer) works.

Prerequisites
-------------
1. A vLLM server exposing an OpenAI-compatible endpoint, e.g.

   CUDA_VISIBLE_DEVICES=0,1 uv run verifiers/inference/vllm_server.py \
    --model 'Qwen/Qwen2.5-7B-Instruct' \
    --tensor-parallel-size 4 \
    --max-model-len 8192 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --enable-prefix-caching \
    --host 0.0.0.0 \
    --port 8000 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes

2. All worker processes must see

   export OPENAI_API_BASE=http://127.0.0.1:8000/v1
   export OPENAI_API_KEY=dummy-key          # any non-empty string

3. Launch 

   CUDA_VISIBLE_DEVICES=2,3 accelerate launch --config-file configs/zero3.yaml --num-processes 2 verifiers/examples/taubench_env.py
"""

from __future__ import annotations

import os

from datasets import Dataset

import verifiers as vf
from verifiers.envs.taubench_env import TauBenchEnv
from verifiers.trainers.grpo_config import GRPOConfig
from verifiers.prompts.few_shots import TAUBENCH_RETAIL_FEW_SHOT

# ---------------------------------------------------------------------
# Model & environment configuration
# ---------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"  # must be an alias served by vLLM
DOMAIN = "retail"
MAX_TURNS = 20  # stop after 20 dialogue turns

# ---------------------------------------------------------------------
# Load assistant/user model
# ---------------------------------------------------------------------
model_kwargs = dict(
    torch_dtype="bfloat16",  # or torch.float16 / bf16 etc.
    attn_implementation="sdpa",  # TODO: <- turn Flash-Attn OFF
    use_cache=False,
)

model, tokenizer = vf.get_model_and_tokenizer(
    MODEL_NAME, use_liger=False, model_kwargs=model_kwargs
)

run_name = "taubench-smoke_" + MODEL_NAME.split("/")[-1].lower()

args = vf.grpo_defaults(run_name=run_name)

host, port = args.vllm_server_host, args.vllm_server_port
os.environ["OPENAI_API_KEY"] = "EMPTY"  # just like GRPOTrainer
os.environ["OPENAI_API_BASE"] = f"http://{host}:{port}/v1"  # point to your vLLM server

# τ-Bench environment (user-LLM lives inside)
env = TauBenchEnv(
    user_model_name=MODEL_NAME,
    domain=DOMAIN,
    max_turns=MAX_TURNS,
    few_shot=TAUBENCH_RETAIL_FEW_SHOT,
    tools_to_remove=["think"],
    max_concurrent=1,  # single-threaded for the test
)

# Keep exactly one task for the smoke-test
env.dataset = env.dataset.select([0])  # type: ignore
env.eval_dataset = env.dataset  # type: ignore

print("System prompt:\n", env.dataset[0]["prompt"][0]["content"])  # type: ignore


# ---------------------------------------------------------------------
# GRPO configuration (trimmed for a one-step test)
# ---------------------------------------------------------------------
peft_config = vf.lora_defaults()

args = GRPOConfig(
    output_dir=f"outputs/{run_name}",
    run_name=run_name,
    per_device_train_batch_size=2,
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
    max_prompt_length=8192,
)

# ---------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------
trainer = vf.GRPOTrainer(
    model=model, processing_class=tokenizer, env=env, args=args, peft_config=peft_config
)

trainer.train()
