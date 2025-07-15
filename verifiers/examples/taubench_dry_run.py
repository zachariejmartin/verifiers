"""verifiers/examples/taubench_dry_run.py

Quick sanity-check that TauBenchEnv, parser and rubric work end-to-end.

It runs **one** τ-Bench task with **concurrency = 1** and prints the
assistant completion plus the environment-computed reward.

Prerequisites
-------------
• `tau-bench` must be installed (install extra: `pip install "verifiers[taubench]"`).
• A valid OpenAI API key in the `OPENAI_API_KEY` environment variable.

Run
---
python verifiers/examples/taubench_dry_run.py
"""

from __future__ import annotations

import asyncio
import os

from openai import AsyncOpenAI

from verifiers.envs.taubench_env import TauBenchEnv

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DOMAIN = "retail"  # "airline" or "retail"
ASSISTANT_MODEL = MODEL_NAME  # the same model τ-Bench expects by default
MAX_TURNS = 20  # keep the rollout short for a dry-run

import verifiers as vf

model_kwargs = dict(
    torch_dtype="bfloat16",  # or torch.float16 / bf16 etc.
    attn_implementation="sdpa",  # <- turn Flash-Attn OFF
    use_cache=False,
)

model, tokenizer = vf.get_model_and_tokenizer(
    MODEL_NAME, use_liger=False, model_kwargs=model_kwargs
)
run_name = "taubench-grpo_" + MODEL_NAME.split("/")[-1].lower()

args = vf.grpo_defaults(run_name=run_name)

host, port = args.vllm_server_host, args.vllm_server_port
os.environ["OPENAI_API_KEY"] = "EMPTY"  # just like GRPOTrainer
os.environ["OPENAI_API_BASE"] = f"http://{host}:{port}/v1"  # point to your vLLM server


async def main() -> None:
    # if "OPENAI_API_KEY" not in os.environ:
    #     raise EnvironmentError("Please set OPENAI_API_KEY before running the dry-run.")

    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    env = TauBenchEnv(user_model_name=MODEL_NAME, domain=DOMAIN, max_turns=MAX_TURNS)

    messages = [
        {
            "role": "system",
            "content": env._wiki
        },
        {
            "role": "user",
            "content": "Hi, I would like to return something."
        }
    ]

    content, response_obj = await env.get_model_response(
        prompt=messages,
        client=client,
        model=model,
        # sampling_args=sampling_args,
        message_type=env.message_type,
        tools=env._tools_info,
        tool_choice="auto",
    )
    print(response_obj)

#     # Take the very first task from the internal HF dataset
#     row = env.get_dataset(n=1)[0]
#     prompt = row["prompt"]
#     answer = row["answer"]  # empty string – τ-Bench handles evaluation internally

#     print("––– SYSTEM PROMPT –––")
#     print(prompt[0]["content"])
#     print("–––––––––––––––––––––\n")

#     completion, state = await env.rollout(
#         client=client,
#         model=ASSISTANT_MODEL,
#         prompt=prompt,
#         answer=answer,
#         # sampling_args={"temperature": 0.2},
#     )

#     print("Assistant messages:\n")
#     for msg in completion:
#         role = msg.get("role", "assistant")
#         print(f"[{role.upper()}] {msg['content']}\n")

#     print("Final state:", state)
#     print("Reward from τ-Bench:", state.get("reward"))


if __name__ == "__main__":
    asyncio.run(main())
