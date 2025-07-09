# End-to-End Flow
## Starting Point
```# verifiers/examples/smola_math_tools.py
...
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=args,
)
trainer.train()          # ← ENTRY-POINT
```
`trainer.train()` launches HuggingFace / Accelerate 4 training processes (GPUs 4-7) while a separate `vllm_server.py` (already running) serves 4 generation GPUs (GPUs 0-3).

## 1. Environment & Data
| Component               | Purpose                                                                                   |
|-------------------------|-------------------------------------------------------------------------------------------|
| SmolaToolEnv            | Formats system prompt, executes SmolaAgents tools, counts steps, delegates rewards to rubric |
| SmolaToolRubric         | Reward funcs: correct answer, XML formatting, per-tool usage …                           |
| SmolaParser             | Extracts `<reasoning/>`, `<tool/>`, `<answer/>`, `<result/>`                             |
| RepeatSampler           | Repeats prompts G times for grouped GRPO advantages                                      |
| AsyncDataLoaderWrapper | Pre-fetch ring-buffer of dataset batches   

## 1a. Environment Class Hierarchy (+ key methods)
```
Environment        (abstract)
    │
    └── MultiTurnEnv    (adds turn-loop + tool logic)
            │
            └── SmolaToolEnv  (task-specific: Smola tools + rubric)
```

| Class | Inherits | Purpose & key overrides / methods |
|-------|----------|-----------------------------------|
| **Environment** (`verifiers/envs/environment.py`) | `ABC` | • `generate()` – umbrella call used by `AsyncBatchGenerator`; delegates to `run_rollouts` then scores.<br>• `run_rollouts()` – async fan-out to `_run_single`.<br>• `get_model_response()` – **only place that calls `oai_client.chat.completions`**.<br>• `process_env_results()` → `process_chat_format` / `process_completion_format` – converts text ↦ token ids, applies `mask_env_responses`, `mask_truncated_completions`. |
| **MultiTurnEnv** (`verifiers/envs/multiturn_env.py`) | `Environment` | Adds a loop for iterative tool use.<br>• `rollout()` – maintains `(prompt, completion)` history, calls `env_response()` after each assistant turn.<br>• Default `call_tool()` (no-op) & `env_response()` (echo error) – meant to be overridden. |
| **SmolaToolEnv** (`verifiers/envs/smola_tool_env.py`) | `MultiTurnEnv` | Task-specific environment for SmolaAgents tools.<br>• `__init__()` formats system prompt with tool schemas.<br>• `_format_tool_descriptions()` – pretty-prints tool JSON schema.<br>• `get_reward_funcs()` / `get_reward_weights()` – delegates to **SmolaToolRubric**.<br>• `_get_step_count()` – counts assistant tool calls to enforce `max_steps`.<br>• `is_completed()` – stop condition: valid `<answer>` or max steps.<br>• `call_tool()` – executes the real Python tool, returns string result.<br>• `env_response()` – wraps tool result into `<result>…</result>` user message. |

Relation to training loop
-------------------------
* `AsyncBatchGenerator` (in **trainer**) calls `env.generate()` every time it needs fresh completions.
* `Environment.generate()` uses `self.client` (the shared **oai_client**) to query the vLLM server and then scores rewards via the attached rubric.
* Because `SmolaToolEnv` inherits all of this, the only custom pieces you typically touch are the **tool execution** and **reward specification**, everything else is handled by its parents.

## 2. Training-Side Pipeline (GPUs 4-7)
### Key Methods
`_prepare_inputs`
- weight sync to vLLM when needed
- launches / maintains async generator pipeline
- gathers rewards → computes advantages → shuffles & buffers
`compute_loss` (GRPO / BNPO / DR-GRPO)
flowchart TD
    A[HF Trainer loop] --> B[_prepare_inputs()]
    B --> C{need new completions?}
    C -->|yes| D[MAIN rank<br>_move_model_to_vllm()]
    D --> E[MAIN rank<br>AsyncBatchGenerator.submit()]
    E --> F[wait for AsyncBatchGenerator.get_batch()] 
    F --> G[broadcast (prompt_ids, completion_ids,<br>rewards) to all ranks]
    G --> H[compute advantages & buffer<br>gradient_accumulation_steps slices]
    H --> I[compute_loss() -> backward]
    I --> J[optimizer.step()]

## 2a. AsyncBatchGenerator ‑ life-cycle and threads
The `AsyncBatchGenerator` is built **once** in `GRPOTrainer.__init__` on the **main training process only**:
```python
# grpo_trainer.py
self.async_generator = AsyncBatchGenerator(
    env         = self.env,
    client      = self.oai_client,   # ← shared OpenAI wrapper
    model_name  = self._get_model_name(),
    sampling_args = self._get_sampling_args(),
    num_batches_ahead = self.num_batches_ahead,
    ...
)
```

### When is it started?
Inside `_prepare_inputs()` the first time we need completions:
```python
if not self._async_started and self.accelerator.is_main_process:
    self.async_generator.start()      # ← spawns a **daemon thread**
```

### Main-process responsibilities
1. **Submit** a `BatchRequest` for every new batch:
   ```python
   self.async_generator.submit_batch(request)   # enqueue → request_queue
   ```
2. **Wait** for a matching `BatchResult`:
   ```python
   batch_result = self.async_generator.get_batch(batch_id_to_retrieve)
   ```
3. **Broadcast** processed tensors (prompt_ids, completion_ids, rewards) to all other ranks.

Non-main processes never call `submit_batch`; they simply wait for the broadcast.

### Inside the AsyncBatchGenerator thread
```
_generation_worker()
    while True:
        request = request_queue.get()
        result  = _generate_batch(request)
        result_queue.put(result)          # producer side
```
* `_generate_batch` → `env.generate()` → `Environment.get_model_response()`
  calls `oai_client.chat.completions.create(…)` → **vLLM server**.
* After scoring and tokenisation a `BatchResult` is pushed to `result_queue`.

### get_batch() – consumer side (still main process)
```
result = self.result_queue.get(timeout=…)
self.completed_batches[result.batch_id] = result
```
so the same object that submitted the request also retrieves and processes the result.

### Queues & threads summary
| Queue | Producer | Consumer | Thread / Process |
|-------|----------|----------|------------------|
| `request_queue` | main process (trainer) via `submit_batch` | worker thread (`_generation_worker`) | same process |
| `result_queue`  | worker thread | main process (`get_batch`) | same process |

No additional queue exists on the vLLM side; the HTTP response returns synchronously to `_generate_batch`, which wraps it into the `BatchResult`.

### Key code references
* `grpo_trainer.py`  line ~920  – call to `self.async_generator.start()`
* `grpo_trainer.py`  line ~960  – building `BatchRequest`
* `trainers/async_batch_generator.py`
  * class `BatchRequest` (metadata & masking flags)
  * `submit_batch` / `get_batch`
  * `_generation_worker` (thread body)
  * `_generate_batch` – the bridge to `Environment.generate()`
* `envs/environment.py`  line ~170 – `client.chat.completions.create` (vLLM call)

With this design **computation overlaps**:
while the main process is computing gradients for batch *k*, the worker thread is already querying vLLM and scoring batch *k+1*, keeping GPUs and vLLM busy in parallel.

## 2b. Two-GPU timeline (sequence diagram)
```text
Time →

MP-0 main     │ build BatchRequest            │  wait get_batch()                    │ compute loss
              │ submit_batch()                │  (blocks)                            │
              ▼                               │                                       ▼
request_queue │                               │                                       
              │                               │                                       
WG-thread      ◀─ dequeue ── _generate_batch ─┤                                       
              │   env.generate()             │                                       
              │   ↳ vLLM HTTP call           │                                       
              │   scoring + tokenise         │                                       
              └─ result_queue.put() ─────────►│                                       
                                               
MP-0 main     ◀──────────────── result_queue ──┘ broadcast to MP-1
                                               │
MP-1          ◀─ accelerator.broadcast ─────────┘ slices + lossBackprop
                                               
vLLM-srv      POST /v1/chat/completions  (sync HTTP)  → assistant text
```
The solid arrows show data movement:
* **submit_batch** enqueues the request.
* Worker thread pulls it, synchronously calls vLLM, scores rewards, pushes a `BatchResult`.
* Main process retrieves the result, broadcasts tensors to the other rank, and proceeds with gradient computation.

## 3. Asynchronous Generation (MAIN Rank → GPUs 0-3)
### AsyncBatchGenerator
1. Creates BatchRequest containing full batch prompts.
2. Calls OpenAI endpoint served by vLLM (/v1/chat/completions).
3. Receives completions, lets SmolaToolEnv:
    - parse assistant XML
    - execute real Python tools (CalculatorTool, PythonInterpreterTool, …)
    - compute rewards via SmolaToolRubric.
4. Packs token IDs, masks, rewards → returns to trainer.

## 4. Weight Synchronisation
### Trainer-side
```  per_token_logps(model)
  old_logps (buffer)
  advantages
  ──► GRPO clipping (ε_low, ε_high)
       + β·KL(ref)     → loss
```
### VLLMClient
- Sub-classes openai.OpenAI (same REST API for generation).
- `init_communicator()`
GET /get_world_size
POST /init_communicator (host, port, world_size)
Builds PyNcclCommunicator (rank = world_size-1).
- `update_named_param`
POST /update_named_param (dtype, shape).
Broadcast tensor via NCCL (pynccl_comm.broadcast).

## 5. Generation-Side Server (vllm_server.py, GPUs 0-3)
### FastAPI Endpoints
| Path                                | Purpose                                              |
|-------------------------------------|------------------------------------------------------|
| /v1/chat/completions, /v1/completions | OpenAI-compatible generation (pooled, token-chunked) |
| /init_communicator                  | Create NCCL group in every worker                   |
| /update_named_param                 | Notify workers before NCCL broadcast                |
| /batch_update_named_params          | Batched variant                                     |
| /reset_prefix_cache                 | Clear KV-cache                                      |
| /close_communicator                 | Tear down NCCL group 

### Internals
1. Spawns worker processes (llm_worker) each owning a vllm.LLM.
2. WeightSyncWorkerExtension inside every worker:
    - init_communicator ⇒ creates its own PyNcclCommunicator.
    - update_named_param ⇒ allocates empty tensor on GPU, listens to broadcast, loads weights.
3. batch_processing_loop
    - Pools requests by signature, performs token-chunk dynamic batching, streams chunks back.

## 6. GPU Topology
```_move_model_to_vllm()
    for name, param in model.named_parameters():
        vllm_client.update_named_param(name, param.data)
    vllm_client.reset_prefix_cache()
```

## 7. Training Timeline (Simplified)
┌─────────────── Generation ───────────────┐  ┌──────────── Training ────────────┐
| GPUs 0-3                                 |  | GPUs 4-7                         |
| + vllm_server (FastAPI)                  |  | + Accelerate (4 procs)           |
|   + n × llm_worker (vLLM.LLM)            |  |   + GRPOTrainer                  |
|   + WeightSyncWorkerExtension            |  |   + AsyncBatchGenerator          |
|                                          |  |   + VLLMClient (NCCL rank = 8)   |
└──────────────────────────────────────────┘  └──────────────────────────────────┘
          └────── HTTP + NCCL broadcast channel ──────►

## 8. Key Classes Cheat-Sheet
| File                        | Class                                | Role                                                       |
|-----------------------------|--------------------------------------|------------------------------------------------------------|
| smola_tool_env.py           | SmolaToolEnv                         | RL environment, tool execution, reward hooks               |
| smola_tool_rubric.py        | SmolaToolRubric                      | Defines reward functions & weights                         |
| grpo_trainer.py             | GRPOTrainer                          | HF-Trainer subclass, async generation, loss/advantage, weight sync |
| async_batch_generator.py    | AsyncBatchGenerator                  | Keeps generator pipeline full                              |
| repeat_sampler (in grpo_trainer.py) | RepeatSampler                  | Repeats prompts G times for grouped rewards                |
| vllm_client.py              | VLLMClient                           | OpenAI-compatible client + NCCL broadcaster                |
| vllm_server.py              | FastAPI app & WeightSyncWorkerExtension | Serves generation, receives/broadcasts weights         |

### Summary
The system implements an actor–generator split:
    - Training GPUs (actor): compute GRPO gradients, periodically push updated parameters.
    - Generation GPUs (generator): serve low-latency batched inference via vLLM
    - NCCL + custom REST endpoints guarantee the generator always hosts the freshest policy without pausing generation.

Detailed walkthrough
====================

1. **Creating the request**  
   MP-0 (inside `_prepare_inputs`) packages a list of *raw* prompts +
   dataset answers into `BatchRequest`.  
   Flags such as `mask_env_responses`, `max_completion_length` are copied in.

2. **Submitting**  
   `submit_batch()` simply enqueues; only MP-0 ever calls it.

3. **Worker thread consumes**  
   `_generation_worker()` runs inside MP-0 but on its own thread.  
   It pops the request, calls `_generate_batch()`.

4. **`_generate_batch()` → `env.generate()`**  
   • `env.generate()` fans out *prompts* to `run_rollouts`.  
   • Each rollout calls **SmolaToolEnv.rollout** which may execute **multiple**
     vLLM calls if the task needs several tool steps.  
   • Each vLLM call is synchronous: the thread blocks until the HTTP response
     is back.

5. **Environment response & scoring**  
   After every assistant turn `env_response()` injects `<result>…</result>`
   into the conversation.  
   Once `is_completed()` returns True, the full conversation + final answer is
   passed to the rubric for reward computation.

6. **Tokenisation & masks**  
   `process_env_results()` converts prompt / completion text into token IDs,
   applying `mask_env_responses` so those `<result>` tokens can be zero-masked.

7. **BatchResult to the main process**  
   WG-thread pushes the finished `BatchResult` into `result_queue`.

8. **Retrieval**  
   MP-0 waits on `get_batch(batch_id)`; as soon as the item is present it
   receives **all** tensors needed for loss computation (prompt_ids,
   completion_ids, rewards, masks).

9. **Broadcast to other ranks**  
   MP-0 uses `accelerator` to send the processed tensors to MP-1.  
   Other training ranks therefore never touch queues or vLLM.

10. **Gradient step**  
    Every process slices its share of the batch → `compute_loss()` →
    backwards → optimiser update.

11. **Overlap**  
    While MP-0/MP-1 do forward+backward for batch `k`, the WG-thread is
    already talking to vLLM to generate batch `k+1`, thus keeping inference
    and training overlapped.

Two-GPU specifics
-----------------

* `world_size = 2` comes from `accelerate launch --num-processes 2 …`.  
* Only rank 0 owns the `AsyncBatchGenerator`.  
* vLLM itself is external; it runs on *generation* GPUs (0-N) and returns
  responses via HTTP; it does **not** push to any Python queue.

What about longer multi-turn conversations?
-------------------------------------------

`SmolaToolEnv.rollout()` controls the loop:

```python
messages = prompt            # system + user
while not self.is_completed(messages, state):
    assistant = get_model_response(messages)
    messages.append({"role":"assistant", "content":assistant})

    env_msg, state = env_response(messages, state)
    messages.append(env_msg)
```

So each assistant turn (from vLLM) is immediately followed by an environment
response and the loop repeats **within the same WG-thread** until completion.
Only when the rollout finishes does the thread return the final completion to
the trainer.

Therefore, by the time `get_batch()` finishes, **all tool interactions and
environment messages are already embedded** in `completion_ids`, and rewards
have been calculated—ready for gradient computation.

## 3a. Concurrency inside `Environment.run_rollouts`
```
run_rollouts()  (sync fn)
   └─ creates new asyncio event-loop (private to WG-thread)
        ├─ ThreadPoolExecutor(max_workers = max_concurrent)
        │     • blocking `rollout()` (tool exec + sync HTTP) run here
        └─ coroutine _run_all()
              ├─ semaphore = asyncio.Semaphore(max_concurrent)
              └─ for each prompt:
                     await _run_single()

_run_single()
   async with semaphore:                 # limit #parallel rollouts
       completion, state = await asyncio.to_thread(self.rollout, ...)
```
Key points
* **Semaphore** – ensures at most `max_concurrent` rollouts are active.
* **`asyncio.to_thread`** – submits the blocking `rollout()` to the thread-pool;
  returns control to the event-loop immediately.
* **ThreadPoolExecutor** – hosts real OS threads; each thread runs synchronous
  code: tool calls, `client.chat.completions.create(...)` HTTP call, XML
  parsing, reward scoring.
* The event-loop itself never performs blocking I/O; it just schedules tasks
  and aggregates results with `tqdm.asyncio.gather()`.
* Default `max_concurrent` = **128**, so up to 128 vLLM requests + tool calls
  can be in-flight simultaneously (bounded also by the semaphore).

This hybrid design lets the WG-thread overlap many slow HTTP/tool operations
without rewriting everything to fully-async libraries.