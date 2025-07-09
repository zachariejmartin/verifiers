# τ-Bench Integration – Design & Implementation Plan
Author: <your-name>  Date: YYYY-MM-DD  
Target branch: `spike/walkthrough`

---

## 0. Goal

Add support for τ-Bench (Airline & Retail domains) to **verifiers** so we can train GRPO agents on interactive tasks that include an LLM-simulated user, τ-Bench tools and custom rewards – _without changing the trainer, async generator or weight-sync code_.

---

## 1. High-level Architecture

```
┌───────────────────────────────────────────────┐
│ GRPOTrainer / AsyncBatchGenerator (unchanged) │
└───────────────────────────────────────────────┘
                    │
                    ▼
           verifiers.envs.TauBenchEnv
  ┌────────────────────────┴────────────────────────┐
  │                         │                        │
Parser (JSON/function-call) │                τ-Bench Python tools
verifiers.parsers.          │                  verifiers.tools.taubench.*
 function_call_parser       │
                            │
                    Reward computation
                verifiers.rubrics.TauBenchRubric
```

Key differences from existing XML/static-QA environments  
1. **Two LLMs per rollout**  
   • **user LLM** generates the next user utterance.  
   • **agent LLM** (policy) decides tool use / answer.  
2. Conversation continues until τ-Bench’s env signals `done`.  
3. Reward = weighted sum of **Pass@1** success and **valid function-call formatting** (no JSON / schema errors).

---

## 2. New External Dependency

```
pyproject.toml
[project.optional-dependencies]
taubench = ["tau-bench @ git+https://github.com/sierra-research/tau-bench@main"]
```
Kept optional to avoid bloating non-τ-Bench setups.

---

## 3. New Modules & Files

| File | Purpose |
|------|---------|
| `verifiers/parsers/function_call_parser.py` | Generic OpenAI function-call JSON parser. |
| `verifiers/parsers/taubench_parser.py` | Thin wrapper that injects τ-Bench tool schema. |
| `verifiers/tools/taubench/airline_tools.py` | Light wrapper around `tau_bench.envs.airline.tools`. |
| `verifiers/tools/taubench/retail_tools.py` | Wrapper around `tau_bench.envs.retail.tools`. |
| `verifiers/envs/taubench_env.py` | Implements new multi-turn environment with dual-LLM loop. |
| `verifiers/rubrics/taubench_rubric.py` | Reward functions & default weights. |
| `docs/environments/taubench.md` | Usage & config guide. |
| `configs/taubench_example.yaml` | Sample HF-Trainer config. |
| **(this file)** `spec.md` | Implementation blueprint & checklist. |

---

## 4. Core Classes

### 4.1 `FunctionCallParser`
```python
parse_assistant(raw_text) -> dict(reasoning, tool_name, tool_args, answer)
format_tool_result(result: str) -> str
format_answer(answer: str) -> str
```

### 4.2 `TauBenchEnv` (inherits `MultiTurnEnv`)
• Attributes  
```python
self.agent_client   # already exists
self.user_client    # NEW – cheaper model
self.tau_state      # τ-Bench task state
self.task_meta      # task id, domain
```
• Loop
```
user_msg  = user_client(prompt)
while not done:
    assistant = agent_client(history)
    if tool_call:
        result = execute_tool(...)
        history += <result>
    tau_state = tau_env.step(...)
    if done: break
    user_msg  = user_client(history)
```

### 4.3 `TauBenchRubric`
Reward vector `R = [R0, R1]`
| Id | Component | Formula |
|----|-----------|---------|
| R0 | `pass@1` | 0/1 (1 if task solved on first evaluation) |
| R1 | formatting validity | 0/1 (1 if every assistant message conforms to valid function-call schema) |

Default weight vector: `[1.0, 1.0]`  ⇒ total reward ∈ {0,1,2}.

---

## 5. Configuration

Example YAML
```yaml
env:
  name: taubench
  domain: retail
  user_model:
    provider: openai
    model_name: gpt-4o-mini
    temperature: 0.7
  max_steps: 12
  reward_weights: [1, 1]
```

---

## 6. Estimated Timeline

| Task | Effort |
|------|--------|
| Dependency & boiler-plate | 0.5 d |
| Parser + tests | 1.0 d |
| Tool wrappers | 0.5 d |
| TauBenchEnv implementation | 2.0 d |
| Rubric | 0.5 d |
| Docs / config | 0.5 d |
| Integration & CI tests | 0.5 d |
| **Subtotal** | **5–6 days** |
| Contingency | +2 d |

Total: **1½ – 2 weeks**.

---

## 7. Progress Checklist

- [x] Add `tau-bench` optional dependency & CI gate
- [ ] `function_call_parser.py`
- [ ] `taubench_parser.py`
- [ ] Wrap airline tools
- [ ] Wrap retail tools
- [ ] `taubench_env.py` with dual-LLM loop
- [ ] `taubench_rubric.py`
- [ ] Unit tests: parser round-trip
- [ ] Unit tests: single rollout (mocked clients)
- [ ] Example config & docs
- [ ] CI workflow updated for optional extras
- [ ] Manual dry-run on 1 airline + 1 retail task
- [ ] Docs: `docs/README.md`

---

## 8. Acceptance Criteria

1. `pytest -m taubench` passes locally & in CI.  
2. Running
   ```bash
   python verifiers/examples/smola_math_tools.py \
          --env-config configs/taubench_example.yaml \
          --num-rollouts 1
   ```
   completes with a non-crashing reward tensor and at least one pass flag in logs.  
3. No regression in existing environments.  
4. Documentation builds without warnings and explains how to enable τ-Bench.

––– end –––

---

## 9. Notes from τ-Bench Source Review (run.py, envs/base.py, envs/user.py, agents/tool_calling_agent.py)

### 9.1 `tau_bench/run.py`
* Entry-point for the CLI benchmark.  High-level flow:
  1. `load_env(domain)`  → returns a *τ-Bench Env* instance (contains task set & tools).
  2. `build_user_simulator(cfg)` → constructs a `User` object (LLM-backed).
  3. `build_agent(cfg)`         → constructs an `Agent` (ToolCallingAgent or baselines).
  4. `for task in env.list_tasks(): agent.solve(env, user, task)` – **loop lives inside the Agent**.
* **Implication for us**: verifiers already owns the rollout loop. We will **not** call `Agent.solve`; instead we import only `load_env()` and `User`/tool objects.

### 9.2 `tau_bench/envs/base.py`
* Minimal RL-style API: `reset(task_id) -> state`, `step(message, state) -> new_state`.
* Keeps track of: `done` flag, accumulating messages, `tool_schema`, pass flags.
* Does **not** generate language messages; that is left to the Agent/User objects.
* Comparison with `verifiers.envs.*`:
  | Topic                    | τ-Bench `BaseEnv`          | verifiers `Environment` |
  |--------------------------|----------------------------|-------------------------|
  | rollout loop            | external (Agent)           | internal (`rollout()`)  |
  | reward computation      | inside `evaluate(state)`   | inside `Rubric`         |
  | tool execution          | `handle_action()`          | `call_tool()`           |
  | async / concurrency     | none                       | `run_rollouts()` async  |
* **Integration plan**: we will **wrap** a real τ-Bench env instance inside `TauBenchEnv`.  Our wrapper will forward `reset`, `step`, `evaluate`, `tool_schema`, but *own* the multi-turn prompt loop so it plugs into AsyncBatchGenerator.

### 9.3 `tau_bench/envs/user.py`
* `User` class encapsulates the *user simulator* logic.
  * Holds its own OpenAI client, strategy (`llm`, `react`, `verify`, `reflection`).
  * Method `respond(history, state) -> str` returns next user message.
* Important knobs: model name, temperature, verification loops.
* **Action item**: expose this as `TauBenchUserWrapper` so `TauBenchEnv._call_user` just calls `self.user_sim.respond(...)` – we reuse τ-Bench’s built-in user strategies instead of re-implementing.

### 9.4 `tau_bench/agents/tool_calling_agent.py`
* Implements a *full game-loop* inside `solve(env, user, task_id)`:
  1. assistant = model.chat();
  2. env.handle_action → tool / update state;
  3. user.respond();    repeat until done.
* Contains helper to convert OpenAI function-call output to tool invocations.
* **We only need the parsing code**; the outer loop conflicts with verifiers. The useful bits:
  * `parse_assistant_message()`  – we'll copy-or-import into `FunctionCallParser`.
  * JSON schema formatting for tools.

#### 9.4 Native tool schema & validation

```python
from pydantic import BaseModel, Field
from jsonschema import validate

class SearchFlightsArgs(BaseModel):
    from_airport: str = Field(..., min_length=3, max_length=3)
    to_airport: str   = Field(..., min_length=3, max_length=3)
    date: str         = Field(..., regex=r"^\\d{4}-\\d{2}-\\d{2}$")

TOOL_SCHEMA = {
    "name": "search_flights",
    "description": "Search for available flights on a given date",
    "parameters": {
        "type": "object",
        "required": ["from_airport", "to_airport", "date"],
        "properties": {
            "from_airport": {"type": "string"},
            "to_airport":   {"type": "string"},
            "date":         {"type": "string", "pattern": r"^\\d{4}-\\d{2}-\\d{2}$"}
        }
    }
}

# Validation helper used by τ-Bench
validate(instance=args, schema=TOOL_SCHEMA["parameters"])  # raises jsonschema.ValidationError
```

Key take-aways for integration:
* Each tool exposes **OpenAI-style function metadata**: `name`, `description`, and JSON-Schema `parameters`.
* Validation happens **before** executing the Python implementation – if arguments don’t match, τ-Bench raises `jsonschema.ValidationError`.
* Tool argument models are often mirrored as **Pydantic BaseModel** classes so devs can benefit from editor/type hints.

Implications for verifiers wrapper (Option-2 chosen):
1. Our `FunctionCallParser` will be extended to accept τ-Bench’s `tools` list and run the same jsonschema validation so behaviour is identical.
2. The wrapper can still emit **XML `<result>` tags** back to the policy model to keep backward-compatibility with existing prompts; only the assistant-to-tool message changes to JSON.
3. Failure modes map cleanly onto our current reward vector: invalid schema ⇒ formatting penalty.

Checklist additions
- [ ] `taubench_tool_adapter.py` – thin layer converting τ `TOOLS` dict into verifiers `Tool` dataclass.
- [ ] Unit tests: round-trip JSON → validation → execution → XML result.

### 9.5 Integration impact summary
1. Keep τ-Bench Env     → used for task set, tool validation, pass@1 flag.
2. Use τ-Bench User      → wrapped, called once per turn.
3. Borrow parse helpers  → power our `FunctionCallParser`.
4. Ignore τ-Bench Agent  → rollout loop replaced by `MultiTurnEnv.rollout`.

No change needed to AsyncBatchGenerator because `TauBenchEnv.rollout()` still returns a finished completion & reward in a single blocking call.

(End of section)
