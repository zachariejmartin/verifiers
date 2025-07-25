"""verifiers.envs.taubench_env
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A *very thin* wrapper that lets the Verifiers RL stack treat τ-Bench
(Airline/Retail) as just another ``MultiTurnEnv``.

The class is intentionally conservative — it delegates *all* task logic,
tool execution and reward computation to the original τ-Bench packages.

We still need a few hooks so GRPO can:
1. decide when a rollout is finished (`is_completed`)
2. inject the *environment* (i.e. **user-LLM + external tools**) turn
   back into the chat history (`env_response`).

The details will be filled incrementally in follow-up patches; for now we
keep the implementation minimal but runnable so the rest of the
code-base can import it without crashing.

Usage (pseudo-code)
-------------------
>>> env = TauBenchEnv(domain="airline")
>>> completion, final_state = env.rollout(client, model, prompt, answer)

"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from typing import Any, Dict, List, Tuple

from datasets import Dataset
from openai import AsyncOpenAI, OpenAI

from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.prompts.system_prompts import TAU_BENCH_PROMPT

logger = logging.getLogger(__name__)

try:
    import tau_bench  # type: ignore
    from tau_bench.envs import Env, get_env  # type: ignore
    from tau_bench.types import (
        RESPOND_ACTION_NAME,
        Action,
        EnvResetResponse,
        EnvResponse,
        RewardResult,
    )
except ModuleNotFoundError as err:  # pragma: no cover
    raise ImportError(
        "TauBenchEnv requires the optional 'tau-bench' dependency.\n"
        "Install with 'uv add verifiers[all]'"
    ) from err


class TauBenchEnv(MultiTurnEnv):
    """Wrap Airline / Retail τ-Bench environment into Verifiers API."""

    SUPPORTED_DOMAINS = {"airline", "retail"}

    def __init__(
        self,
        user_model_name: str,
        domain: str = "retail",
        task_split: str = "train",
        task_ids: (
            List[int] | None
        ) = None,  # TODO: pass the number of tasks you want to run
        max_turns: int = 10,
        few_shot: List[Dict[str, str]] | None = None,
        tools_to_remove: List[str] = [],
        **kwargs: Any,
    ):
        if domain not in self.SUPPORTED_DOMAINS:
            raise ValueError(f"domain must be one of {self.SUPPORTED_DOMAINS}")

        # Use this to remove tools from the wrapped env being populated in the tool desc pass to model
        self._tools_to_remove = tools_to_remove

        # Only allowing for OpenAI model that τ-Bench supports out-of-the-box.
        # Using a remote model for the user keeps the training stack simple
        # while the assistant still runs on vLLM / local GPUs.
        self._user_model_name = user_model_name
        self._user_strategy = "llm"

        # Persist parameters for later per-rollout env construction
        self._domain = domain
        self._task_split = task_split

        # We create a *throw-away* τ-Bench env here purely to obtain the task list.
        tmp_env: Env = get_env(
            env_name=self._domain,
            task_split=self._task_split,
            user_strategy=self._user_strategy,
            user_model=self._user_model_name,
            user_provider="openai",
        )
        # print(f"τ-Bench environment tools info:\n{tmp_env.tools_info}")
        # print(f"τ-Bench environment wiki:\n{tmp_env.wiki}")

        self._tasks = tmp_env.tasks  # store Task objects for dataset rows / iteration
        self._wiki: str = getattr(tmp_env, "wiki", "")
        if not self._wiki:
            raise ValueError("τ-Bench environment policy not found.")

        self._tools_info = getattr(tmp_env, "tools_info", None)
        if self._tools_info is None:
            raise ValueError("No tools provided to τ-Bench environment.")
        self._few_shot = ""
        if few_shot:
            self._few_shot = f"Here are some examples:\n{few_shot}"

        hf_ds_train, hf_ds_eval = self._build_hf_datasets()

        super().__init__(
            dataset=hf_ds_train,
            eval_dataset=hf_ds_eval,
            max_turns=max_turns,
            message_type="chat",
            system_prompt=None,  # we embed system in dataset rows explicitly
            few_shot=few_shot or [],
            mask_env_response=True,
            **kwargs,
        )

        # ---------------- Rubric & parser setup ----------------------
        from verifiers.parsers.taubench_parser import TauBenchParser
        from verifiers.rubrics.taubench_rubric import TauBenchRubric

        self.llm_parser = TauBenchParser()
        self.rubric = TauBenchRubric()

        logger.info(
            "TauBenchEnv initialised (%s) with %s tasks",
            domain,
            len(self._tasks),
        )
        self._log_prompt_token_stats()

    # ------------------------------------------------------------------
    # MultiTurnEnv abstract methods
    # ------------------------------------------------------------------

    def is_completed(
        self, messages: List[Dict[str, Any]], state: Dict[str, Any], **_: Any
    ) -> bool:  # noqa: D401
        """Rollout is over when τ-Bench signals done or we exceeded max_turns."""
        return state.get("done", False)

    def env_response(
        self, messages: List[Dict[str, Any]], state: Dict[str, Any], **kwargs: Any
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Return the next *user* message by directly calling τ-Bench env.

        The wrapped `tau_env` already embeds its own ``user`` simulator, so
        we simply forward assistant actions and relay the observation string
        back to GRPO as a chat message.
        """

        # FIRST TURN (assistant hasn't acted yet)
        if "tau_state" not in state:

            # Build a fresh τ-Bench environment for *this* rollout only
            if "tau_env" not in state:
                state["tau_env"] = get_env(
                    env_name=self._domain,
                    task_split=self._task_split,
                    user_strategy=self._user_strategy,
                    user_model=self._user_model_name,
                    user_provider="openai",
                )

            tau_env: Env = state["tau_env"]  # type: ignore[assignment]
            # Determine task id from state["task"] (string) – convert to int, allow "default" for random
            task_str = state.get("task", "default")
            if task_str == "default" or task_str is None:
                task_id = None  # τ-Bench will pick randomly
            else:
                try:
                    task_id = int(task_str)
                except (TypeError, ValueError):
                    task_id = None

            reset_res: EnvResetResponse = tau_env.reset(task_id)  # EnvResetResponse
            state["tau_state"] = reset_res
            state["task_id"] = task_id
            state["done"] = False

            return {"role": "user", "content": reset_res.observation}, state

        # ---------- SUBSEQUENT USER STEPS ----------
        if not messages:
            raise ValueError("Assistant message history is empty on environment step")

        assistant_content = messages[-1]
        action = self._message_to_action(message=assistant_content)

        tau_env: Env = state["tau_env"]  # type: ignore[assignment]
        step_res: EnvResponse = tau_env.step(action)

        # Update state with most recent τ-Bench response object
        state["tau_state"] = step_res
        state["done"] = step_res.done

        # If episode finished compute reward once and store
        if step_res.done:
            reward_res: RewardResult = tau_env.calculate_reward()
            state["reward"] = reward_res.reward

        # ------------------------------------------------------------------
        # Build the message that will be fed back to the assistant.
        # If the assistant called a tool, we emit a `role="tool"` message with
        # the execution result (OpenAI function-call convention). Otherwise we
        # behave like a regular user turn.
        # ------------------------------------------------------------------
        if action.name != RESPOND_ACTION_NAME:
            tool_call = assistant_content["tool_calls"][0]
            tool_msg = {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "name": tool_call["function"]["name"],
                "content": step_res.observation,
            }
            return tool_msg, state

        return {"role": "user", "content": step_res.observation}, state

    # ------------------------------------------------------------------
    # Convenience helpers (non-mandatory for MultiTurnEnv)
    # ------------------------------------------------------------------

    def _format_tool_descriptions(self) -> str:
        """(Deprecated) Kept for backward-compatibility, now returns ''.

        We no longer embed the full tool schema in the system prompt because
        vLLM/OpenAI receive it via the `tools=[…]` argument instead.
        """
        return ""

    def _log_prompt_token_stats(self) -> None:
        """
        Log the number of tokens the assistant will see **before** the first
        user turn: wiki + system instructions + tool descriptions.
        Uses `tiktoken` if available, otherwise falls back to whitespace split.
        """
        system_txt = f"{self._wiki}\n{TAU_BENCH_PROMPT}"
        tool_txt = json.dumps(self._tools_info, ensure_ascii=False)
        full_txt = f"{system_txt}\n{tool_txt}"

        try:
            import tiktoken  # type: ignore

            enc = tiktoken.get_encoding("cl100k_base")
            n_prompt_tokens = len(enc.encode(system_txt))
            n_tool_tokens = len(enc.encode(tool_txt))
            n_total_tokens = len(enc.encode(full_txt))
        except Exception:
            # crude fallback – still useful as an order-of-magnitude hint
            n_total_tokens = len(full_txt.split())

        logger.info(
            "TauBenchEnv: %s prompt tokens + %s tool tokens ≈ %s total tokens",
            n_prompt_tokens,
            n_tool_tokens,
            n_total_tokens,
        )

    def _message_to_action(self, message: Dict[str, Any]) -> Action:
        """Convert assistant message to τ-Bench Action using vLLM `tool_calls`."""

        if (
            "tool_calls" in message
            and message["tool_calls"]
            and len(message["tool_calls"]) > 0
            and message["tool_calls"][0].get("function") is not None
        ):
            tool_call = message["tool_calls"][0]
            return Action(
                name=tool_call["function"]["name"],
                kwargs=json.loads(tool_call["function"]["arguments"]),
            )

        # Fallback: plain respond action (strip private tags just in case)
        return Action(
            name=RESPOND_ACTION_NAME,
            kwargs={"content": self.llm_parser.strip_private_tags(message["content"])},
        )

    def _build_hf_datasets(self):
        """Convert τ-Bench Task objects into minimal HF datasets."""

        tool_txt = (
            ""  # tool schema now passed via vLLM `tools` param, keep prompt clean
        )

        def _row(idx, task):  # type: ignore[annassign]
            return {
                "prompt": [
                    {
                        "role": "system",
                        "content": f"{self._wiki}\n{TAU_BENCH_PROMPT.format(tool_txt=tool_txt)}\n{self._few_shot}",
                    },
                ],
                "answer": "",  # reward is computed by τ-Bench
                "task": str(idx),  # string task identifier passed to rollout
                "task_id": idx,
                "info": {},
            }

        train_rows = [_row(i, t) for i, t in enumerate(self._tasks)]
        hf_ds_train = Dataset.from_list(train_rows)

        # For evaluation we load the official test split if available
        try:
            if self._domain == "retail":
                from tau_bench.envs.retail.tasks_test import (
                    TASKS_TEST as test_tasks,  # type: ignore
                )
            else:
                from tau_bench.envs.airline.tasks_test import (
                    TASKS as test_tasks,  # type: ignore
                )
            eval_rows = [_row(i, t) for i, t in enumerate(test_tasks)]
            hf_ds_eval = Dataset.from_list(eval_rows)
        except Exception:
            hf_ds_eval = hf_ds_train

        return hf_ds_train, hf_ds_eval

    # ------------------------------------------------------------------
    # Custom rollout to ensure user speaks before first assistant turn
    # ------------------------------------------------------------------

    async def rollout(
        self,
        client: AsyncOpenAI,
        model: str,
        prompt: List[Dict[str, Any]],
        answer: str,
        task: str = "0",
        info: Dict[str, Any] | None = None,
        sampling_args: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Generate a rollout where the *environment/user* speaks first.

        Sequence per turn:
            user (τ-Bench) → assistant (LLM)
        until τ-Bench signals ``done`` or ``max_turns`` is reached.
        """

        if info is None:
            info = {}
        if sampling_args is None:
            sampling_args = {}

        # Track reward + raw τ-Bench responses
        state: Dict[str, Any] = {"answer": answer, "task": task, "responses": []}

        messages = deepcopy(prompt)  # system + wiki only
        completion: List[Dict[str, Any]] = []

        # First environment (user) message
        env_msg, state = self.env_response(messages, state, **kwargs)
        print(f"ENV MESSAGE: {env_msg}")
        messages.append(env_msg)
        completion.append(env_msg)

        turn = 0
        while True:
            # Assistant step --------------------------------------------------
            content, response_obj = await self.get_model_response(
                prompt=messages,
                client=client,
                model=model,
                sampling_args=sampling_args,
                message_type=self.message_type,
                tools=self._tools_info,  # Pass full tool schema for auto-parsing
            )

            assistant_msg_full = response_obj.choices[0].message.model_dump()

            # Ensure content is a string
            if assistant_msg_full.get("content") is None:
                assistant_msg_full["content"] = ""

            # ------------------------------------------------------------------
            # Build the *public* assistant message
            # ------------------------------------------------------------------

            assistant_msg_public = assistant_msg_full.copy()

            xml_call = None
            if assistant_msg_public.get("tool_calls"):
                # Keep only the first tool call (τ-Bench supports 1-shot actions)
                assistant_msg_public["tool_calls"] = assistant_msg_public["tool_calls"][
                    :1
                ]

                fn_call = assistant_msg_public["tool_calls"][0]["function"]
                # Convert to XML wrapper so downstream rubrics / logging stay unchanged
                xml_call = (
                    "<tool_call>\n"
                    + json.dumps(
                        {
                            "name": fn_call["name"],
                            "arguments": json.loads(fn_call["arguments"]),
                        }
                    )
                    + "\n</tool_call>"
                )

            # Strip private reasoning tags from `content`
            assistant_msg_public = self.llm_parser.clean_assistant_message(
                assistant_msg_public
            )

            # This is what user/env sees: openai spec with clean content
            messages.append(assistant_msg_public)

            # This is what we use for RL signal. If assistant call a tool, use
            # <tool_call>{...}</tool_call> for RL; else we use <reasoning>...</reasoning> + plain text
            # TODO: This is tightly coupled to vLLM hermes parser
            completion.append(xml_call if xml_call else assistant_msg_full)

            turn += 1

            # Check termination after assistant reply
            if self.is_completed(messages, state, **kwargs) or turn >= self.max_turns:
                break

            # Environment (user) step ---------------------------------------
            env_msg, state = self.env_response(messages, state, **kwargs)
            messages.append(env_msg)
            completion.append(env_msg)

            if self.is_completed(messages, state, **kwargs):
                break

        return completion, state


__all__ = ["TauBenchEnv"]
