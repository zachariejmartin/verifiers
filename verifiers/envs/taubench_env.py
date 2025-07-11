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

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

from datasets import Dataset
from openai import OpenAI

from verifiers.envs.multiturn_env import MultiTurnEnv

try:
    from tau_bench.envs.base import Action  # type: ignore
    from tau_bench.types import EnvResetResponse, EnvResponse, RewardResult

    RESPOND_ACTION_NAME = "respond"
except Exception:  # pragma: no cover
    Action = None  # type: ignore
    RESPOND_ACTION_NAME = "respond"

logger = logging.getLogger(__name__)

try:
    import tau_bench  # type: ignore
    from tau_bench.envs import Env, get_env  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    tau_bench = None  # type: ignore
    load_env = None  # type: ignore
    build_user_simulator = None  # type: ignore
    logger.warning(
        "tau-bench not found – TauBenchEnv is a stub. Did you install verifiers[all]?"
    )


class TauBenchEnv(MultiTurnEnv):
    """Wrap Airline / Retail τ-Bench environment into Verifiers API."""

    SUPPORTED_DOMAINS = {"airline", "retail"}

    def __init__(
        self,
        domain: str = "retail",
        task_split: str = "train",
        task_ids: (
            List[int] | None
        ) = None,  # TODO: pass the number of tasks you want to run
        max_turns: int = 10,
        **kwargs: Any,
    ):
        if domain not in self.SUPPORTED_DOMAINS:
            raise ValueError(f"domain must be one of {self.SUPPORTED_DOMAINS}")

        if tau_bench is None:
            raise ImportError(
                "TauBenchEnv requires the tau-bench extra. Install with 'uv add verifiers[all]'."
            )

        # Only allowing for OpenAI model that τ-Bench supports out-of-the-box.
        # Using a remote model for the user keeps the training stack simple
        # while the assistant still runs on vLLM / local GPUs.
        self._user_model_name = "gpt-4o-mini"
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
        self._tasks = tmp_env.tasks  # store Task objects for dataset rows / iteration

        hf_ds_train, hf_ds_eval = self._build_hf_datasets()

        super().__init__(
            dataset=hf_ds_train,
            eval_dataset=hf_ds_eval,
            max_turns=max_turns,
            message_type="chat",
            few_shot=[],
            mask_env_response=True,
            **kwargs,
        )

        # Iterator over Task objects
        self._task_iter = iter(task_ids) if task_ids is not None else iter(self._tasks)

        logger.info(
            "TauBenchEnv initialised (%s) with %s tasks",
            domain,
            len(self._tasks),
        )

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
            # Pick next task id (None → random) and reset env
            try:
                task_id = next(self._task_iter)
            except StopIteration:
                task_id = None  # let τ-Bench choose random task

            reset_res: EnvResetResponse = tau_env.reset(task_id)  # EnvResetResponse
            state["tau_state"] = reset_res
            state["task_id"] = task_id
            state["done"] = False

            return {"role": "user", "content": reset_res.observation}, state

        # ---------- SUBSEQUENT USER STEPS ----------
        if not messages:
            raise ValueError("Assistant message history is empty on user step")

        if Action is None:  # safety net if tau_bench missing
            raise RuntimeError("tau_bench Action class not available")

        assistant_content = messages[-1]["content"]
        action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": assistant_content})  # type: ignore[arg-type]

        tau_env: Env = state["tau_env"]  # type: ignore[assignment]
        step_res: EnvResponse = tau_env.step(action)

        # Update state with most recent τ-Bench response object
        state["tau_state"] = step_res
        state["done"] = step_res.done

        # If episode finished compute reward once and store
        if step_res.done:
            reward_res: RewardResult = tau_env.calculate_reward()
            state["reward"] = reward_res.reward

        return {"role": "user", "content": step_res.observation}, state

    # ------------------------------------------------------------------
    # Convenience helpers (non-mandatory for MultiTurnEnv)
    # ------------------------------------------------------------------

    def _build_hf_datasets(self):
        """Convert τ-Bench Task objects into minimal HF datasets."""

        def _row(idx, task):  # type: ignore[annassign]
            return {
                "prompt": [{"role": "user", "content": task.instruction}],
                "answer": "",  # reward is computed by τ-Bench, not by string match
                "task_id": idx,
                "info": {},
            }

        train_rows = [_row(i, t) for i, t in enumerate(self._tasks)]
        hf_ds_train = Dataset.from_list(train_rows)

        # For evaluation we load the official test split if available
        try:
            if self.domain == "retail":
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


__all__ = ["TauBenchEnv"]
