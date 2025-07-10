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

from openai import OpenAI

from verifiers.envs.multiturn_env import MultiTurnEnv

try:
    from tau_bench.envs.base import Action  # type: ignore

    RESPOND_ACTION_NAME = "respond"
except Exception:  # pragma: no cover
    Action = None  # type: ignore
    RESPOND_ACTION_NAME = "respond"

logger = logging.getLogger(__name__)

try:
    # Lazy import so that users who do *not* install the extra still work.
    import tau_bench  # type: ignore
    from tau_bench.envs import get_env  # type: ignore
    from tau_bench.envs.user import load_user  # type: ignore
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
        domain: str = "airline",
        task_ids: List[int] | None = None,
        # user_model_name: str | None = None,
        # user_strategy: str = "llm",
        max_turns: int = 10,
        **kwargs: Any,
    ):
        if domain not in self.SUPPORTED_DOMAINS:
            raise ValueError(f"domain must be one of {self.SUPPORTED_DOMAINS}")
        if tau_bench is None:
            raise ImportError(
                "TauBenchEnv requires the tau-bench extra. Install with 'uv add verifiers[all]'."
            )

        super().__init__(max_turns=max_turns, **kwargs)

        # Only allowing for OpenAI model that τ-Bench supports out-of-the-box.
        # Using a remote model for the user keeps the training stack simple
        # while the assistant still runs on vLLM / local GPUs.
        self._user_model_name = "gpt-4o-mini"
        self._user_strategy = "llm"

        # Build underlying τ-Bench environment and user LLM simulator
        self._tau_env = get_env(
            env_name=domain,
            user_strategy=self._user_strategy,
            user_model=self._user_model_name,
            user_provider="openai",
        )

        # τ-Bench task iterator (if caller provided explicit IDs use those)
        self._task_iter = (
            iter(task_ids) if task_ids is not None else iter(self._tau_env.list_tasks())
        )

        logger.info(
            "TauBenchEnv initialised (%s) with %s tasks",
            domain,
            len(self._tau_env.list_tasks()),
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

        # ---------- FIRST TURN (assistant hasn't acted yet) ----------
        if "tau_state" not in state:
            # Pick next task id (None → random) and reset env
            try:
                task_id = next(self._task_iter)  # may raise StopIteration
            except StopIteration:
                task_id = None  # let τ-Bench choose random task

            reset_res = self._tau_env.reset(task_id)  # EnvResetResponse
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

        step_res = self._tau_env.step(action)

        # Update state with most recent τ-Bench response object
        state["tau_state"] = step_res
        state["done"] = step_res.done
        state["reward"] = step_res.reward

        return {"role": "user", "content": step_res.observation}, state

    # ------------------------------------------------------------------
    # Convenience helpers (non-mandatory for MultiTurnEnv)
    # ------------------------------------------------------------------

    @property
    def tau_env(self):
        """Expose underlying τ-Bench environment (read-only)."""
        return self._tau_env

    # User simulator is internal to τ-Bench; no direct handle needed.


__all__ = ["TauBenchEnv"]
