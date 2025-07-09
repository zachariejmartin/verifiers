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

logger = logging.getLogger(__name__)

try:
    # Lazy import so that users who do *not* install the extra still work.
    import tau_bench  # type: ignore
    from tau_bench.envs.user import load_user  # type: ignore
    from tau_bench.run import load_env  # type: ignore
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
        user_model: str = "gpt-4o-mini",
        user_strategy: str = "llm",
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

        # Build underlying τ-Bench environment and user LLM simulator
        self._tau_env = load_env(domain)
        # tau_bench switched to `load_user` (see tau_bench.envs.user)
        self._user_sim = load_user(
            user_strategy=user_strategy,
            model=user_model,
            provider="openai",
        )

        self._task_iter = iter(self._tau_env.list_tasks())
        if task_ids is not None:
            self._task_iter = iter(task_ids)

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
        """Generate *user* turn via τ-Bench user simulator.

        Very simplified for the initial cut – we delegate to the built-in
        helper and mark the rollout *done* when the τ-Bench env reports
        it.  Proper tool execution, reward gathering and error handling
        will arrive in later commits.
        """
        if "tau_state" not in state:
            # First call → reset underlying env on the next available task.
            try:
                task_id = next(self._task_iter)
            except StopIteration:
                task_id = None  # type: ignore[assignment]
            state["task_id"] = task_id
            state["tau_state"] = self._tau_env.reset(task_id)
            logger.debug("Reset τ-Bench task %s", task_id)

        # Ask the user simulator for the next message given the current env state.
        # The exact API surface of build_user_simulator may evolve; keep a loose
        # wrapper for now.
        user_msg_str = self._user_sim.respond(state["tau_state"], messages)
        env_msg = {"role": "user", "content": user_msg_str}

        # Step τ-Bench env with the agent’s *previous* assistant message if any
        # so that `tau_state` stays in sync. For a first stub we simply echo the
        # last assistant content.
        if messages and messages[-1]["role"] == "assistant":
            assistant_content = messages[-1]["content"]
            state["tau_state"], reward, done, _ = self._tau_env.step(
                assistant_content, state["tau_state"]
            )
            state["reward"] = reward
            state["done"] = done
        return env_msg, state

    # ------------------------------------------------------------------
    # Convenience helpers (non-mandatory for MultiTurnEnv)
    # ------------------------------------------------------------------

    @property
    def tau_env(self):
        """Expose underlying τ-Bench environment (read-only)."""
        return self._tau_env

    @property
    def user_simulator(self):
        return self._user_sim


__all__ = ["TauBenchEnv"]
