from __future__ import annotations

from typing import Any, Callable, Dict, List, Union

from verifiers import RewardFunc
from verifiers.parsers.taubench_parser import TauBenchParser
from verifiers.rubrics.rubric import Rubric

# -----------------------------------------------------------------------------
# Reward helpers
# -----------------------------------------------------------------------------


def env_reward(*, state: Dict[str, Any], **_ignored) -> float:  # noqa: D401
    """Return the scalar reward computed by the wrapped τ-Bench environment.

    Tau-Bench stores the score in ``state['reward']`` (set by TauBenchEnv after
    ``tau_env.calculate_reward()``).  If the key is missing we return 0.0 so the
    training loop doesn't crash.
    """
    return float(state.get("reward", 0.0))


class TauBenchRubric(Rubric):
    """Reward = env-provided score  +  formatting bonus for <reasoning> tags."""

    def __init__(self, reasoning_weight: float = 0.2, env_weight: float = 1.0):
        parser = TauBenchParser()

        # format reward helper supplied by XMLParser
        format_func: RewardFunc = parser.get_format_reward_func()
        format_func.__name__ = "reasoning_format"  # nicer key in results

        funcs: List[RewardFunc] = [env_reward, format_func]
        weights: List[float] = [env_weight, reasoning_weight]

        super().__init__(funcs=funcs, weights=weights, parser=parser)


__all__ = ["TauBenchRubric"]
