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


def planning_reward(
    completion: List[Dict[str, str]], parser: TauBenchParser | None = None, **_
) -> float:
    """
    Return 1.0 if the first assistant message includes a valid <reasoning> block,
    otherwise 0.0.

    We pass a parser instance explicitly so the function stays pure and is
    easy to unit-test.
    """
    if parser is None:
        parser = TauBenchParser()

    for msg in completion:
        # If the first assistant message contains <reasoning>
        if msg.get("role") == "assistant":
            parsed = parser.parse(msg["content"])
            if hasattr(parsed, "reasoning") and parsed.reasoning is not None:
                return 1.0
            return 0.0


class TauBenchRubric(Rubric):
    """Reward = env-provided score  +  formatting bonus for <reasoning> tags."""

    def __init__(self, reasoning_weight: float = 0.2, env_weight: float = 1.0):
        parser = TauBenchParser()

        def _planning_reward(completion, **kw):
            return planning_reward(completion, parser=parser)

        # nicer name in output
        _planning_reward.__name__ = "planning_reward"

        funcs: List[RewardFunc] = [env_reward, _planning_reward]
        weights: List[float] = [env_weight, reasoning_weight]

        super().__init__(funcs=funcs, weights=weights, parser=parser)


__all__ = ["TauBenchRubric"]
