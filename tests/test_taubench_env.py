"""Basic smoke-tests for the TauBenchEnv wrapper.

We stub-out the external `tau_bench` package so the tests can run in
isolation without the real dependency.
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import Dict, List, Tuple

import pytest

# ---------------------------------------------------------------------------
# Stub-out the tau_bench package ------------------------------------------------
# ---------------------------------------------------------------------------


def _install_stub_tau_bench() -> None:
    """Inject minimal stub modules for `tau_bench` into sys.modules."""
    tau_bench = types.ModuleType("tau_bench")
    envs_mod = types.ModuleType("tau_bench.envs")
    base_mod = types.ModuleType("tau_bench.envs.base")
    types_mod = types.ModuleType("tau_bench.types")

    # --- τ-Bench type stubs ----------------------------------------------------
    class StubEnvResetResponse:
        def __init__(self, observation: str):
            self.observation = observation

    class StubEnvResponse:
        def __init__(self, observation: str, done: bool = False):
            self.observation = observation
            self.done = done

    class StubRewardResult:
        def __init__(self, reward: float):
            self.reward = reward

    # --- Action stub -----------------------------------------------------------
    class Action:
        def __init__(self, name: str, kwargs: Dict):
            self.name = name
            self.kwargs = kwargs

    # --- Environment stub ------------------------------------------------------
    class StubEnv:
        """Mimic the public surface used by TauBenchEnv."""

        def __init__(self, *_, **__):
            self.tasks = [f"task-{i}" for i in range(3)]
            self.wiki = "Stub wiki"

        # First observation
        def reset(self, *_):
            return StubEnvResetResponse("Hello from stub user")

        # Subsequent observation + finish signal
        def step(self, _action: Action):
            return StubEnvResponse("Good-bye from stub user", done=True)

        # Reward calculation at end of episode
        def calculate_reward(self):
            return StubRewardResult(0.8)

    # Factory used by TauBenchEnv
    def get_env(*_, **__) -> StubEnv:
        return StubEnv()

    # Wire everything together
    envs_mod.Env = StubEnv
    envs_mod.get_env = get_env
    base_mod.Action = Action
    types_mod.EnvResetResponse = StubEnvResetResponse
    types_mod.EnvResponse = StubEnvResponse
    types_mod.RewardResult = StubRewardResult

    # Register stubs so regular imports succeed
    sys.modules.update(
        {
            "tau_bench": tau_bench,
            "tau_bench.envs": envs_mod,
            "tau_bench.envs.base": base_mod,
            "tau_bench.types": types_mod,
        }
    )
    tau_bench.envs = envs_mod  # make `tau_bench.envs` attribute available


# ---------------------------------------------------------------------------
# PyTest fixtures ------------------------------------------------------------
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def stub_tau_bench(monkeypatch):
    """Automatically stub `tau_bench` and reload TauBenchEnv for each test."""
    _install_stub_tau_bench()

    # Reload the module so it picks up our stub before use
    import verifiers.envs.taubench_env as te  # import after stubbing

    importlib.reload(te)
    yield

    # Clean-up so other test modules are unaffected
    for mod_name in [
        "tau_bench",
        "tau_bench.envs",
        "tau_bench.envs.base",
        "tau_bench.types",
    ]:
        sys.modules.pop(mod_name, None)


@pytest.fixture
def TauBenchEnvClass():
    """Convenience fixture returning the (reloaded) TauBenchEnv class."""
    from verifiers.envs.taubench_env import TauBenchEnv  # import after stub

    return TauBenchEnv


# ---------------------------------------------------------------------------
# Tests ----------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_initialisation(TauBenchEnvClass):
    env = TauBenchEnvClass(domain="retail", max_turns=5)
    assert env._domain == "retail"
    assert env.max_turns == 5
    assert env._wiki == "Stub wiki"
    # `_tasks` were provided by the stub environment
    assert len(env._tasks) == 3
    # Dataset built from tasks should mirror task count
    assert len(env.get_dataset()) == 3


def test_first_env_response_generates_user_message(TauBenchEnvClass):
    env = TauBenchEnvClass(domain="retail")
    messages: List[Dict] = []
    state: Dict = {}

    env_msg, new_state = env.env_response(messages, state)

    assert env_msg["role"] == "user"
    assert "Hello" in env_msg["content"]
    assert "tau_state" in new_state  # τ-Bench state stored
    assert new_state["done"] is False


def test_second_env_response_completes_episode_and_sets_reward(TauBenchEnvClass):
    env = TauBenchEnvClass(domain="retail")
    messages: List[Dict] = []
    state: Dict = {}

    # First user turn (env.reset)
    env_msg, state = env.env_response(messages, state)
    messages.append(env_msg)

    # Assistant replies
    messages.append({"role": "assistant", "content": "Assistant reply"})

    # Second user turn (env.step)
    env_msg2, state = env.env_response(messages, state)

    assert env_msg2["role"] == "user"
    assert "Good" in env_msg2["content"]
    # Episode should now be finished and reward computed
    assert state["done"] is True
    assert state["reward"] == 0.8
