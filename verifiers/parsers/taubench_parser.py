from __future__ import annotations

import re
from typing import Any, Dict

from verifiers.parsers.smola_parser import SmolaParser


class TauBenchParser(SmolaParser):
    """
    τ-Bench parser that recognises internal tags such as <reasoning> … </reasoning>
    but can also remove them before the message is shown to the simulated user.
    """

    def __init__(self) -> None:
        # Parse <reasoning> blocks (private) *and* tool call XML tags.
        # - "reasoning" → stripped before surfacing to the user.
        # - ("tool", "tool_call") → retained so downstream logic can
        #   extract the first tool invocation JSON.
        super().__init__(fields=["reasoning", ("tool", "tool_call")])

    # ------------------------------------------------------------------ #
    # Public utilities
    # ------------------------------------------------------------------ #
    def strip_private_tags(self, text: str) -> str:
        """
        Remove *all* tags defined in the schema (currently just <reasoning>)
        together with their contents.  The remaining text is trimmed and
        returned unchanged.
        """
        # Only hide *private* tags (currently just <reasoning>).  Tool tags
        # must remain so they can be parsed and forwarded to τ-Bench.
        for canonical, alternatives in self._fields:
            if canonical != "reasoning":
                continue
            for tag in alternatives:
                # lazy DOTALL to remove the whole block – DOTALL so newline spans are removed too
                text = re.sub(rf"<{tag}>\s*.*?\s*</{tag}>", "", text, flags=re.DOTALL)
        return text.strip()

    def clean_assistant_message(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return a shallow copy of an assistant message whose `content` has had
        private tags stripped out.  Any `tool_calls` structure is left intact.
        """
        cleaned = msg.copy()
        if isinstance(cleaned.get("content"), str):
            cleaned["content"] = self.strip_private_tags(cleaned["content"])
        return cleaned


__all__ = ["TauBenchParser"]
