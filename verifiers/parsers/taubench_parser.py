from verifiers.parsers.smola_parser import SmolaParser


class TauBenchParser(SmolaParser):
    """Parser that only cares about the <tool> XML tag used in TauBench prompts."""

    def __init__(self):
        super().__init__(fields=["tool"])


__all__ = ["TauBenchParser"]
