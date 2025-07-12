from verifiers.parsers.xml_parser import XMLParser


class TauBenchParser(XMLParser):
    """Parser that only cares about the <reasoning> XML tag used in TauBench prompts."""

    def __init__(self):
        super().__init__(fields=["reasoning"])


__all__ = ["TauBenchParser"]
