from .embedder import Embedder
from .runner import RunResult, StrategyRunner

__all__ = ["Embedder", "Retriever", "RunResult", "StrategyRunner"]


def __getattr__(name):
    if name == "Retriever":
        from .retriever import Retriever
        return Retriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
