class ConfigurationError(Exception):
    """Required configuration is missing or unusable.

    Deliberately NOT a RAGError. RAGError means "this item failed" and is caught
    per-item by the eval runner and by main_batch. Missing configuration is a
    run-level condition: under RAGError every eval item would fail identically
    and the run would write a results file full of errored items instead of
    aborting, and main_batch would report a config error as N failed questions.
    """


class RAGError(Exception):
    """Base for all rag-starter errors."""


class RetrievalError(RAGError):
    """Vector store query or chunk assembly failed."""


class GenerationError(RAGError):
    """LLM answer generation failed (after SDK retries exhausted)."""


class ScoringError(RAGError):
    """A judge call or its JSON parse failed."""


class ResponseParseError(RAGError):
    """LLM returned text that did not parse into the expected shape."""

    def __init__(self, message: str, *, raw: str | None = None) -> None:
        super().__init__(message)
        self.raw = raw
