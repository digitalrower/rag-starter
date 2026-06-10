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

    def __init__(self, message: str, *, raw: str) -> None:
        super().__init__(message)
        self.raw = raw
