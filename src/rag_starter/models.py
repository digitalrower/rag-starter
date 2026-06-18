from pydantic import BaseModel, ConfigDict, Field


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    source: str


class QueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str
    sources: list[str]
    chunks: list[Chunk]
    trace_id: str


class RatingsScore(BaseModel):  # faithfulness, relevance (1-5)
    model_config = ConfigDict(extra="forbid")
    reasoning: str
    score: int = Field(ge=1, le=5)


class BinaryScore(BaseModel):  # precision (0 or 1)
    model_config = ConfigDict(extra="forbid")
    reasoning: str
    score: int = Field(ge=0, le=1)


class EvalItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    category: str
    question: str
    expected_answer: str
    notes: str | None = None
    pair_id: str | None = None


class EvalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    category: str
    question: str
    expected_answer: str
    actual_answer: str | None = None
    faithfulness_score: int | None = None
    faithfulness_reasoning: str | None = None
    relevance_score: int | None = None
    relevance_reasoning: str | None = None
    precision_score: int | None = None
    precision_reasoning: str | None = None
    sources: list[str] | None = None
    error: str | None = None
