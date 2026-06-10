# Patterns Applied

Production patterns lifted from a read of the Anthropic Python SDK and applied to
rag-starter to make it production-ready as the fork point for Project 1 (Docs Copilot).

**Source codebase:** anthropic-sdk-python, version 0.105.2
**Applied during:** W7E production codebase deep-dive
**Method:** read one production codebase end to end, extract concrete patterns, apply them
to rag-starter. Each pattern is its own commit. Line references in the deep-dive notes were
verified against the 0.105.2 source and can drift on upgrade.

Each entry below records the source (where the pattern appears in the SDK), what it replaced
in rag-starter, and why it improves the code.

---

## Pattern 1: Centralized client factory with configured retries

**Status:** applied, committed.

**Source.** The SDK constructs its client in `src/anthropic/_client.py` (`Anthropic.__init__`)
with explicit `max_retries` and `timeout`, with defaults living in `src/anthropic/_constants.py`
and the backoff math in `src/anthropic/_base_client.py` (`_calculate_retry_timeout`,
`_should_retry`). The SDK already implements exponential backoff with jitter, honors the
`Retry-After` header, and retries on 408, 409, 429, and 5xx. The retry loop is not something
the application needs to build.

**What it replaced.** Four inline `Anthropic()` constructions: one in `query.py`
(`generate_answer`) and three in `scorer.py` (one per scorer). Each picked up the SDK default
of two retries implicitly, with no shared or tunable configuration, and each module called
`load_dotenv()` separately.

**Why it improves the code.** A single factory, `get_anthropic_client()` in
`src/rag_starter/client.py`, is now the one place the client is constructed, with explicit and
tunable settings (`max_retries=4`, a 30 second read timeout with a 5 second connect timeout).
The key finding from reading the SDK is that this pattern is configure-and-centralize, not
implement-a-retry-loop, because the backoff already exists. The factory is the seam that later
resilience work hangs off without a refactor: per-tenant client construction for the
multi-tenant phase, and fallback or circuit-breaker behavior for the production resilience
phase.

**Files.** New: `src/rag_starter/client.py`. Modified: `query.py`, `scorer.py`.

---

## Pattern 3A: Typed Pydantic boundaries

**Status:** applied, committed. No eval regression.

**Source.** The SDK's response types all inherit from `BaseModel` in
`src/anthropic/_models.py`, which sets `model_config = ConfigDict(extra="allow", ...)`. That is
why an SDK call returns a typed object (for example `message.usage.input_tokens`) rather than a
raw dict. The important detail is the `extra="allow"` policy: the SDK deliberately keeps unknown
fields so that a server adding a new field never breaks an older client (forward compatibility).

**What it replaced.** The dicts and `TypedDict`s that crossed layer boundaries in rag-starter:
the `QueryResponse` `TypedDict`, chunk dicts (`{"text": ..., "source": ...}`), the scorer return
dicts (`dict[str, int | str]` parsed by hand from the judge's JSON), the dataset item dicts, and
the large results dict literal assembled in the eval runner.

**Why it improves the code.** Internal boundaries want the opposite policy from the SDK:
`extra="forbid"`, so a typo or a drifted shape fails loudly at the boundary instead of flowing
through silently. New models live in `src/rag_starter/models.py` (vanilla Pydantic v2
`BaseModel`, not the SDK's underscore-internal base):

- `Chunk` (`text`, `source`)
- `QueryResponse` (`answer`, `sources`, `chunks: list[Chunk]`, `trace_id`)
- `ScoreResult` (`score`, `reasoning`)
- `EvalItem` (`id`, `category`, `question`, `expected_answer`, plus `notes` and `pair_id`)
- `EvalResult` (the per-item result fields, with an optional `error` field and `int | None`
  scores)

Three concrete wins. First, validation happens at the edges, so malformed data fails where it
enters rather than several layers downstream. Second, `ScoreResult.model_validate_json(raw)`
replaces the hand-rolled `json.loads` plus `cast` in all three scorers, doing the parse and the
validation in one step. Third, and the largest long-term payoff, the retrieval layer now returns
`Chunk` models regardless of backend, so the planned ChromaDB to pgvector migration becomes a
storage-layer change behind a stable contract rather than a ripple through `query.py` and the
runner. The models are also FastAPI-native, so the request and response types needed at the
Project 1 fork are build-once rather than build-twice.

**What `extra="forbid"` caught.** On the first full run, validation rejected every dataset item
because `EvalItem` did not declare the `notes` and `pair_id` fields present in the data. This was
a real, previously silent mismatch between the dataset schema and the code's assumptions about
it. The strict boundary surfaced it immediately and forced an explicit decision (declare `notes`
as required, `pair_id` as optional since it appears only on paired items) rather than letting the
extra fields flow through unnoticed.

**Errored-item handling.** `EvalResult` carries `error: str | None` and types the three score
fields as `int | None`. This lets an item that fails (for example an API error after retries are
exhausted, or a judge response that fails to parse) be recorded as errored with no scores, rather
than being dropped or recorded as a misleading zero. The summary functions filter `None` scores
out of the quality averages and the divisor matches the filtered population, so a reliability
failure does not corrupt the quality metrics. The error path is wired in Pattern 2.

**Eval (structural change, expected to be within noise):**

| Metric | Before | After |
|---|---|---|
| Faithfulness | 4.95 | 4.95 |
| Relevance | 3.55 | 3.58 |
| Precision@3 | 0.53 | 0.53 |

n = 40 (30 standalone items plus 5 question pairs). The 0.03 movement in relevance is within
run-to-run judge variation. Faithfulness and Precision@3 are identical, confirming the change is
structural and did not alter retrieval or generation behavior.

**Files.** New: `src/rag_starter/models.py`. Modified: `query.py`, `scorer.py`, `runner.py`.

---

## Pattern 2: Typed error hierarchy

**Status:** PENDING (not yet applied). Placeholder; do not treat as complete.

**Source.** `src/anthropic/_exceptions.py`: one root (`AnthropicError(Exception)`), then
`APIError`, then status and connection branches, with each concrete status subclass pinning a
status code. The lesson to lift is that an exception is a place to carry structured context
(status, request id, body), not just a message string.

**What it will replace.** Two gaps. In `query.py`, `generate_answer` currently catches the API
error and returns the error text as the answer, so a failure masquerades as a valid answer to
every caller. In `scorer.py`, there is no protection around the judge call or its parse, so one
malformed judge response can end the whole eval pass.

**Plan.** New `src/rag_starter/errors.py` with a small categorical tree (`RAGError` base plus
`RetrievalError`, `GenerationError`, `ScoringError`, `ResponseParseError`). `generate_answer`
raises `GenerationError` instead of returning an error string. The scorers wrap their call and
parse and raise `ScoringError` carrying the raw text. The eval runner wraps each item so one
failure records an errored `EvalResult` (using the `error` field and `None` scores from Pattern
3A) and the loop continues. This is the change that makes the `int | None` score paths live.

_Fill in source detail, files, and eval numbers once applied._

---

## What these patterns do not do

None of these patterns touches the retrieval gap. The W5E baseline put Precision@3 at 0.53, and
that, not code structure, is the binding constraint on answer quality. These patterns make
rag-starter production-ready in the structural sense: robust under load, failing cleanly, typed
and maintainable, and clean to fork. Making it production-good in the quality sense (retrieving
the right chunks) is a separate workstream. "Patterns applied" should not be read as "retrieval
solved."
