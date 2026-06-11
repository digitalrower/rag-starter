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
- `ScoreResult` (`score`, `reasoning`; reordered to `reasoning`, `score` in Pattern 4)
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

**Status:** applied, committed. Error path verified by forced failure.

**Source.** `src/anthropic/_exceptions.py`: one root (`AnthropicError(Exception)`), then
`APIError`, then status and connection branches, with each concrete status subclass pinning a
status code. Low-level httpx exceptions are caught in `_base_client.py`, wrapped with context,
and re-raised as typed SDK errors. The lesson lifted: exceptions are categorical and carry
structured context, and the application chains the SDK's errors (`raise ... from e`) rather
than copying their fields.

**What it replaced.** Two gaps. In `query.py`, `generate_answer` caught the API error and
returned the error text as the answer, so a failure masqueraded as a valid answer to every
caller, and the eval scorers then graded the error string. In `scorer.py`, there was no
protection around the judge call or its parse, so one malformed judge response could end the
whole eval pass.

**What was built.** New `src/rag_starter/errors.py` with a small categorical tree: `RAGError`
base plus `RetrievalError`, `GenerationError`, `ScoringError`, and `ResponseParseError` (the
one class with a custom field, carrying the raw text that failed to parse). `generate_answer`
raises `GenerationError` instead of returning an error string; the broad `except Exception`
was removed deliberately so unanticipated bugs fail loudly during development rather than
being mislabeled as generation failures. Each scorer wraps the judge call (`ScoringError`) and
the parse (`ResponseParseError`) in narrow try/excepts, both chained with `from e`. The eval
runner wraps each item in `except RAGError`, records an errored `EvalResult` (the `error`
field and `None` scores from Pattern 3A), marks the Langfuse span ERROR, and continues. The
summaries report the error count as a separate first-class metric and exclude errored items
from quality averages, with denominators matching the filtered population.

**Verification, and what it surfaced.** The error path was tested by forcing one scorer to
raise on a single item, then running the full eval. The run survived, the item was isolated
and reported, and the surviving items averaged correctly. The same verification run surfaced
two real findings. First, a latent divide-by-zero in both summary functions when a category
had zero scored items, now guarded. Second, and more significant: 13 of 40 items failed with
`ResponseParseError` because the judge model intermittently returns prose preambles ("I need
to evaluate whether...") instead of the requested JSON. The old untyped parse had been masking
this nondeterministic unreliability; the typed boundary made it visible and isolatable. The
fix (migrating the scorers to structured outputs) is Pattern 4 below.

**Eval note.** No before/after table for this pattern. The verification run is not comparable
to baseline because the judge-reliability issue contaminates it, and the clean comparison
belongs to Pattern 4, which changes the judging instrument itself.

**Files.** New: `src/rag_starter/errors.py`. Modified: `query.py`, `scorer.py`, `runner.py`.

---

## Pattern 4: Structured outputs for the judge (reliability fix)

**Status:** applied, committed. Clean full run, 0 errored items.

**Why it exists.** Pattern 2's verification exposed that the LLM judge does not reliably
follow "Return JSON only": at temperature 0, the same prompts intermittently yield prose
preambles or markdown-wrapped output, and on one full run 13 of 40 items failed to parse.
Prompt-level instructions cannot guarantee format. The API-level fix is structured outputs
(constrained decoding), which makes schema-conforming JSON the only output the model can emit
and eliminates the failure class rather than narrowing it.

**Source / reference.** Anthropic structured outputs (GA), Claude docs, Build with Claude,
structured-outputs page. In SDK 0.105.2, verified against the local clone:
`client.messages.parse(...)` (`resources/messages/messages.py`) accepts
`output_format=<PydanticModel>`, derives the JSON schema from the model via
`pydantic.TypeAdapter`, and returns a `ParsedMessage`. The parsed object is read from the
`ParsedMessage.parsed_output` property (`types/parsed_message.py`), typed
`Optional[ResponseFormatT]`: it returns the validated model or `None` if no parseable
structured output is present.

**What was built.** All three scorers migrated from `client.messages.create(...)` plus
fence-stripping plus `ScoreResult.model_validate_json(...)` to
`client.messages.parse(..., output_format=ScoreResult)`. The fence-strip blocks and the manual
parse were deleted, along with the now-unused `ValidationError` and `TextBlock` imports. The
parse-failure path changed from catching a `ValidationError` to checking
`response.parsed_output is None` and raising `ResponseParseError` (whose `raw` field was relaxed
to optional, since the `None` path has no clean raw text to carry). The `APIError` try/except
around the call was kept unchanged. The redundant "Return JSON only" sentence was dropped from
each system prompt, since the schema now enforces format at the decode level. `ScoreResult`
field order was reversed to `reasoning` then `score`, so under in-order constrained decoding the
judge writes its rationale before committing to a number rather than after.

**New canonical baseline.** The first run after this migration recorded a misleading
faithfulness number due to a separate scorer bug found immediately afterward (see below); the
corrected baseline is the one that stands.

| Metric | Baseline |
|---|---|
| Faithfulness | 4.95 |
| Relevance | 3.45 |
| Precision@3 | 0.45 |

n = 40, 0 errored items. Faithfulness is high and uniform (38 of 40 items score 5, the rest 4),
matching the standing finding that generation is strong and retrieval is the weak link.
Precision@3 at 0.45 is that retrieval weakness. These numbers are close to the earlier W5E
free-text-judge read (4.95 / 3.55 / 0.53), which is the expected outcome: faithfulness is
identical and the small relevance and precision differences are ordinary run-to-run and
instrument variation, not a regression. The structured-output migration did not move the quality
scores; its value is reliability (eliminating the 13-of-40 parse failures), not a score change.

**Scorer bug found and fixed during re-baselining (honesty log).** The very first post-migration
run reported faithfulness 3.40, and an initial read misattributed that drop to the new judge
"scoring more decisively." That was wrong. The real cause was a bug introduced earlier the same
day when the scorer prompt strings were rewrapped to satisfy line-length limits:
`score_faithfulness` was passing `expected_answer` to the judge instead of `generated_answer`,
so it had been grading whether the golden answer was grounded in the context rather than whether
the model's actual answer was. A one-line fix (`generated_answer` in the user message) restored
the correct measurement, and faithfulness returned to 4.95. Lesson reinforced: a passing eval
with in-range numbers can still be measuring the wrong thing; read the judge reasoning, not just
the score. The corrected run is the baseline above.

**Provider coupling, logged.** `messages.parse` / structured outputs is Anthropic-specific API
surface. Acceptable per the standing posture (Claude is the default for sellable artifacts; the
judge stays Claude). Invoice-to-JSON (Project 2) is built on this same technique, so this is a
head start, not a one-off.

**Files.** Modified: `src/rag_starter/models.py` (field reorder), `src/rag_starter/errors.py`
(`raw` optional), `evals/scorer.py` (all three scorers migrated to `messages.parse`; also the
`generated_answer` fix in `score_faithfulness`).

---

## What these patterns do not do

None of these patterns touches the retrieval gap. The W5E baseline put Precision@3 at 0.53, and
that, not code structure, is the binding constraint on answer quality. These patterns make
rag-starter production-ready in the structural sense: robust under load, failing cleanly, typed
and maintainable, and clean to fork. Making it production-good in the quality sense (retrieving
the right chunks) is a separate workstream. "Patterns applied" should not be read as "retrieval
solved."
