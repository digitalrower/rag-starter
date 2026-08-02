# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A RAG pipeline: ingest markdown docs → chunk → embed locally (Chroma, all-MiniLM-L6-v2) → store in a persistent Chroma collection → retrieve top-K chunks for a query → generate a grounded answer with Claude, constrained to answer only from context (else "I don't know"). Everything is traced in Langfuse. An LLM-as-judge eval harness scores faithfulness, relevance, and precision@3.

The deeper rationale, eval baselines, and the production-pattern history live in `README.md` and `patterns-applied.md`. Read those before changing retrieval, judging, or the typed/error boundaries.

## Setup & commands

The package uses a `src/` layout and is installed **editable**, which is why `python -m rag_starter.<x>` works from the repo root.

**Requires; Python >= 3.13**

```bash
pip install -r requirements.txt
pip install -e ".[dev]"
cp .env.example .env              # then set ANTHROPIC_API_KEY + LANGFUSE_* keys
```

`data/` holds a demonstration corpus of third-party documentation, committed under separate terms (see `data/NOTICE.md`), not MIT. Do not treat those files as project-authored content and do not modify them. The eval baseline in `README.md` and `evals/dataset.json` are both keyed to this corpus, so swapping it invalidates the published scores.

```bash
# Ingest (run once; collection persists to ./chroma_db/)
python -m rag_starter.ingest

# Query
python -m rag_starter.query "What are agent skills?"

# Eval: canonical local harness (writes evals/results/{results,summary}.json)
python -m evals.runner
python -m evals.runner --limit 5          # first N items
python -m evals.runner --ids 027 031      # specific dataset IDs (mutually exclusive with --limit)

# Eval: Langfuse Experiments (additive, for run-over-run UI comparison)
python -m scripts.seed_langfuse_dataset   # one-time: seed dataset.json into Langfuse Dataset "rag-starter-eval"
python -m evals.experiment --name run-1   # --name must be unique per run
```

### Lint / typecheck / tests (CI gates merge to `main` on these)

```bash
mypy src/ evals/ scripts/
ruff check src/ evals/ scripts/ tests/
ruff format --check src/ evals/ scripts/ tests/
ruff format src/ evals/ scripts/ tests/   # auto-fix formatting
pytest -v
```

CI (`.github/workflows/ci.yml`) runs these as two jobs (lint-and-typecheck, test) on every push. mypy is configured pragmatic-strict (`disallow_untyped_defs`), so every function needs full type hints. Note `tests/` is included in the ruff targets but not the mypy targets.

`tests/` covers the concurrency-sensitive parts of `query.py`, where correctness isn't obvious from reading the code: the `main_batch` dispatcher contract (one result per input, order preserved, no drops/dupes), the per-call `AsyncAnthropic` client being closed inside its owning event loop, and the `RAGError` vs. unexpected-exception error boundary. `tests/conftest.py` provides `seeded_collection` / `empty_collection` (in-memory Chroma, no network) and `stub_langfuse_trace`; `test_query_errors.py` patches `rag_starter.query.generate_answer` (not the Anthropic client) so a single question can be made to fail on demand, keying off prompt text since completion order isn't guaranteed under concurrency.

Runtime dependencies are pinned in `requirements.txt` only. `pyproject.toml` declares no `[project.dependencies]`, so `pip install -e ".[dev]"` installs the package and the dev extra but no runtime deps. Both commands above are required; neither is sufficient alone.

## Architecture & invariants

**Data flow:** `ingest.py` (load `./data/*.md` → `chunk_text` → Chroma `add`) writes the collection; `query.py` reads it (`retrieve_chunks` → `build_prompt` → `generate_answer`). `query.main()` returns a typed `QueryResponse`; the CLI prints `answer` + `sources`.

**The collection name `anthropic_docs` is the contract between ingest and query.** Hardcoded in `ingest.ingest_documents` (`get_or_create_collection`) and `query.get_collection` (`get_collection`). Changing it in one place silently breaks the other ("Collection not found").

**Centralized Anthropic client.** Always construct the client via `get_anthropic_client()` in `src/rag_starter/client.py` (sets `max_retries=4`, 30s read / 5s connect timeout). Never call `Anthropic()` inline. The SDK already does exponential backoff with jitter; this factory is the single seam for future per-tenant / fallback config.

**Typed boundaries (Pydantic v2, `extra="forbid"`).** All cross-layer data uses models in `src/rag_starter/models.py` (`Chunk`, `QueryResponse`, `ScoreResult`, `EvalItem`, `EvalResult`). `extra="forbid"` is deliberate and strict: adding a field to `evals/dataset.json` without declaring it on `EvalItem` makes the whole dataset fail to load. Update the model when the data shape changes.

**Typed error hierarchy.** `src/rag_starter/errors.py`: `RAGError` base → `RetrievalError`, `GenerationError`, `ScoringError`, `ResponseParseError`. `generate_answer` raises `GenerationError` rather than returning the error text as an answer (a past bug, an error string was being graded as a real answer). The eval runner catches `RAGError` **per item**, records an errored `EvalResult` (`error` set, scores `None`), and continues; errored items are excluded from quality averages (denominators match the filtered population). Do not add a broad `except Exception`. Unexpected bugs should fail loudly.

**Langfuse tracing is woven through every public function** via `langfuse.start_as_current_observation`. Production traffic (`query.main`) produces only `main_rag_query → retrieval / generation`; scorer spans appear only during eval runs. Eval scores are emitted as Langfuse score objects via `langfuse.create_score` (runner) / `Evaluation` (experiment), not just span output. **Always `langfuse.flush()` before a script exits** or trailing events are lost.

**`load_dotenv()` must run before any langfuse-touching import.** `evals/runner.py`, `evals/experiment.py`, `scripts/seed_langfuse_dataset.py`, and `scripts/benchmark_batch.py` call `load_dotenv()` at the very top, before importing `langfuse` or `rag_starter`. That ordering is why each carries an `E402` per-file ignore in `pyproject.toml`. Preserve it.

## Models & judging

- **Generation + all three judges use `claude-haiku-4-5-20251001`** (hardcoded in `query.generate_answer` and each scorer in `evals/scorer.py`).
- **Judges use structured outputs**, not free-text parsing: `client.messages.parse(..., output_format=ScoreResult)` at `temperature=0`, reading `response.parsed_output`. This is Anthropic-specific (constrained decoding) and replaced an earlier parse-and-strip approach that intermittently failed when the judge prefaced JSON with prose. `ScoreResult` is ordered `reasoning` then `score` on purpose so the judge reasons before committing to a number.
- **`runner.py` is the source of truth for the baseline** (`results.json` / `summary.json`); `experiment.py` is additive (Langfuse Experiments) and shares the same three scorers. Keep them consistent.

Default tuning knobs: `chunk_size=1500` / `overlap=200` (`ingest.py`), `n_results=3` / `max_tokens=500` (`query.py`). The standing finding (see README/`patterns-applied.md`) is that **retrieval is the weak link, not generation**, precision@3 is the metric to move.
