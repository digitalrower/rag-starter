# =============================================================================
# benchmark_batch.py
#
# Sequential vs parallel benchmark for the rag-starter query pipeline. Times the
# full pipeline, not just raw API latency: retrieval (Chroma) + prompt build +
# Langfuse-traced generation + typed Pydantic boundaries, all intact. Runs the
# same N dataset questions two ways and prints both wall-clock times plus the
# speedup ratio.
#
#   sequential : await query.main once per question, one after another
#   parallel   : query.main_batch (asyncio.gather bounded by a semaphore)
#
# Both phases hit the same async `main`, so the ONLY variable is concurrency, not
# event-loop overhead. That keeps the published ratio honest.
#
# Measured result (40 dataset questions, max_concurrency 5):
#   sequential 91.13s | parallel 17.15s | 5.31x | 40 ok / 0 failed both phases.
# The speedup is bounded by design: concurrency is capped at 5 (not n), retrieval
# is synchronous so only generation overlaps, and tracing is paid on both sides.
# That is the number a real client workload would see, not a raw-API-call ceiling.
#
# Both phases emit Langfuse traces grouped by a per-phase session_id, so this run
# doubles as a tracing-under-concurrency check: each query must produce its own
# independent root trace, not collapse under a shared parent.
#
# Run from the project root as a module (so evals/ and rag_starter/ resolve):
#   python -m scripts.benchmark_batch                     # default cap 5
#   python -m scripts.benchmark_batch --max-concurrency 10
# =============================================================================

import argparse
import asyncio
import logging
import time
from datetime import UTC, datetime

from dotenv import load_dotenv

# load_dotenv must run before any import that initializes the Langfuse client,
# otherwise the client reads a missing key. Hence the imports split around it.
# (This placement is why ruff's E402 is exempted for this file in pyproject.)
load_dotenv()

from pathlib import Path

from chromadb import Collection
from langfuse import get_client

# Reused from the eval runner rather than re-declared, so the dataset loader and
# logging config have a single source of truth. Importing runner also triggers
# its module-level load_dotenv() as a side effect (harmless, already loaded).
from evals.runner import configure_logging, load_dataset
from rag_starter import query
from rag_starter.errors import RAGError

# get_client() returns the process-wide Langfuse singleton; we hold it only to
# call .flush() at the very end so background trace exports finish before exit.
langfuse = get_client()

logger = logging.getLogger(__name__)


async def run_sequential(
    collection: Collection, questions: list[str], session_id: str
) -> tuple[float, int, int]:
    # Baseline phase: await each query one at a time. This is the honest
    # comparator, same async main as the parallel phase, just no overlap. The
    # loop is where concurrency is deliberately withheld.
    successes = 0
    failures = 0
    start = time.perf_counter()  # perf_counter (monotonic), not time.time, for durations

    for q in questions:
        try:
            await query.main(collection, q, session_id=session_id, tags=["benchmark", "sequential"])
            successes += 1
        except RAGError:
            # Match the batch path's partial-failure tolerance: one bad question
            # is counted, not fatal, so both phases behave the same under failure.
            failures += 1
            logger.error(f"Run sequential question {q} failed")

    elapsed = time.perf_counter() - start
    return elapsed, successes, failures


async def run_parallel(
    collection: Collection, questions: list[str], session_id: str, max_concurrency: int
) -> tuple[float, int, int]:
    # Concurrent phase: main_batch fans all questions out through a semaphore-
    # bounded gather. The semaphore lives inside main_batch (bound to the running
    # loop); here we just hand it the cap. It returns a BatchResult already split
    # into successes and failures, so no per-item inspection is needed.
    start = time.perf_counter()
    result = await query.main_batch(
        collection,
        questions,
        session_id=session_id,
        tags=["benchmark", "parallel"],
        max_concurrency=max_concurrency,
    )
    elapsed = time.perf_counter() - start
    return elapsed, len(result.successes), len(result.failures)


async def run_benchmark(
    collection: Collection,
    questions: list[str],
    max_concurrency: int,
) -> None:
    # Distinct session ids per phase so the Langfuse dashboard shows two separate
    # trace groups (bench-seq-* and bench-par-*) that can be compared directly.
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    seq_session = f"bench-seq-{timestamp}"
    par_session = f"bench-par-{timestamp}"

    # Sequential first, then parallel, on the same warmed collection. Order can in
    # principle let the second phase benefit from OS disk cache; the effect is
    # small next to API latency but worth noting if the ratio is ever scrutinized.
    seq_time, seq_ok, seq_fail = await run_sequential(collection, questions, seq_session)
    par_time, par_ok, par_fail = await run_parallel(
        collection, questions, par_session, max_concurrency
    )

    # Guard against divide-by-zero if the parallel phase somehow reports ~0s.
    ratio = seq_time / par_time if par_time > 0 else float("inf")

    # max_concurrency is printed because the ratio is meaningless without it:
    # "5.31x" only means something paired with the cap it was measured at.
    print(f"n queries: {len(questions)}")
    print(f"sequential: {seq_time:.2f}s ({seq_ok} ok, {seq_fail} failed)")
    print(f"parallel: {par_time:.2f}s ({par_ok} ok, {par_fail} failed)")
    print(f"speedup: {ratio:.2f}x")
    print(f"max_concurrency: {max_concurrency}")


def positive_int(raw: str) -> int:
    # Cast arg to int and ensure it is >= 1.
    value = int(raw)  
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


if __name__ == "__main__":
    configure_logging()

    parser = argparse.ArgumentParser(description="Benchmark Batch for AI datasets")
    # Optional flag; the hyphen form maps to args.max_concurrency automatically.
    # Default 5 keeps the script runnable with no args (None would crash the
    # semaphore, which requires an int).
    parser.add_argument(
        "--max-concurrency",
        type=positive_int,
        default=5,
        help="Run at most N queries concurrently",
    )
    args = parser.parse_args()

    # Dataset lives in evals/, one level up from scripts/, so reach across.
    DATASET_PATH = Path(__file__).parent.parent / "evals" / "dataset.json"
    dataset = load_dataset(DATASET_PATH)

    # Extract just the question strings; both phases take a plain list[str].
    questions = [item.question for item in dataset]
    # questions = questions[:5]  # smoke-test slice: uncomment to prototype cheaply

    collection = query.get_collection()
    asyncio.run(run_benchmark(collection, questions, max_concurrency=args.max_concurrency))

    # Must be last: trace exports run in the background, and without an explicit
    # flush the script can exit before they land, silently losing traces.
    langfuse.flush()
