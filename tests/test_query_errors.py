# =============================================================================
# test_query_errors.py
#
# Coverage for the error boundary: which failures a batch counts and continues
# through, which ones abort it, and the argument guards that would otherwise
# fail silently.
#
# The batch tests patch generate_answer rather than the Anthropic client, so one
# specific question can be made to fail on demand. Completion order is not
# guaranteed under concurrency, so the fake keys off the prompt text rather than
# call position.
# =============================================================================

from collections.abc import Callable, Coroutine
from unittest.mock import patch

import pytest
from chromadb import Collection

from rag_starter import query
from rag_starter.errors import GenerationError, RetrievalError

# Patched where it is looked up. main() calls generate_answer through the query
# module's namespace, so that is the name the patch has to replace.
GENERATE = "rag_starter.query.generate_answer"

QUESTIONS = ["What is Claude?", "What is ChromaDB?", "What is FastAPI?"]
FAILING_QUESTION = "What is ChromaDB?"


def _generate_failing_on_one(
    exc: BaseException,
) -> Callable[[str], Coroutine[None, None, str]]:
    async def generate(prompt: str) -> str:
        # The prompt carries the question verbatim under "User question:", and no
        # seeded document contains that phrasing, so this matches one query only.
        if FAILING_QUESTION in prompt:
            raise exc
        return "a fake answer"

    return generate


@pytest.mark.asyncio
async def test_ragerror_is_counted_and_batch_continues(
    seeded_collection: Collection,
    stub_langfuse_trace: object,
) -> None:
    # A RAGError is a failed question, not a broken run. It lands in failures
    # paired with its question, and the remaining queries still return.
    failure = GenerationError("simulated generation failure")

    with patch(GENERATE, _generate_failing_on_one(failure)):
        result = await query.main_batch(seeded_collection, QUESTIONS, max_concurrency=2)

    assert len(result.successes) == len(QUESTIONS) - 1
    assert len(result.failures) == 1

    question, exc = result.failures[0]
    assert question == FAILING_QUESTION
    assert isinstance(exc, GenerationError)


@pytest.mark.asyncio
async def test_unexpected_exception_aborts_the_batch(
    seeded_collection: Collection,
    stub_langfuse_trace: object,
) -> None:
    # Anything outside the RAGError hierarchy is a bug rather than a bad question,
    # so it is raised after the remaining tasks settle instead of being counted.
    # Paired with the test above, this pins the branch order: swap the two
    # isinstance checks and one of these two tests fails.
    with patch(GENERATE, _generate_failing_on_one(RuntimeError("a bug"))):
        with pytest.raises(BaseExceptionGroup) as excinfo:
            await query.main_batch(seeded_collection, QUESTIONS, max_concurrency=2)

    assert len(excinfo.value.exceptions) == 1
    assert isinstance(excinfo.value.exceptions[0], RuntimeError)


@pytest.mark.asyncio
async def test_main_batch_rejects_non_positive_concurrency(
    seeded_collection: Collection,
) -> None:
    # Semaphore(0) is legal and blocks forever, so this must raise rather than hang.
    with pytest.raises(ValueError, match="max_concurrency"):
        await query.main_batch(seeded_collection, QUESTIONS, max_concurrency=0)


def test_retrieve_chunks_rejects_non_positive_n_results(
    seeded_collection: Collection,
) -> None:
    # Chroma raises a bare TypeError from inside its own validation for this, so
    # the guard exists to name the argument the caller got wrong.
    with pytest.raises(ValueError, match="n_results"):
        query.retrieve_chunks(seeded_collection, "What is Claude?", n_results=0)


def test_retrieve_chunks_raises_on_empty_collection(
    empty_collection: Collection,
) -> None:
    # An unseeded store returns no chunks and no error, which would otherwise
    # produce a confident "I don't know" instead of a visible failure.
    with pytest.raises(RetrievalError):
        query.retrieve_chunks(empty_collection, "anything")
