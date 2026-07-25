import asyncio
import logging
import sys
from typing import cast

import chromadb
from anthropic import APIError
from anthropic.types import TextBlock
from chromadb import Collection
from chromadb.errors import ChromaError
from langfuse import get_client, propagate_attributes

from rag_starter.client import get_async_anthropic_client
from rag_starter.errors import GenerationError, RAGError, RetrievalError
from rag_starter.models import BatchResult, Chunk, QueryResponse

# from dotenv import load_dotenv
# load_dotenv()

# langfuse: initialize langfuse client
langfuse = get_client()

# logging
logger = logging.getLogger(__name__)


def get_collection() -> Collection:
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_collection(name="anthropic_docs")


def retrieve_chunks(collection: Collection, q: str, n_results: int = 3) -> list[Chunk]:
    with langfuse.start_as_current_observation(
        as_type="span", name="retrieval", input={"query": q, "n_results": n_results}
    ) as span:
        try:
            results = collection.query(query_texts=[q], n_results=n_results)
        except ChromaError as e:
            # ChromaError is the base for every chromadb exception. Wrapping as
            # RAGError-family makes a vector-store failure a counted failure the
            # batch tolerates, rather than a bug that aborts the whole run.
            error_msg = f"Retrieval failed: {e}"
            logger.error(error_msg)
            span.update(level="ERROR", status_message=error_msg)
            raise RetrievalError(error_msg) from e

        # Outside the try on purpose: a results dict with an unexpected shape is a
        # contract violation, not an operational failure, and should crash loudly.
        docs = cast(list[list[str]], results["documents"])[0]
        metas = cast(list[list[dict[str, str]]], results["metadatas"])[0]

        chunks = []
        for doc, meta in zip(docs, metas, strict=False):
            chunks.append(Chunk(text=doc, source=meta.get("source", "unknown")))

        span.update(output=chunks)
        logger.info(f"Retrieval complete: found {len(chunks)} chunks.")
        return chunks


def build_prompt(user_question: str, chunks: list[Chunk]) -> str:
    system_prompt = (
        "You are a helpful assistant answering questions about Anthropic's documentation. "
        "Answer ONLY from the provided context. "
        "If the context doesn't contain the answer, say "
        "'I don't know based on the provided documentation.'"
    )
    context = "Context:\n"
    for i, item in enumerate(chunks):
        context += f" [{i}] (from {item.source}): {item.text}"

    prompt = system_prompt + "\n\n" + context + "\n\nUser question: " + user_question
    return prompt


async def generate_answer(prompt: str) -> str:
    model_name = "claude-haiku-4-5-20251001"
    with langfuse.start_as_current_observation(
        as_type="generation",
        name="generation",
        model=model_name,
        input={"messages": [{"role": "user", "content": prompt}]},
    ) as gen:
        client = get_async_anthropic_client()

        # COST NOTE: Anthropic prompt caching (cache_control breakpoints) is not
        # applied to this system prompt + retrieved context yet. At the current
        # corpus scale, caching adds complexity without a measurable token baseline
        # to justify it. Revisit once there is a baseline to measure against.
        try:
            message = await client.messages.create(
                model=model_name,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )

            if not message.content:
                error_msg = f"API returned no content blocks (stop_reason={message.stop_reason})"
                logger.error(f"Generation failed: {error_msg}")
                gen.update(level="ERROR", status_message=error_msg)
                raise GenerationError(error_msg)

            block = message.content[0]
            output_text = block.text if isinstance(block, TextBlock) else ""

            # langfuse: record successful output, and optionally token usage if parsed
            gen.update(
                output=output_text,
                usage={
                    "input": message.usage.input_tokens,
                    "output": message.usage.output_tokens,
                },
            )

            # log: info when generation completes (answer produced)
            logger.info("Generation complete: answer successfully produced.")

            return output_text

        except APIError as e:
            error_msg = f"Couldn't reach Claude: {e}"
            logger.error(f"Anthropic API call failed: {error_msg}")
            gen.update(level="ERROR", status_message=error_msg)
            raise GenerationError(error_msg) from e
        finally:
            await client.close()


async def main(
    collection: Collection,
    user_question: str,
    session_id: str | None = None,
    tags: list[str] | None = None,
) -> QueryResponse:
    # log: INFO when a query is received (start of the request)
    logger.info(f"Query received: '{user_question}'")

    with langfuse.start_as_current_observation(
        as_type="span", name="main_rag_query", input={"user_question": user_question}
    ) as span:
        # langfuse: capture the trace id
        trace_id = langfuse.get_current_trace_id()
        assert trace_id is not None, "trace_id should never be None inside an active span"

        if session_id or tags:
            with propagate_attributes(session_id=session_id, tags=tags or []):
                chunks = retrieve_chunks(collection, user_question)
                prompt = build_prompt(user_question, chunks)
                answer = await generate_answer(prompt)
        else:
            chunks = retrieve_chunks(collection, user_question)
            prompt = build_prompt(user_question, chunks)
            answer = await generate_answer(prompt)

        sources: list[str] = list(dict.fromkeys(item.source for item in chunks))
        response = QueryResponse(
            answer=answer,
            sources=sources,
            chunks=chunks,
            trace_id=trace_id,
        )

        span.update(output=response.model_dump())

    return response


def main_sync(
    collection: Collection,
    user_question: str,
    session_id: str | None = None,
    tags: list[str] | None = None,
) -> QueryResponse:
    return asyncio.run(main(collection, user_question, session_id, tags))


async def main_batch(
    collection: Collection,
    questions: list[str],
    session_id: str | None = None,
    tags: list[str] | None = None,
    max_concurrency: int = 5,
) -> BatchResult:

    # Semaphore(0) is legal and starts with zero permits, so every task blocks on
    # a release that never comes: a silent hang, not an error. Guard here rather
    # than only at the CLI, since this is the layer that builds the semaphore.
    if max_concurrency < 1:
        raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")

    semaphore = asyncio.Semaphore(max_concurrency)

    async def run_one(q: str) -> QueryResponse:
        async with semaphore:
            return await main(collection, q, session_id=session_id, tags=tags)

    coroutines = [run_one(q) for q in questions]

    # return_exceptions=True keeps one bad question from cancelling its siblings.
    # Cost: gather captures bugs too, so the partition below sorts them back apart.
    results = await asyncio.gather(*coroutines, return_exceptions=True)

    successes: list[QueryResponse] = []
    failures: list[tuple[str, BaseException]] = []
    unexpected: list[BaseException] = []

    for question, outcome in zip(questions, results, strict=True):
        # Cancellation is neither a failure nor a bug, and must propagate bare and
        # immediately. Wrapped in a group it stops reading as cancellation, so
        # asyncio.timeout never converts it and TaskGroup treats it as a crash.
        if isinstance(outcome, asyncio.CancelledError):
            raise outcome

        # RAGError before the broad check: it is itself a BaseException, so this
        # order decides whether a bad question is counted or crashes the batch.
        if isinstance(outcome, RAGError):
            failures.append((question, outcome))
            logger.error(f"Batch run failed: {question}: {outcome}")
        elif isinstance(outcome, BaseException):
            unexpected.append(outcome)
            logger.error(f"Unexpected exception for {question}", exc_info=outcome)
        else:
            successes.append(outcome)

    if unexpected:
        # Raise rather than return a partial result: a bug hitting every query
        # would otherwise read as "N failures" to the caller.
        # BaseExceptionGroup, not ExceptionGroup: the latter raises TypeError on
        # members that are BaseException but not Exception.
        raise BaseExceptionGroup(
            f"{len(unexpected)} unexpected exception(s) during batch run", unexpected
        )

    return BatchResult(successes=successes, failures=failures)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    if len(sys.argv) > 1:
        user_question = " ".join(sys.argv[1:])

        collection = get_collection()
        result = asyncio.run(main(collection, user_question))

        print("\nAnswer:", result.answer)
        print("\nSources:", result.sources)

        # langfuse: ensure all background events are sent before script exists
        langfuse.flush()

    else:
        print("Error: No string provided.")
