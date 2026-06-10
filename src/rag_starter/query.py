import logging
import sys
from typing import cast

import chromadb
from anthropic import APIError
from anthropic.types import TextBlock
from chromadb import Collection
from langfuse import get_client, propagate_attributes

from rag_starter.client import get_anthropic_client
from rag_starter.models import Chunk, QueryResponse

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
        results = collection.query(query_texts=[q], n_results=n_results)
        docs = cast(list[list[str]], results["documents"])[0]
        metas = cast(list[list[dict[str, str]]], results["metadatas"])[0]

        chunks = []
        for doc, meta in zip(docs, metas, strict=False):
            chunks.append(Chunk(text=doc, source=meta.get("source", "unknown")))

        # langfuse: return output and end the span
        span.update(output=chunks)

        # log: info when retrieval completes (chunks found)
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


def generate_answer(prompt: str) -> str:
    model_name = "claude-haiku-4-5-20251001"
    with langfuse.start_as_current_observation(
        as_type="generation",
        name="generation",
        model=model_name,
        input={"messages": [{"role": "user", "content": prompt}]},
    ) as gen:

        client = get_anthropic_client()

        # COST NOTE: Anthropic prompt caching (cache_control breakpoints) will be
        # applied to this system prompt + retrieved context at W13 once we have
        # a measurable token baseline. At W4 scale, caching adds complexity without
        # a benchmark to justify it. See W13 hardening sprint.
        try:
            message = client.messages.create(
                model=model_name,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )

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

            # log: ERROR if the Anthropic API call fails
            logger.error(f"Anthropic API call failed: {error_msg}")

            gen.update(level="ERROR", status_message=error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            gen.update(level="ERROR", status_message=error_msg)
            return error_msg


def main(
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
                answer = generate_answer(prompt)
        else:
            chunks = retrieve_chunks(collection, user_question)
            prompt = build_prompt(user_question, chunks)
            answer = generate_answer(prompt)

        sources: list[str] = list(dict.fromkeys(item.source for item in chunks))
        response = QueryResponse(
            answer=answer,
            sources=sources,
            chunks=chunks,
            trace_id=trace_id,
        )

        span.update(output=response.model_dump())

    return response


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    if len(sys.argv) > 1:
        user_question = " ".join(sys.argv[1:])

        collection = get_collection()
        result = main(collection, user_question)

        print("\nAnswer:", result.answer)
        print("\nSources:", result.sources)

        # langfuse: ensure all background events are sent before script exists
        langfuse.flush()

    else:
        print("Error: No string provided.")
