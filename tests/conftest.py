# =============================================================================
# conftest.py
#
# Shared fixtures for the query-pipeline tests. They live here because both test
# files need them and pytest auto-discovers conftest fixtures without an import.
#
# There is deliberately no Langfuse trace stub. main() used to assert that
# get_current_trace_id() was not None, which forced every test through a patch;
# a None trace id is now a supported degraded mode, so tests run against the
# real (keyless, no-op) client and only the trace_id test patches that seam.
#
#   seeded_collection  : a real in-memory Chroma collection holding three tiny
#                        hand-written docs. Real Chroma behavior (so retrieval
#                        runs against a real collection) but deterministic and fast, because
#                        the corpus is three known sentences, not the shipped data
#                        set. Injected into main()/main_batch(), which take the
#                        collection as a parameter, so the disk-backed production
#                        collection is never touched.
#
#   mock_async_client  : a fake AsyncAnthropic client. Serves two purposes at
#                        once: it stops any real network call (messages.create is
#                        stubbed), and its aclose() is an AsyncMock, which records
#                        whether it was awaited. That recording is what the client
#                        -lifecycle test asserts on.
#
#   empty_collection   : an in-memory collection with nothing in it, for the
#                        unseeded-store branch of retrieval.
#
# The async methods (messages.create, aclose) are AsyncMock, not MagicMock,
# because the code under test awaits them; awaiting a plain MagicMock raises
# TypeError. The plain attribute reads (usage.input_tokens, etc.) stay MagicMock.
# =============================================================================

import uuid
from unittest.mock import AsyncMock, MagicMock

import chromadb
import pytest
from chromadb import Collection


@pytest.fixture
def seeded_collection() -> Collection:
    # EphemeralClient is fully in-memory: no ./chroma_db directory, no disk I/O,
    # and it starts empty on every test, so the collection is a clean slate.
    client = chromadb.EphemeralClient()

    # Unique name per test. EphemeralClient state is shared within a single
    # process, so a fixed name collides on the second test with "collection
    # already exists". A uuid suffix keeps every test's collection distinct.
    collection = client.create_collection(name=f"test_docs_{uuid.uuid4().hex[:8]}")

    # Three short, unrelated docs. Each metadata dict carries a "source", because
    # retrieve_chunks reads meta.get("source", ...) when it builds Chunk objects;
    # omitting it would still work (it defaults to "unknown") but seeding it keeps
    # the test data shaped exactly like production retrieval expects.
    collection.add(
        ids=["1", "2", "3"],
        documents=[
            "Claude is an AI assistant made by Anthropic.",
            "ChromaDB is a vector database for storing embeddings.",
            "FastAPI is a Python web framework for building APIs.",
        ],
        metadatas=[
            {"source": "claude.md"},
            {"source": "chroma.md"},
            {"source": "fastapi.md"},
        ],
    )

    return collection


@pytest.fixture
def mock_async_client() -> MagicMock:
    # ---- the fake message returned by messages.create() -------------------------
    # generate_answer reads several attributes off the returned message, so the
    # fake has to expose all of them or the code raises AttributeError before the
    # test can assert anything:
    #   message.content[0].text          -> the answer text
    #   message.stop_reason              -> only read on the empty-content path
    #   message.usage.input_tokens       -> passed to the Langfuse usage record
    #   message.usage.output_tokens      -> same
    fake_block = MagicMock()
    fake_block.text = "a fake answer"  # NOTE: this fake is not a real TextBlock,
    #                                    so generate_answer's isinstance(block,
    #                                    TextBlock) check is False and the stored
    #                                    answer ends up "". That is fine here:
    #                                    neither test asserts on answer content,
    #                                    only on shape (QueryResponse) and on the
    #                                    aclose lifecycle.

    fake_message = MagicMock()
    fake_message.content = [fake_block]  # non-empty list, so the "no content
    #                                      blocks" GenerationError path is skipped
    fake_message.stop_reason = "end_turn"
    fake_message.usage.input_tokens = 10  # plain int reads; MagicMock auto-creates
    fake_message.usage.output_tokens = 20  # the nested .usage attribute for us

    # ---- the fake client --------------------------------------------------------
    # The client container itself can be a MagicMock; only the two awaited methods
    # need to be AsyncMock.
    client = MagicMock()

    # messages.create is awaited in generate_answer -> AsyncMock. return_value is
    # the fake message above (AsyncMock returns it from the awaited call).
    client.messages.create = AsyncMock(return_value=fake_message)

    # close is the spy surface. It is an AsyncMock so that, after the code runs,
    # a test can call client.close.assert_awaited_once(). Right now the code
    # never calls it (that is the bug the lifecycle test drives), so that
    # assertion fails until the teardown fix adds `await client.close()`.
    # NOTE: the public async method on AsyncAnthropic is close(), not aclose();
    # aclose() lives on the inner httpx client that close() wraps.
    client.close = AsyncMock()

    return client


@pytest.fixture
def empty_collection() -> Collection:
    # A collection that exists but holds nothing. Chroma answers a query against
    # it with an empty result set rather than an error, so the code has to tell an
    # unseeded store apart from a query that legitimately matched nothing.
    client = chromadb.EphemeralClient()
    return client.create_collection(name=f"empty_docs_{uuid.uuid4().hex[:8]}")
