# =============================================================================
# test_query_batch.py
#
# Coverage for the async query dispatcher and the sync wrapper's client
# lifecycle. Two tests, deliberately in two different event-loop regimes:
#
#   test_batch_returns_n_responses  (async): awaits main_batch directly, so it
#       runs inside the loop pytest-asyncio provides. Checks the dispatcher
#       contract: N questions in -> N QueryResponse out, no failures.
#
#   test_main_sync_closes_async_client  (sync): calls main_sync, which runs its
#       OWN asyncio.run() and therefore opens and closes its own loop. That is
#       the exact condition under which the per-call async client outlives its
#       loop, so this test asserts the client is closed explicitly. It is a plain
#       `def` on purpose: an async test would already hold a running loop, and
#       main_sync's asyncio.run() would then raise "cannot be called from a
#       running event loop", a different failure than the one under test.
#
# Both tests patch rag_starter.query.get_async_anthropic_client (patched where it
# is LOOKED UP, i.e. in the query module, not where it is defined in client.py)
# so the fake client is used and no real API call is made.
# =============================================================================

from unittest.mock import MagicMock, patch

import pytest
from chromadb import Collection

from rag_starter import query
from rag_starter.models import QueryResponse

# The dotted path we patch. query.py does
#   `from rag_starter.client import get_async_anthropic_client`
# which binds the name into the query module's namespace, so the patch has to
# target that bound name (query.get_async_anthropic_client), not the original in
# client.py. Patching the definition site would leave query's copy untouched.
CLIENT_FACTORY = "rag_starter.query.get_async_anthropic_client"


@pytest.mark.asyncio  # strict mode: async tests are opt-in via this marker
async def test_batch_returns_n_responses(
    seeded_collection: Collection,
    mock_async_client: MagicMock,
) -> None:
    questions = ["What is Claude?", "What is ChromaDB?", "What is FastAPI?"]

    # get_async_anthropic_client is called inside generate_answer for every query,
    # so every concurrent call in the batch receives the same fake client.
    with patch(CLIENT_FACTORY, return_value=mock_async_client):
        result = await query.main_batch(seeded_collection, questions, max_concurrency=2)

    # Contract: every question produced a success, nothing errored.
    assert len(result.successes) == len(questions)
    assert result.failures == []

    # Every success is the typed response object, not a coroutine or a dict.
    assert all(isinstance(r, QueryResponse) for r in result.successes)

    # NOTE on ordering: asyncio.gather preserves input order and main_batch keeps
    # that order, so successes correspond to questions positionally. We do not
    # assert per-item order by content here because the fake returns the same
    # answer for every question, so there is nothing to distinguish them by. Count
    # + type + no-failures is the meaningful contract at this fake fidelity.


def test_main_sync_closes_async_client(
    seeded_collection: Collection,
    mock_async_client: MagicMock,
) -> None:
    # Plain def (see module header): main_sync must own its asyncio.run loop for
    # this to reproduce the real teardown condition.
    with patch(CLIENT_FACTORY, return_value=mock_async_client):
        result = query.main_sync(seeded_collection, "What is Claude?")

    # main_sync must return a resolved QueryResponse, not a coroutine. This also
    # guards the sync-wrapper contract (item: "main_sync returns a QueryResponse,
    # not a coroutine, for a single query").
    assert isinstance(result, QueryResponse)

    # The lifecycle assertion that drives the teardown fix. The per-call async
    # client must be closed explicitly, inside the loop that owns it, rather than
    # left to garbage collection after asyncio.run has already closed that loop.
    #
    #   BEFORE the fix: generate_answer never calls aclose(), so this FAILS.
    #   AFTER the fix (await client.close() added in generate_answer): PASSES.
    mock_async_client.close.assert_awaited_once()


def test_main_returns_none_trace_id_when_tracing_disabled(
    seeded_collection: Collection,
    mock_async_client: MagicMock,
) -> None:
    # Plain def, not async: main_sync owns its own asyncio.run loop, and an async
    # test would already hold a running one (see the module header).
    #
    # Tracing is optional. With no LANGFUSE_* keys the client degrades to a no-op
    # tracer and get_current_trace_id() returns None, which used to trip an assert
    # in main() before any retrieval ran. That id must now flow through to
    # QueryResponse untouched rather than raising.
    #
    # The seam is patched rather than relying on the ambient environment, since
    # whether the real client is keyless depends on what the developer's shell
    # exports. Patching pins the condition under test either way. This works only
    # because _langfuse is @cache'd: get_client() itself returns a fresh facade
    # per call, so patching the object it returns would patch a throwaway.
    #
    # CLIENT_FACTORY is patched too, and must be: without it main() reaches the
    # real generate_answer, which calls get_async_anthropic_client and now raises
    # ConfigurationError when ANTHROPIC_API_KEY is unset. The subject here is the
    # trace id, not Anthropic auth.
    with (
        patch.object(query._langfuse(), "get_current_trace_id", return_value=None),
        patch(CLIENT_FACTORY, return_value=mock_async_client),
    ):
        result = query.main_sync(seeded_collection, "What is Claude?")

    assert isinstance(result, QueryResponse)
    assert result.trace_id is None
