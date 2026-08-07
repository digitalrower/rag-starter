import logging
import os

import httpx
from anthropic import Anthropic, AsyncAnthropic

from rag_starter.errors import ConfigurationError

DEFAULT_TIMEOUT = httpx.Timeout(timeout=30.0, connect=5.0)
DEFAULT_MAX_RETRIES = 4

ANTHROPIC_KEY_VAR = "ANTHROPIC_API_KEY"
LANGFUSE_KEY_VARS = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")

logger = logging.getLogger(__name__)


def _require_anthropic_key() -> None:
    """Fail with an actionable error before the SDK's opaque one.

    The Anthropic SDK resolves auth lazily: constructing a client with no key
    succeeds, and the failure surfaces at request time as a bare TypeError
    ("Could not resolve authentication method") that `except APIError` does not
    catch. Checking here turns that into a named, typed error at the point the
    client is built.
    """
    if not os.environ.get(ANTHROPIC_KEY_VAR):
        raise ConfigurationError(
            f"{ANTHROPIC_KEY_VAR} is not set. Copy .env.example to .env and add your key "
            f"(get one at console.anthropic.com), or export {ANTHROPIC_KEY_VAR} directly. "
            "Note that .env is only read by the CLI entry points, not on import."
        )


def preflight_env() -> None:
    """Check configuration once, at entry-point startup, before doing real work.

    Warnings are emitted before the hard failure, deliberately: a fresh checkout
    is typically missing every key at once, and raising first would report the
    Anthropic key, then only reveal the Langfuse warning on the next run. One
    pass should surface every configuration problem.

    Tracing is optional, so missing Langfuse keys only warn -- but they warn
    loudly, because a keyless eval run otherwise writes zero scores and still
    looks successful. A missing Anthropic key is fatal, and failing here puts
    that error before retrieval and before the one-time embedding-model
    download rather than after.

    Env vars are read directly rather than asking the Langfuse client whether it
    is enabled: on the keyless path it keeps _tracing_enabled True and only swaps
    in a no-op tracer, and auth_check() is a blocking network call its own
    docstring discourages.
    """
    missing = [var for var in LANGFUSE_KEY_VARS if not os.environ.get(var)]
    if missing:
        logger.warning(
            "Langfuse tracing disabled: %s not set. Queries will run and answer "
            "normally, but no traces will be recorded and eval runs will produce "
            "no scores.",
            ", ".join(missing),
        )

    _require_anthropic_key()


def get_anthropic_client(
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: httpx.Timeout = DEFAULT_TIMEOUT,
) -> Anthropic:
    _require_anthropic_key()
    return Anthropic(timeout=timeout, max_retries=max_retries)


def get_async_anthropic_client(
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: httpx.Timeout = DEFAULT_TIMEOUT,
) -> AsyncAnthropic:
    _require_anthropic_key()
    return AsyncAnthropic(timeout=timeout, max_retries=max_retries)
