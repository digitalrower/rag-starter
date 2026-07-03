import httpx
from anthropic import Anthropic, AsyncAnthropic

DEFAULT_TIMEOUT = httpx.Timeout(timeout=30.0, connect=5.0)
DEFAULT_MAX_RETRIES = 4


def get_anthropic_client(
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: httpx.Timeout = DEFAULT_TIMEOUT,
) -> Anthropic:
    return Anthropic(timeout=timeout, max_retries=max_retries)


def get_async_anthropic_client(
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: httpx.Timeout = DEFAULT_TIMEOUT,
) -> AsyncAnthropic:
    return AsyncAnthropic(timeout=timeout, max_retries=max_retries)
