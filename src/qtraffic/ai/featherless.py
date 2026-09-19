"""The ONLY module that talks to Featherless (an OpenAI-compatible API).

Nothing else in the project imports the SDK or knows the endpoint. The client turns every failure into a
``FeatherlessError`` with a coarse ``kind`` so the explanation layer can fall back to deterministic text.
It never logs or returns the API key: every message that leaves this module is redacted.

The ``openai`` package is imported lazily. If it is not installed the explanation layer simply reports
``package_missing`` and uses the deterministic fallback; the rest of the project is unaffected.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, Sequence

from .config import FeatherlessConfig

# failure kinds the explanation layer reports
KINDS = ("missing_api_key", "missing_model", "invalid_base_url", "package_missing", "authentication", "rate_limit",
         "timeout", "connection", "server_error", "malformed_response", "validation_failed", "request_failed")
Message = dict  # {"role": "system" | "user", "content": str}


class FeatherlessError(Exception):
    """A request could not produce usable text. ``kind`` is one of ``KINDS``; ``detail`` is redacted."""

    def __init__(self, kind: str, detail: str = ""):
        super().__init__(f"{kind}: {detail}" if detail else kind)
        self.kind = kind
        self.detail = detail


class ChatClient(Protocol):
    """What the explanation layer needs. Tests provide fakes; production uses ``OpenAICompatibleClient``."""

    def complete(self, messages: Sequence[Message], max_tokens: int, timeout_seconds: float) -> str: ...


def classify_exception(exc: BaseException) -> str:
    """Map an SDK / network exception to a failure kind, without importing the SDK's exception classes."""
    names = {c.__name__ for c in type(exc).__mro__}
    status = getattr(exc, "status_code", None)
    if isinstance(exc, TimeoutError) or "APITimeoutError" in names or "Timeout" in names or "ReadTimeout" in names:
        return "timeout"
    if status in (401, 403) or names & {"AuthenticationError", "PermissionDeniedError"}:
        return "authentication"
    if status == 429 or "RateLimitError" in names:
        return "rate_limit"
    if isinstance(status, int) and status >= 500 or "InternalServerError" in names:
        return "server_error"
    if names & {"APIConnectionError", "ConnectError", "ConnectionError"} or isinstance(exc, ConnectionError):
        return "connection"
    return "request_failed"


def _extract_text(response: Any) -> str:
    """Pull the assistant text out of a chat-completion response, or raise ``malformed_response``."""
    try:
        text = response.choices[0].message.content
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise FeatherlessError("malformed_response", "response has no choices[0].message.content") from exc
    if not isinstance(text, str):
        raise FeatherlessError("malformed_response", "message content is not text")
    return text


def _default_sdk_factory(**kwargs: Any) -> Any:
    try:
        from openai import OpenAI  # lazy: the project runs without this package
    except ImportError as exc:
        raise FeatherlessError("package_missing", "the 'openai' package is not installed") from exc
    return OpenAI(**kwargs)


class OpenAICompatibleClient:
    """Chat-completions client for an OpenAI-compatible endpoint (Featherless). One request per call, no retries."""

    def __init__(self, config: FeatherlessConfig, sdk_factory: Callable[..., Any] | None = None):
        reason = config.unavailable_reason()
        if reason is not None:
            raise FeatherlessError(reason)
        self._config = config
        self._factory = sdk_factory or _default_sdk_factory
        self._sdk: Any = None

    def _client(self, timeout_seconds: float) -> Any:
        if self._sdk is None:
            self._sdk = self._factory(api_key=self._config.api_key, base_url=self._config.base_url,
                                      timeout=timeout_seconds, max_retries=0)
        return self._sdk

    def complete(self, messages: Sequence[Message], max_tokens: int, timeout_seconds: float) -> str:
        try:
            sdk = self._client(timeout_seconds)
            response = sdk.chat.completions.create(
                model=self._config.model, messages=list(messages), max_tokens=max_tokens, temperature=0.2)
        except FeatherlessError:
            raise
        except Exception as exc:  # noqa: BLE001 - every SDK/network failure becomes a FeatherlessError
            raise FeatherlessError(classify_exception(exc), self._config.redact(str(exc))[:200]) from None
        return _extract_text(response)
