"""Featherless configuration, read from environment variables only.

    FEATHERLESS_API_KEY    required for any API call; never hard-coded, never logged, never in a repr
    FEATHERLESS_MODEL      the model to use; NO default is assumed (set the one supplied for the project)
    FEATHERLESS_BASE_URL   optional; default https://api.featherless.ai/v1

With no key (or no model) the explanation layer does not call anything and uses the deterministic
fallback. Nothing in the simulator, optimizer or emergency controller reads this configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping
from urllib.parse import urlparse

ENV_API_KEY = "FEATHERLESS_API_KEY"
ENV_MODEL = "FEATHERLESS_MODEL"
ENV_BASE_URL = "FEATHERLESS_BASE_URL"
DEFAULT_BASE_URL = "https://api.featherless.ai/v1"
DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_MAX_TOKENS = 400  # explanations are short (about 100-250 words)


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


@dataclass(frozen=True)
class FeatherlessConfig:
    api_key: str | None = field(default=None, repr=False)  # excluded from repr on purpose
    model: str | None = None
    base_url: str = DEFAULT_BASE_URL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_tokens: int = DEFAULT_MAX_TOKENS

    def __post_init__(self) -> None:
        if not self.timeout_seconds > 0 or self.max_tokens < 1:
            raise ValueError("timeout_seconds and max_tokens must be positive")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> FeatherlessConfig:
        """Read the three variables. Blank values count as unset; nothing is validated by calling out."""
        env = os.environ if environ is None else environ
        return cls(api_key=_clean(env.get(ENV_API_KEY)), model=_clean(env.get(ENV_MODEL)),
                   base_url=_clean(env.get(ENV_BASE_URL)) or DEFAULT_BASE_URL)

    @property
    def has_api_key(self) -> bool:
        return self.api_key is not None

    @property
    def has_model(self) -> bool:
        return self.model is not None

    @property
    def base_url_is_valid(self) -> bool:
        parsed = urlparse(self.base_url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)

    def unavailable_reason(self) -> str | None:
        """Why no request may be made, or None if the configuration is complete."""
        if not self.has_api_key:
            return "missing_api_key"
        if not self.has_model:
            return "missing_model"
        if not self.base_url_is_valid:
            return "invalid_base_url"
        return None

    def public_summary(self) -> dict:
        """Safe to print or log: never contains the key."""
        return {"api_key_set": self.has_api_key, "model": self.model, "base_url": self.base_url,
                "timeout_seconds": self.timeout_seconds, "max_tokens": self.max_tokens}

    def redact(self, text: str) -> str:
        """Remove the key from any text before it is stored, printed or returned."""
        return text.replace(self.api_key, "***") if self.api_key else text
