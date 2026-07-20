"""
Provider configuration for the LLM client.

Groq, Ollama, and OpenAI all speak the same OpenAI-compatible chat
completions wire protocol, so switching providers is just a base_url +
api_key swap -- no new SDK, no branching call logic. Set LLM_PROVIDER in
your environment to choose; defaults to "groq".

Why Groq is the default: it's free with no credit card, and unlike a local
Ollama model it works unmodified inside GitHub Actions in Phase 5 (CI
runners can't reasonably pull multi-GB model weights on every run).
Ollama is left in as an option for fully offline local development.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Loads .env into the environment on import, so GROQ_API_KEY / OPENAI_API_KEY
# work the same way whether you're on bash, zsh, or PowerShell -- no manual
# `export`/`$env:` needed as long as a .env file exists next to this project.
load_dotenv()


@dataclass(frozen=True)
class ProviderConfig:
    base_url: str | None  # None = OpenAI's default endpoint
    api_key_env: str | None  # env var to read the key from; None = no key needed
    dummy_key: str | None = None  # used when the provider ignores the key (e.g. Ollama)


_PROVIDERS: dict[str, ProviderConfig] = {
    "groq": ProviderConfig(
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
    ),
    "openai": ProviderConfig(
        base_url=None,
        api_key_env="OPENAI_API_KEY",
    ),
    "ollama": ProviderConfig(
        base_url="http://localhost:11434/v1",
        api_key_env=None,
        dummy_key="ollama",  # Ollama's OpenAI-compatible endpoint ignores the key but the SDK requires one
    ),
}


def get_provider_config() -> ProviderConfig:
    name = os.environ.get("LLM_PROVIDER", "groq").lower()
    if name not in _PROVIDERS:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{name}'. Supported: {sorted(_PROVIDERS)}"
        )
    return _PROVIDERS[name]


def get_api_key(config: ProviderConfig) -> str:
    if config.api_key_env is None:
        return config.dummy_key or "not-needed"
    key = os.environ.get(config.api_key_env)
    if not key:
        raise RuntimeError(
            f"LLM_PROVIDER='{os.environ.get('LLM_PROVIDER', 'groq')}' requires "
            f"the {config.api_key_env} environment variable to be set."
        )
    return key


def get_extra_body(model: str) -> dict:
    """
    Provider/model-specific extra API fields, passed through the OpenAI SDK's
    extra_body kwarg (rejected as unknown fields by strict APIs otherwise).

    Groq's gpt-oss models are reasoning models -- they spend hidden tokens
    "thinking" before answering. For a simple 4-way classification task we
    don't need deep reasoning, so capping it to "low" leaves more of
    max_tokens for the actual JSON output and responds faster.
    """
    if model.startswith("openai/gpt-oss"):
        return {"reasoning_effort": "low"}
    return {}
