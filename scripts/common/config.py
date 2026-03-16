"""Environment configuration and client factories for all providers.

Environment variables are resolved with fallback chains:
  BASE_URL  -> <PROVIDER>_BASE_URL  -> provider default
  API_KEY   -> <PROVIDER>_API_KEY   (required)
  MODEL     -> <PROVIDER>_MODEL     -> "gpt-4o-mini"
  STREAM    (shared, default "true")
"""

import os

from dotenv import load_dotenv

load_dotenv()


def _env(*names: str, default: str | None = None) -> str:
    """Return the first set env var from *names*, or *default*.

    Raises OSError if nothing is found and no default is provided.
    """
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    if default is not None:
        return default
    raise OSError(
        f"Required environment variable not set (looked for: {', '.join(names)})"
    )


def _get_provider_config(
    provider_prefix: str,
    default_base_url: str,
    default_model: str = "gpt-4o-mini",
) -> dict:
    """Build config for a provider with generic -> provider-specific fallback."""
    return {
        "base_url": _env(
            "BASE_URL", f"{provider_prefix}_BASE_URL", default=default_base_url
        ),
        "api_key": _env("API_KEY", f"{provider_prefix}_API_KEY"),
        "model": _env("MODEL", f"{provider_prefix}_MODEL", default=default_model),
        "stream": _env("STREAM", default="true").lower() == "true",
    }


def create_openai_client():
    """Create an OpenAI client."""
    from openai import OpenAI

    cfg = _get_provider_config("OPENAI", "https://api.openai.com/v1")
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"]), cfg


def create_anthropic_client():
    """Create an Anthropic client."""
    import anthropic

    cfg = _get_provider_config("ANTHROPIC", "https://api.anthropic.com")
    return anthropic.Anthropic(api_key=cfg["api_key"], base_url=cfg["base_url"]), cfg


def create_google_client():
    """Create a Google GenAI client."""
    from google import genai
    from google.genai.types import HttpOptions

    cfg = _get_provider_config("GOOGLE", "https://generativelanguage.googleapis.com")
    client = genai.Client(
        api_key=cfg["api_key"],
        http_options=HttpOptions(base_url=cfg["base_url"]),
    )
    return client, cfg


def create_openai_responses_client():
    """Create an OpenAI client for the Responses API."""
    from openai import OpenAI

    cfg = _get_provider_config(
        "OPENAI_RESPONSES",
        _env("OPENAI_BASE_URL", default="https://api.openai.com/v1"),
        _env("OPENAI_MODEL", default="gpt-4o-mini"),
    )
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"]), cfg
