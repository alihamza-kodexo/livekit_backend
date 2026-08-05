"""Environment access, mirroring the pattern in dashboard/lib/env.ts.

Every secret is read lazily through one of these functions rather than at
import time, so a worker without, say, a Slack webhook configured still boots
and takes calls -- it just skips the notification instead of crashing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv(".env.local")
load_dotenv(".env")


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable {name}. "
            f"Copy agent-worker/.env.example to agent-worker/.env.local and fill it in."
        )
    return value


@dataclass(frozen=True)
class SupabaseSettings:
    url: str
    service_role_key: str


@dataclass(frozen=True)
class LiveKitSettings:
    url: str
    api_key: str
    api_secret: str
    agent_name: str


@dataclass(frozen=True)
class ProviderSettings:
    """Deepgram covers both ends of the audio pipeline (STT + Aura TTS) --
    see VOICE_STACK_DECISION.md for why that beat keeping Cartesia.

    Groq, DeepSeek, and Gemini Live are all lazy-optional -- an agent only
    needs the key for whichever llm_provider it's actually set to, checked at
    the moment a call needs it (see entrypoint.py), not at worker startup. A
    worker that never runs a Groq-configured agent can leave GROQ_API_KEY
    unset entirely, same as the other two.
    """

    deepgram_api_key: str
    groq_api_key: str | None
    groq_model: str
    deepseek_api_key: str | None
    deepseek_model: str
    deepseek_base_url: str
    # Only needed for agents with llm_provider='gemini_live' -- same
    # lazy-optional pattern as DeepSeek above.
    gemini_api_key: str | None
    gemini_model: str


@dataclass(frozen=True)
class SlackSettings:
    """Direct Slack incoming-webhook notifications -- there is no n8n hop in
    this build, per the project decision to skip n8n entirely. The worker
    posts straight to Slack and straight to Supabase after each call."""

    webhook_url: str | None


@lru_cache
def supabase_settings() -> SupabaseSettings:
    return SupabaseSettings(
        url=_required("SUPABASE_URL"),
        service_role_key=_required("SUPABASE_SERVICE_ROLE_KEY"),
    )


@lru_cache
def livekit_settings() -> LiveKitSettings:
    return LiveKitSettings(
        url=_required("LIVEKIT_URL"),
        api_key=_required("LIVEKIT_API_KEY"),
        api_secret=_required("LIVEKIT_API_SECRET"),
        agent_name=os.environ.get("LIVEKIT_AGENT_NAME", "kodexo-inbound-agent"),
    )


@lru_cache
def provider_settings() -> ProviderSettings:
    return ProviderSettings(
        deepgram_api_key=_required("DEEPGRAM_API_KEY"),
        groq_api_key=os.environ.get("GROQ_API_KEY") or None,
        groq_model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
        deepseek_api_key=os.environ.get("DEEPSEEK_API_KEY") or None,
        deepseek_model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        deepseek_base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        gemini_api_key=os.environ.get("GEMINI_API_KEY") or None,
        gemini_model=os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-live-preview"),
    )


@lru_cache
def slack_settings() -> SlackSettings:
    return SlackSettings(webhook_url=os.environ.get("SLACK_WEBHOOK_URL") or None)
