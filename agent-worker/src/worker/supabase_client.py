"""Async Supabase access for the worker: resolve an agent by dialed number,
load its full config, log the finished call, and record run-time errors.

Mirrors dashboard/lib/queries.ts, but read-heavy and scoped to what a single
call needs rather than what an admin screen needs.
"""

from __future__ import annotations

import logging
from typing import Any

from supabase import AsyncClient, acreate_client

from .models import (
    Agent,
    AgentConfig,
    CallOutcome,
    Department,
    Tool,
)
from .settings import supabase_settings

logger = logging.getLogger("worker.supabase")

_client: AsyncClient | None = None


async def get_client() -> AsyncClient:
    global _client
    if _client is None:
        settings = supabase_settings()
        _client = await acreate_client(settings.url, settings.service_role_key)
    return _client


async def load_agent_config_by_number(dialed_number: str) -> AgentConfig | None:
    """The routing lookup Project Plan v2 describes: which agent owns a call
    is decided here, by the number Twilio/LiveKit says was dialed -- not by
    any Twilio-side per-number configuration."""

    client = await get_client()
    result = (
        await client.table("agents")
        .select("*")
        .eq("twilio_number", dialed_number)
        .maybe_single()
        .execute()
    )
    if result is None or result.data is None:
        return None

    return await _load_config_for_agent_row(client, result.data)


async def load_agent_config_by_id(agent_id: str) -> AgentConfig | None:
    """Used for local/console testing, where there's no real SIP call to read
    a dialed number from."""

    client = await get_client()
    result = (
        await client.table("agents").select("*").eq("agent_id", agent_id).maybe_single().execute()
    )
    if result is None or result.data is None:
        return None

    return await _load_config_for_agent_row(client, result.data)


async def _load_config_for_agent_row(client: AsyncClient, agent_row: dict[str, Any]) -> AgentConfig:
    agent = Agent.from_row(agent_row)

    departments_result = (
        await client.table("departments").select("*").eq("agent_id", agent.agent_id).execute()
    )
    # Tools are a global library now (see 0014_global_tools.sql) -- this joins
    # through agent_tools to whichever ones this agent has selected.
    agent_tools_result = (
        await client.table("agent_tools")
        .select("tools(*)")
        .eq("agent_id", agent.agent_id)
        .execute()
    )

    return AgentConfig(
        agent=agent,
        departments=[Department.from_row(r) for r in departments_result.data],
        tools=[
            Tool.from_row(row["tools"]) for row in agent_tools_result.data if row.get("tools")
        ],
    )


async def insert_call_log(
    *,
    call_sid: str | None,
    room_id: str | None,
    agent_id: str | None,
    caller_number: str | None,
    transcript: str | None,
    duration_seconds: int | None,
    outcome: CallOutcome | None,
    matched_department: str | None,
    lead_name: str | None,
    lead_company: str | None,
    lead_need: str | None,
    is_test: bool = False,
) -> None:
    """Writes the post-call record directly to Supabase. There is no n8n hop
    in this build -- see settings.SlackSettings -- so this and
    notify.py::send_call_summary are the entire Phase 6 replacement."""

    client = await get_client()
    try:
        await client.table("call_logs").insert(
            {
                "call_sid": call_sid,
                "room_id": room_id,
                "agent_id": agent_id,
                "caller_number": caller_number,
                "transcript": transcript,
                "duration_seconds": duration_seconds,
                "outcome": outcome,
                "matched_department": matched_department,
                "lead_name": lead_name,
                "lead_company": lead_company,
                "lead_need": lead_need,
                "is_test": is_test,
            }
        ).execute()
    except Exception:
        # A failed write shouldn't crash call teardown -- the call already
        # happened. Logged loudly so it's caught in worker monitoring instead.
        logger.exception("failed to write call_logs row for call_sid=%s", call_sid)
