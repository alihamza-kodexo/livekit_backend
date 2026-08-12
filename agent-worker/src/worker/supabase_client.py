"""Async Supabase access for the worker: resolve an agent by dialed number,
load its full config, log the finished call, and record run-time errors.

Mirrors dashboard/lib/queries.ts, but read-heavy and scoped to what a single
call needs rather than what an admin screen needs.
"""

from __future__ import annotations

import asyncio
import logging
import weakref
from typing import Any

from supabase import AsyncClient, acreate_client

from .models import (
    Agent,
    AgentConfig,
    CallOutcome,
    Tool,
)
from .settings import supabase_settings

logger = logging.getLogger("worker.supabase")

# Keyed by event loop, not a single module-level instance.
#
# The client owns an httpx connection pool, and httpx's HTTP/2 machinery binds
# asyncio primitives (anyio Events, futures) to whichever loop first used the
# connection. A worker process is reused across jobs and every job runs on its
# own event loop, so a process-wide singleton hands job N+1 a pool bound to job
# N's dead loop. The result is a bare
#
#     RuntimeError: <asyncio.locks.Event ...> is bound to a different event loop
#
# raised from the very first Supabase call in the job -- loading the agent
# config -- so the agent never joins and the caller sits on "Connecting" with no
# other symptom. It presents as random flakiness because it only bites when a
# job lands on an already-used process.
#
# Weak keys so an entry disappears with its loop instead of accumulating one
# dead client per job for the life of the process.
_clients: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, AsyncClient] = (
    weakref.WeakKeyDictionary()
)


async def get_client() -> AsyncClient:
    loop = asyncio.get_running_loop()
    client = _clients.get(loop)
    if client is None:
        settings = supabase_settings()
        client = await acreate_client(settings.url, settings.service_role_key)
        _clients[loop] = client
    return client


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

    # Tools are a global library now (see 0014_global_tools.sql) -- this joins
    # through agent_tools to whichever ones this agent has selected. Every
    # tool_type (including transfer_call, which replaced the old departments
    # directory -- see 0016_tool_types_and_transfer_call.sql) comes through
    # here identically.
    agent_tools_result = (
        await client.table("agent_tools")
        .select("tools(*)")
        .eq("agent_id", agent.agent_id)
        .execute()
    )

    return AgentConfig(
        agent=agent,
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
    # None whenever recording is off, or on for a call whose egress or upload
    # didn't work out -- see recording.py. The column has always been nullable
    # and every row written before recording existed has it NULL.
    recording_url: str | None,
    duration_seconds: int | None,
    outcome: CallOutcome | None,
    matched_department: str | None,
    # Which spam detector ended the call and why -- see spam.py. Null on every
    # call that wasn't dropped as spam. Paired with a spam_bot/spam_sales
    # outcome, and the only trace of a hangup the caller was given no
    # explanation for, so it's what a false positive is found from.
    spam_detection: str | None = None,
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
                "recording_url": recording_url,
                "duration_seconds": duration_seconds,
                "outcome": outcome,
                "matched_department": matched_department,
                "spam_detection": spam_detection,
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
