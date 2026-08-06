"""Built-in tools (transfer, lead/call tracking) plus the generic framework
that turns a Supabase `tools` row into a callable LLM function tool at
webhook_url. Matches the Project Plan v2 scope addition: custom tools are a
dashboard entry, not new worker code, because they all funnel through
`build_custom_tool` below.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from livekit import rtc
from livekit.agents import RunContext, get_job_context
from livekit.agents.llm import RawFunctionTool, ToolError, function_tool

from .models import Agent, Tool
from .state import CallState

logger = logging.getLogger("worker.tools")


@function_tool(
    name="transfer_to_department",
    description=(
        "Silently transfer the caller to a department by name. Only call this after you have "
        "already spoken the handoff line out loud to the caller -- the transfer itself is silent, "
        "but the caller must never be moved without being told first."
    ),
)
async def transfer_to_department(
    context: RunContext[CallState], department_name: str
) -> str:
    state = context.userdata
    department = next(
        (
            d
            for d in state.config.departments
            if d.department_name.lower() == department_name.lower()
        ),
        None,
    )
    if department is None:
        available = ", ".join(d.department_name for d in state.config.departments) or "none configured"
        raise ToolError(
            f"No department named '{department_name}'. Available departments: {available}."
        )

    job_ctx = get_job_context()
    participant = _find_sip_participant(job_ctx)
    if participant is None:
        raise ToolError("No SIP participant found in the room to transfer.")

    try:
        await job_ctx.transfer_sip_participant(participant, department.transfer_number)
    except Exception as e:
        logger.exception("SIP transfer to %s failed", department.transfer_number)
        state.outcome = "transfer_failed"
        state.matched_department = department.department_name
        raise ToolError(
            "The transfer failed. Apologize, explain a callback is coming, and ask for the best "
            "number to reach them at -- then call record_lead_info with that number."
        ) from e

    state.outcome = "department_transfer"
    state.matched_department = department.department_name
    return f"Transferred to {department.department_name}."


@function_tool(
    name="record_lead_info",
    description=(
        "Record what you've learned about the caller so far -- their name, company, what they "
        "need, and answers to qualification questions. Call this as you learn each piece of "
        "information during the conversation, not just at the end."
    ),
)
async def record_lead_info(
    context: RunContext[CallState],
    lead_name: str | None = None,
    lead_company: str | None = None,
    lead_need: str | None = None,
    qualification_answers: dict[str, str] | None = None,
) -> str:
    state = context.userdata
    if lead_name:
        state.lead_name = lead_name
    if lead_company:
        state.lead_company = lead_company
    if lead_need:
        state.lead_need = lead_need
    if qualification_answers:
        state.qualification_answers.update(qualification_answers)
    return "Recorded."


@function_tool(
    name="record_callback_number",
    description=(
        "Record a callback number after a failed transfer or any time the caller needs to be "
        "called back. Use this instead of record_lead_info for a callback number specifically."
    ),
)
async def record_callback_number(context: RunContext[CallState], callback_number: str) -> str:
    context.userdata.transfer_failed_callback_number = callback_number
    return "Recorded."


@function_tool(
    name="end_call",
    description=(
        "End the call. Only call this AFTER you have already spoken your closing line out loud "
        "-- this waits for that line to finish playing, then hangs up. Use outcome='not_qualified' "
        "for a normal close, 'dropped' for a scam/fake-verification call you're ending immediately, "
        "or leave outcome unset if it was already set by a transfer."
    ),
)
async def end_call(
    context: RunContext[CallState], outcome: str | None = None
) -> str:
    state = context.userdata
    if outcome and state.outcome is None:
        state.outcome = outcome  # type: ignore[assignment]

    # Let the closing line finish playing before the room disconnects under it.
    await context.speech_handle.wait_for_playout()

    job_ctx = get_job_context()
    job_ctx.delete_room()
    return "Call ending."


def _find_sip_participant(job_ctx: Any):
    for participant in job_ctx.room.remote_participants.values():
        if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
            return participant
    return None


def builtin_tools() -> list:
    return [transfer_to_department, record_lead_info, record_callback_number, end_call]


def build_custom_tool(tool_row: Tool) -> RawFunctionTool:
    """Turns one non-builtin `tools` row into a live function tool that calls
    its configured webhook. This is the entire mechanism custom tools need --
    there's no per-tool Python code, which is the point of the framework."""

    if not tool_row.webhook_url:
        raise ValueError(f"Custom tool '{tool_row.name}' has no webhook_url configured")

    async def _call_webhook(raw_arguments: dict[str, Any], context: RunContext[CallState]) -> str:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(tool_row.webhook_url, json=raw_arguments)
                response.raise_for_status()
                # Tools are free-form webhooks; the LLM can work with either
                # a JSON body or plain text back, so don't force a shape here.
                try:
                    return str(response.json())
                except ValueError:
                    return response.text
        except httpx.HTTPError as e:
            logger.warning("custom tool '%s' webhook call failed: %s", tool_row.name, e)
            raise ToolError(
                f"The {tool_row.name} tool is temporarily unavailable. "
                "Apologize and continue without it, or offer a callback."
            ) from e

    _call_webhook.__name__ = tool_row.name

    return function_tool(
        _call_webhook,
        raw_schema={
            "name": tool_row.name,
            "description": tool_row.description,
            "parameters": tool_row.parameter_schema or {"type": "object", "properties": {}},
        },
    )


def build_agent_tools(tool_rows: list[Tool]) -> list:
    """Custom tools only -- built-ins are added separately per Agent phase
    since not every phase should be able to call every tool (e.g. only the
    qualification phase should transfer)."""

    tools = []
    for row in tool_rows:
        if row.is_builtin:
            continue
        try:
            tools.append(build_custom_tool(row))
        except ValueError:
            logger.exception("skipping malformed custom tool row %s", row.tool_id)
    return tools


def build_knowledge_tool(agent: Agent) -> list[RawFunctionTool]:
    """A single on-demand tool wrapping the agent's whole knowledge base --
    there's one knowledge base per agent, not several documents to pick
    between, so one tool is enough. Its content is only sent to the model
    (and only costs tokens) on the calls where it's actually invoked, instead
    of being concatenated into every turn's prompt regardless of whether it's
    ever needed. Returns an empty list if no content is configured, so an
    agent with nothing to look up doesn't get an empty/useless tool."""

    if not agent.knowledge_base_content.strip():
        return []

    async def _search_knowledge_base(
        raw_arguments: dict[str, Any], context: RunContext[CallState]
    ) -> str:
        return agent.knowledge_base_content

    return [
        function_tool(
            _search_knowledge_base,
            raw_schema={
                "name": "search_knowledge_base",
                "description": agent.knowledge_base_description
                or "Reference material for off-topic caller questions.",
                "parameters": {"type": "object", "properties": {}},
            },
        )
    ]
