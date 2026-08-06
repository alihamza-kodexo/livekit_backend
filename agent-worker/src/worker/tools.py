"""end_call (the one tool every agent always has -- something has to be able
to hang up regardless of configuration) plus the generic framework that turns
a Supabase `tools` row into a callable LLM function tool. Transfer, lead
capture, and callback capture used to be unconditional builtins here too;
they're now admin-created rows like a webhook Function tool, opt-in per
agent -- see build_agent_tools' dispatch by `tool_type` below. Each still
runs fixed Python behavior (there's no webhook for these three), just wrapped
with whatever name/description the admin gave the row instead of a hardcoded
one, and only present at all if attached.
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
    return [end_call]


def build_custom_tool(tool_row: Tool) -> RawFunctionTool:
    """Turns a `tool_type: "function"` row into a live function tool that
    calls its configured webhook. This is the entire mechanism Function tools
    need -- there's no per-tool Python code, which is the point of the
    framework."""

    if not tool_row.webhook_url:
        raise ValueError(f"Function tool '{tool_row.name}' has no webhook_url configured")

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
            logger.warning("Function tool '%s' webhook call failed: %s", tool_row.name, e)
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


def build_transfer_call_tool(tool_row: Tool) -> RawFunctionTool:
    """A `tool_type: "transfer_call"` row -- SIP-transfers to the fixed
    number the admin set on this specific tool. One tool per destination
    (like Vapi's Transfer Call), each with its own admin-written description
    driving when the model reaches for it, rather than one generic tool
    matched against a routing-keywords directory."""

    if not tool_row.destination_number:
        raise ValueError(f"Transfer call tool '{tool_row.name}' has no destination_number configured")

    async def _transfer(raw_arguments: dict[str, Any], context: RunContext[CallState]) -> str:
        state = context.userdata
        job_ctx = get_job_context()
        participant = _find_sip_participant(job_ctx)
        if participant is None:
            raise ToolError("No SIP participant found in the room to transfer.")

        try:
            await job_ctx.transfer_sip_participant(participant, tool_row.destination_number)
        except Exception as e:
            logger.exception("SIP transfer to %s failed", tool_row.destination_number)
            state.outcome = "transfer_failed"
            state.matched_department = tool_row.name
            raise ToolError(
                "The transfer failed. Apologize, explain a callback is coming, and ask for the "
                "best number to reach them at -- then record it if you have a way to."
            ) from e

        state.outcome = "department_transfer"
        state.matched_department = tool_row.name
        return "Transferred."

    _transfer.__name__ = tool_row.name

    return function_tool(
        _transfer,
        raw_schema={
            "name": tool_row.name,
            "description": (
                f"{tool_row.description} The transfer itself happens silently -- tell the caller "
                "you're transferring them before calling this, not after."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    )


def build_record_lead_info_tool(tool_row: Tool) -> RawFunctionTool:
    """A `tool_type: "record_lead_info"` row -- same fixed behavior (writes
    onto CallState, ends up in call_logs) as the old unconditional builtin,
    just opt-in per agent with an admin-written description now instead of a
    fixed one, so an agent that isn't doing lead qualification doesn't have
    this nudging it to ask for a name/company/need it has no use for."""

    async def _record(raw_arguments: dict[str, Any], context: RunContext[CallState]) -> str:
        state = context.userdata
        lead_name = raw_arguments.get("lead_name")
        lead_company = raw_arguments.get("lead_company")
        lead_need = raw_arguments.get("lead_need")
        qualification_answers = raw_arguments.get("qualification_answers")
        if lead_name:
            state.lead_name = lead_name
        if lead_company:
            state.lead_company = lead_company
        if lead_need:
            state.lead_need = lead_need
        if qualification_answers:
            state.qualification_answers.update(qualification_answers)
        return "Recorded."

    _record.__name__ = tool_row.name

    return function_tool(
        _record,
        raw_schema={
            "name": tool_row.name,
            "description": tool_row.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "lead_name": {"type": "string"},
                    "lead_company": {"type": "string"},
                    "lead_need": {"type": "string"},
                    "qualification_answers": {"type": "object"},
                },
            },
        },
    )


def build_record_callback_number_tool(tool_row: Tool) -> RawFunctionTool:
    """A `tool_type: "record_callback_number"` row -- same fixed behavior
    (feeds the transfer-failed Slack alert) as the old unconditional builtin,
    opt-in per agent now."""

    async def _record(raw_arguments: dict[str, Any], context: RunContext[CallState]) -> str:
        context.userdata.transfer_failed_callback_number = raw_arguments.get("callback_number")
        return "Recorded."

    _record.__name__ = tool_row.name

    return function_tool(
        _record,
        raw_schema={
            "name": tool_row.name,
            "description": tool_row.description,
            "parameters": {
                "type": "object",
                "required": ["callback_number"],
                "properties": {"callback_number": {"type": "string"}},
            },
        },
    )


_TOOL_BUILDERS = {
    "function": build_custom_tool,
    "transfer_call": build_transfer_call_tool,
    "record_lead_info": build_record_lead_info_tool,
    "record_callback_number": build_record_callback_number_tool,
}


def build_agent_tools(tool_rows: list[Tool]) -> list:
    """Every attached tool, regardless of type -- dispatches on `tool_type`
    to the matching builder above."""

    tools = []
    for row in tool_rows:
        builder = _TOOL_BUILDERS.get(row.tool_type)
        if builder is None:
            logger.error("unknown tool_type %r for tool row %s", row.tool_type, row.tool_id)
            continue
        try:
            tools.append(builder(row))
        except ValueError:
            logger.exception("skipping malformed tool row %s", row.tool_id)
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
