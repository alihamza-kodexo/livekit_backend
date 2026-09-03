"""Direct Slack notifications -- the Phase 6 replacement for what the FSD
originally routed through n8n (Sections 5.3.1/5.3.2). Since this build skips
n8n entirely, the worker posts to the Slack incoming-webhook URL itself once a
call has ended.

Two messages, not three, and deliberately not one per call:

  send_lead_alert            -- the call captured lead details (analysis.is_lead)
  send_transfer_failure_alert -- a transfer failed and a callback is owed

The per-call summary that used to fire on every non-test call is gone. See the
0026 migration for the reasoning; the short version is that a channel carrying
every robocall and four-second hangup is a channel nobody reads, and it cost
the alerts that mattered. Whether an agent posts at all is its own toggle
(`agents.slack_notifications_enabled`, off by default), checked by the caller
in entrypoint.py rather than in here.

NOTE: the exact message copy/format here is a reasonable best-effort, not a
transcription of FSD Section 5.3.1/5.3.2 -- I don't have that section's exact
wording. Treat these two message builders as the thing to check against the
real FSD text before launch, not as already-verified copy.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from .models import Agent, CallOutcome
from .settings import slack_settings

logger = logging.getLogger("worker.notify")


async def _post_to_slack(payload: dict) -> None:
    settings = slack_settings()
    if not settings.webhook_url:
        logger.info("SLACK_WEBHOOK_URL not set; skipping Slack notification")
        return

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(settings.webhook_url, json=payload)
            response.raise_for_status()
    except Exception:
        logger.exception("failed to post Slack notification")


async def send_lead_alert(
    *,
    agent: Agent,
    caller_number: str | None,
    outcome: CallOutcome | None,
    duration_seconds: int,
    matched_department: str | None,
    lead_name: str | None,
    lead_company: str | None,
    lead_need: str | None,
) -> None:
    """Posted when a call captured lead details -- see analysis.is_lead for what
    counts, and the 0026 migration for why this replaced the summary that used
    to fire on every single call.

    Whether it fires at all is the caller's decision to make, not this
    function's: entrypoint checks the agent's toggle and `is_lead` before
    getting here. This only formats.

    Sent as an attachment with fields rather than the plain markdown lines the
    transfer alert uses, because this one has structured content -- a name, a
    company, a need -- and fields put those in a scannable grid instead of a
    paragraph someone has to read to find the phone number.
    """

    minutes, seconds = divmod(duration_seconds, 60)

    # Slack drops a field whose value is empty, so every one is given a filler.
    # `short` pairs them two-per-row; the need runs full width because it's the
    # only free text and wrapping it into a column makes it unreadable.
    fields: list[dict[str, Any]] = [
        {"title": "Name", "value": lead_name or "not given", "short": True},
        {"title": "Company", "value": lead_company or "not given", "short": True},
        {"title": "Phone", "value": caller_number or "unknown", "short": True},
        {"title": "Duration", "value": f"{minutes}m {seconds}s", "short": True},
    ]
    if matched_department:
        fields.append({"title": "Transferred to", "value": matched_department, "short": True})
    if lead_need:
        fields.append({"title": "What they need", "value": lead_need, "short": False})

    await _post_to_slack(
        {
            "text": f":white_check_mark: New lead -- {agent.name}",
            "attachments": [
                {
                    "color": "good",
                    "fields": fields,
                    # The outcome is a footnote rather than a field: it's the
                    # model's own label, and the captured details above are the
                    # firmer fact. Worth showing, not worth leading with.
                    "footer": f"Outcome: {(outcome or 'not set').replace('_', ' ')}",
                }
            ],
        }
    )


async def send_end_call_webhook(
    *,
    agent: Agent,
    call_sid: str | None,
    room_id: str,
    caller_number: str | None,
    transcript: str,
    recording_url: str | None,
    duration_seconds: int,
    outcome: CallOutcome | None,
    matched_department: str | None,
    lead_name: str | None,
    lead_company: str | None,
    lead_need: str | None,
    qualification_answers: dict[str, Any],
    is_test: bool,
    # Who ended the call and why -- see CallState.claim_end. Sent because a CRM
    # syncing these records cares a great deal whether the caller abandoned the
    # call or the agent finished it, and `outcome` alone doesn't say: a caller
    # who hangs up mid-qualification leaves it null, exactly like a call that
    # broke. Both None on a call that ended in a way nothing claimed.
    ended_by: str | None = None,
    end_reason: str | None = None,
) -> None:
    """Posts the full call record to the agent's own configured webhook
    (Prompt & qualification tab), for whoever wants call data outside Slack/
    the dashboard -- e.g. a CRM sync in n8n. Distinct from Slack: this fires
    even for dashboard test calls, matching how a custom tool call already
    isn't sandboxed there either -- see agent-worker's tools.py.

    recording_url is a Cloudinary link when CALL_RECORDING_ENABLED is on and the
    egress and upload both worked; None otherwise, which includes every call on a
    deployment that leaves recording off. See recording.py. It's an
    `authenticated` Cloudinary asset, so a receiving workflow can't fetch it with
    the bare URL -- it needs a signed one.
    """
    if not agent.end_call_webhook_url:
        return

    payload = {
        "agent_id": agent.agent_id,
        "agent_name": agent.name,
        "call_sid": call_sid,
        "room_id": room_id,
        "caller_number": caller_number,
        "transcript": transcript,
        "recording_url": recording_url,
        "duration_seconds": duration_seconds,
        "outcome": outcome,
        "matched_department": matched_department,
        "lead_name": lead_name,
        "lead_company": lead_company,
        "lead_need": lead_need,
        "qualification_answers": qualification_answers,
        "is_test": is_test,
        "ended_by": ended_by,
        "end_reason": end_reason,
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(agent.end_call_webhook_url, json=payload)
            response.raise_for_status()
    except Exception:
        logger.exception("failed to post end-call webhook for agent %s", agent.agent_id)


async def send_transfer_failure_alert(
    *,
    agent: Agent,
    caller_number: str | None,
    department_name: str,
    callback_number: str | None,
) -> None:
    """FSD 5.3.2 equivalent: the urgent @channel alert when a transfer fails
    and the caller was told a callback is coming -- someone has to actually
    make that callback."""

    lines = [
        "@channel :rotating_light: *Transfer failed -- callback owed*",
        f"*Agent:* {agent.name}",
        f"*Caller:* {caller_number or 'unknown'}",
        f"*Wanted:* {department_name}",
        f"*Callback number given:* {callback_number or 'caller did not provide one'}",
    ]
    await _post_to_slack({"text": "\n".join(lines)})
