"""Per-call state shared across every Agent phase and tool via AgentSession.userdata.

One instance is created per LiveKit job (per call) and threaded through the
whole session -- this is what lets the qualification data gathered in one
Agent phase still be there when a later phase (or the post-call logger) needs it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import AgentConfig, CallOutcome


@dataclass
class CallState:
    config: AgentConfig
    room_name: str
    call_sid: str | None = None
    caller_number: str | None = None
    # The number the caller dialled (sip.trunkPhoneNumber). Read at answer time
    # to resolve which agent owns the call, and kept because it's also the only
    # record of *which* of an agent's numbers was rung -- worth having once an
    # agent fronts more than one. None for a browser test, which dials nothing.
    called_number: str | None = None
    # True for a dashboard "test this agent" session (see entrypoint.py) --
    # skips call_logs/Slack so test runs don't pollute real call history.
    is_test: bool = False

    # Filled in by the record_lead_info built-in tool as the qualification
    # conversation progresses (see tools.py). Keys match qualification_criteria
    # keys, plus the fixed lead_name/lead_company/lead_need fields.
    qualification_answers: dict[str, Any] = field(default_factory=dict)
    lead_name: str | None = None
    lead_company: str | None = None
    lead_need: str | None = None

    # Set once the call flow reaches a terminal state, so the shutdown
    # callback in entrypoint.py knows what to write to call_logs.
    outcome: CallOutcome | None = None
    matched_department: str | None = None
    transfer_failed_callback_number: str | None = None

    # Whether the session broke mid-call, and the first reason it gave. Set by
    # entrypoint's ErrorEvent handler, which until now only reported the failure
    # to the caller and the log -- so "did that call actually work" was a
    # question the call record couldn't answer. The *first* error is kept rather
    # than the last: a failing session tends to emit a cascade, and the one that
    # started it is the useful one.
    has_error: bool = False
    error_message: str | None = None

    # Which spam detector fired and why (see spam.py). Set alongside a
    # spam_bot/spam_sales outcome, and worth more than the outcome alone: those
    # calls are hung up on with no explanation to the caller, so this is the only
    # evidence available when deciding whether a statement list is too broad.
    spam_detection: str | None = None

    # Rotates through the "are you AI?" deflection lines (FSD Section 4.5-ish
    # human-likeness requirement) so no line repeats within a single call.
    ai_deflection_index: int = 0
