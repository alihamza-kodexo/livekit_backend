"""Post-call analysis: the three things about a call only a model can tell us.

This replaces an n8n flow that ran an OpenAI prompt over a Vapi webhook payload
and asked it to extract fourteen fields. That design was forced by where n8n sat
-- outside the call, holding nothing but an opaque payload, so even the caller's
phone number had to be mined out of it by a language model.

This worker *is* the call, so most of those fields are facts rather than
inferences and are read directly:

    conversation_id     -> call_logs.room_id / call_sid
    caller_phone        -> sip.phoneNumber, captured at answer time
    called_number       -> sip.trunkPhoneNumber, likewise
    caller_name         -> lead_name, from the record_lead_info tool
    duration_seconds    -> measured
    transcript          -> the session's own history
    transfer_attempted  -> whether the transfer tool ran
    callback_needed     -> whether a callback number was recorded
    has_error           -> whether the session emitted an ErrorEvent
    call_status         -> derived from the above, in `call_status()`

Asking a model to re-derive any of those would be slower, cost money, and be
*less* accurate -- a SIP attribute is exact where a model reading a transcript
can drop a digit from a phone number. Never infer what you can observe.

That leaves three genuine judgements, which is all this module asks for:
`user_queries`, `call_summary` and `priority`.

Runs in the shutdown callback, after the caller has hung up. Nothing is waiting
on it, which is what makes DeepSeek the right model here: it was rejected as the
conversation engine because ~1.6s per reply is unusable mid-call, but off the
call path that cost is irrelevant and it is roughly a tenth the price of the
alternatives -- about $0.0007 for a three-minute call.

Fails open, on the same principle as spam.py: a timeout, a missing key or a
provider outage leaves the analysis fields NULL and the call_logs row is still
written. Losing a summary is a nuisance; losing the record of a call is not.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from livekit.agents.llm import ChatContext

from .llm_clients import build_utility_llm
from .settings import ProviderSettings
from .state import CallState

logger = logging.getLogger("worker.analysis")

Priority = Literal["High", "Medium", "Low"]
CallStatus = Literal["success", "failed", "incomplete"]

# Generous compared with the spam classifier's 2.5s, because the trade is
# different: that one gates a live call, this one only delays a shutdown
# callback that the caller has already walked away from. Still bounded -- the
# call_logs row and the Slack alert queue behind it.
_TIMEOUT = 20.0

# Below this there is nothing to summarise. A caller who hung up during the
# greeting produces a transcript of one line, and asking a model to find the
# "priority" of that invites it to invent one.
_MIN_TRANSCRIPT_CHARS = 80

_PRIORITIES: set[str] = {"High", "Medium", "Low"}

_PROMPT = """\
You are reviewing a finished phone call to a company's AI receptionist.

Return ONLY a JSON object, with no markdown fence and no commentary:

{
  "user_queries": ["..."],
  "call_summary": "...",
  "priority": "High" | "Medium" | "Low"
}

user_queries: the caller's substantive statements and requests, cleaned up and
normalised. Exclude bare confirmations ("yes", "okay", "hello") unless one is
the actual answer to a question. Use an automated call's own description if
there is no human -- e.g. ["Automated verification message"] or
["Spam/marketing call"]. Empty array if the caller said nothing of substance.

call_summary: under 150 words, covering what the caller wanted and what
resolution they got. Mention a transfer or an arranged callback if either
happened. Write plainly; no preamble.

priority:
  High   - an existing customer with an issue, a development or project
           enquiry, anything affecting the caller's operations, business-critical.
  Medium - general business enquiries, new service questions, follow-ups,
           account questions.
  Low    - general information requests, spam and marketing calls, automated
           verification codes, anything not urgent.

Judge only from what was said. Do not guess at names, numbers or companies --
those are recorded separately and your guess would overwrite a known value."""


@dataclass
class CallAnalysis:
    """The model's three judgements. Every field has a safe default, so a
    partial or malformed answer degrades field by field rather than discarding
    the lot."""

    user_queries: list[str] = field(default_factory=list)
    call_summary: str | None = None
    priority: Priority | None = None


def call_status(state: CallState, duration_seconds: int | None) -> CallStatus:
    """Derived, not asked of a model -- every input here is already known.

    - "failed": the session itself broke (see entrypoint's ErrorEvent handler),
      or a transfer was attempted and didn't connect. Something went wrong that
      the caller experienced.
    - "incomplete": the call ended without reaching any terminal state. No tool
      set an outcome, which means nobody said goodbye -- the caller hung up
      mid-conversation. Also covers a call too short to have been anything.
    - "success": ran to a conclusion, including deliberately hanging up on a
      robocall. A spam call correctly identified and dropped is the system
      working, not failing.
    """
    if state.has_error or state.outcome == "transfer_failed":
        return "failed"
    if state.outcome is None:
        return "incomplete"
    if duration_seconds is not None and duration_seconds < 5:
        return "incomplete"
    return "success"


def transfer_attempted(state: CallState) -> bool:
    """A transfer either ran or it didn't -- the tool records which. True for a
    failed attempt too: the question is whether it was tried."""
    return state.outcome in ("department_transfer", "transfer_failed") or bool(
        state.matched_department
    )


def callback_needed(state: CallState) -> bool:
    """A callback number is only ever recorded by the record_callback_number
    tool, which the agent calls when it has arranged one."""
    return bool(state.transfer_failed_callback_number)


def _parse(raw: str) -> CallAnalysis:
    """Coerce the model's answer into the dataclass, field by field.

    Uses `json_repair` rather than `json.loads` because a model asked for bare
    JSON still occasionally wraps it in a fence or leaves a trailing comma, and
    losing an entire analysis to a stray character would be a poor trade.

    Every field is validated independently. A model that gets `priority` wrong
    shouldn't cost us the summary it got right.
    """
    text = (raw or "").strip()
    if not text:
        return CallAnalysis()

    data: Any
    try:
        data = json.loads(text)
    except ValueError:
        try:
            from json_repair import repair_json

            data = json.loads(repair_json(text))
        except Exception:  # noqa: BLE001
            logger.warning("call analysis wasn't parseable JSON: %r", text[:200])
            return CallAnalysis()

    if not isinstance(data, dict):
        logger.warning("call analysis wasn't a JSON object: %r", text[:200])
        return CallAnalysis()

    out = CallAnalysis()

    queries = data.get("user_queries")
    if isinstance(queries, list):
        out.user_queries = [
            cleaned
            for cleaned in (str(q).strip() for q in queries if q is not None)
            if cleaned
        ]

    summary = data.get("call_summary")
    if isinstance(summary, str) and summary.strip():
        out.call_summary = summary.strip()

    priority = data.get("priority")
    if isinstance(priority, str):
        # Title-cased so "high" and "HIGH" are both accepted; anything that
        # isn't one of the three is dropped rather than stored as a value the
        # enum column would reject anyway.
        candidate = priority.strip().title()
        if candidate in _PRIORITIES:
            out.priority = candidate  # type: ignore[assignment]
        elif priority.strip():
            logger.info("call analysis returned priority %r, which isn't valid", priority[:40])

    return out


async def analyse(
    transcript: str, provider: ProviderSettings, usage: Any | None = None
) -> CallAnalysis:
    """Ask the model for the three judgements. Never raises.

    `usage` is the call's `ModelUsageCollector`. Passed in so this LLM's own
    tokens land in the same bucket as the conversation's: it is a separate client
    rather than part of the AgentSession, so its metrics never reach the session's
    handler on their own. It's about $0.0007 a call -- immaterial next to $0.098,
    but a cost figure that silently omits one of its components is the wrong
    number, and the omission would be invisible.
    """
    if len(transcript.strip()) < _MIN_TRANSCRIPT_CHARS:
        logger.info("transcript too short to analyse (%d chars); skipping", len(transcript.strip()))
        return CallAnalysis()

    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="system", content=_PROMPT)
    chat_ctx.add_message(role="user", content=transcript)

    async def run(llm) -> str:
        parts: list[str] = []
        stream = llm.chat(chat_ctx=chat_ctx)
        try:
            async for chunk in stream:
                if chunk.delta and chunk.delta.content:
                    parts.append(chunk.delta.content)
        finally:
            await stream.aclose()
        return "".join(parts)

    try:
        llm = build_utility_llm(provider.analysis_llm, provider)
        if usage is not None:
            # Registered before the request so the one call it makes is counted.
            # Guarded because a metrics listener has no business being able to
            # lose the analysis it was only meant to measure.
            def _on_metrics(metrics) -> None:
                try:
                    usage.collect(metrics)
                except Exception:  # noqa: BLE001
                    logger.debug("couldn't record analysis LLM usage", exc_info=True)

            llm.on("metrics_collected", _on_metrics)
        raw = await asyncio.wait_for(run(llm), timeout=_TIMEOUT)
    except asyncio.TimeoutError:
        logger.warning(
            "call analysis didn't answer within %.0fs; the call is logged without it", _TIMEOUT
        )
        return CallAnalysis()
    except Exception:
        logger.exception("call analysis failed; the call is logged without it")
        return CallAnalysis()

    result = _parse(raw)
    logger.info(
        "call analysis: priority=%s queries=%d summary=%s",
        result.priority or "-",
        len(result.user_queries),
        f"{len(result.call_summary)} chars" if result.call_summary else "-",
    )
    return result
