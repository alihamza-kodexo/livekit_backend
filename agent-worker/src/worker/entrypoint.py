"""Worker entrypoint: one JobContext per inbound call.

Resolves which agent owns the call from the SIP-dialed number (the Supabase
lookup Project Plan v2 describes), builds the STT/LLM/TTS pipeline from that
agent's config, runs the call, then logs it -- directly to Supabase and Slack,
with no n8n hop, per the decision to skip n8n for this build.
"""

from __future__ import annotations

import json
import logging
import time

from livekit import rtc
from livekit.agents import (
    AgentSession,
    EndpointingOptions,
    InterruptionOptions,
    JobContext,
    JobProcess,
    RoomInputOptions,
    TurnHandlingOptions,
    WorkerOptions,
    cli,
)
from livekit.plugins import deepgram, google, groq, noise_cancellation, openai, silero

from . import deflection, notify
from .flow import InboundCallAgent, stt_keyterm_list
from .models import AgentConfig
from .settings import ProviderSettings, livekit_settings, provider_settings
from .state import CallState
from .supabase_client import insert_call_log, load_agent_config_by_id, load_agent_config_by_number

logger = logging.getLogger("worker.entrypoint")


def _test_agent_id_from_metadata(metadata: str) -> str | None:
    """The dashboard's "test this agent" button creates an explicit dispatch
    with `{"test_agent_id": "<uuid>"}` as the job metadata -- see
    dashboard/app/(protected)/agents/[agentId]/test-actions.ts. A real SIP
    call never carries this, so its absence is what selects the production path.
    """
    if not metadata:
        return None
    try:
        data = json.loads(metadata)
    except ValueError:
        return None
    value = data.get("test_agent_id") if isinstance(data, dict) else None
    return value if isinstance(value, str) and value else None


def _build_session_kwargs(config: AgentConfig, provider: ProviderSettings, vad: silero.VAD) -> dict:
    """The three conversation-engine options a dashboard admin can pick per
    agent -- see VOICE_STACK_DECISION.md for the cost/latency reasoning
    behind each.

    Groq and DeepSeek are both plain text LLMs slotted into the same
    Deepgram-STT / Deepgram-TTS pipeline (`vad`+`stt`+`llm`+`tts`). Gemini
    Live is a different shape: a speech-to-speech realtime model that
    replaces the *whole* pipeline, so it returns only `llm` -- no vad/stt/tts
    keys at all, and the framework handles audio in/out directly through it.
    That's also why Gemini Live gives up the pronunciation-dictionary
    substitution and most tuning knobs: there's no separate TTS step for
    `flow.py`'s `tts_node` override to hook into.
    """

    settings = config.agent.conversation_settings

    if config.agent.llm_provider == "gemini_live":
        if not provider.gemini_api_key:
            raise RuntimeError(
                f"agent {config.agent.agent_id} has llm_provider='gemini_live' but "
                f"GEMINI_API_KEY isn't set on this worker."
            )
        realtime_kwargs: dict = {
            "api_key": provider.gemini_api_key,
            "model": provider.gemini_model,
            "temperature": settings.temperature,
        }
        # Gemini Live's own prebuilt voice set, picked per agent in the
        # dashboard. Omitted entirely when unset so the plugin's default (Puck)
        # applies rather than us hardcoding a second default here.
        if config.agent.gemini_voice:
            realtime_kwargs["voice"] = config.agent.gemini_voice
        return {"llm": google.realtime.RealtimeModel(**realtime_kwargs)}

    if config.agent.llm_provider == "deepseek":
        if not provider.deepseek_api_key:
            raise RuntimeError(
                f"agent {config.agent.agent_id} has llm_provider='deepseek' but "
                f"DEEPSEEK_API_KEY isn't set on this worker."
            )
        llm = openai.LLM(
            api_key=provider.deepseek_api_key,
            base_url=provider.deepseek_base_url,
            model=provider.deepseek_model,
            temperature=settings.temperature,
            # deepseek-v4-flash reasons by default (measured: 174 vs 39 total
            # tokens for the same one-sentence answer). This kills the wasted
            # reasoning-token cost, but does NOT fix DeepSeek's own API latency
            # -- measured ~1.6s either way, still slow for live voice. See
            # VOICE_STACK_DECISION.md.
            extra_body={"thinking": {"type": "disabled"}},
        )
    else:
        if not provider.groq_api_key:
            raise RuntimeError(
                f"agent {config.agent.agent_id} has llm_provider='groq' but "
                f"GROQ_API_KEY isn't set on this worker."
            )
        llm = groq.LLM(
            api_key=provider.groq_api_key,
            model=provider.groq_model,
            temperature=settings.temperature,
        )

    # Deepgram for both ends of the audio pipeline -- Aura is ~2x cheaper than
    # Cartesia and needs no new vendor/API key beyond the STT one below. See
    # VOICE_STACK_DECISION.md for the full cost comparison. Aura has no speed
    # knob, so conversation_settings.speech_rate has no effect here (documented
    # there too, alongside tts_stability/backchannel_frequency).
    tts_kwargs: dict = {"api_key": provider.deepgram_api_key}
    if config.agent.voice_id:
        tts_kwargs["model"] = config.agent.voice_id

    return {
        "vad": vad,
        "stt": deepgram.STT(
            api_key=provider.deepgram_api_key,
            # NOT `or None` -- deepgram.STT does `list(keyterm)` unconditionally
            # when it isn't a str, which crashes on None. An empty list (no
            # pronunciation dictionary entries) is the correct "no boosting" value.
            keyterm=stt_keyterm_list(config),
        ),
        "llm": llm,
        "tts": deepgram.TTS(**tts_kwargs),
    }


def prewarm(proc: JobProcess) -> None:
    """Loads the VAD model once per worker process rather than once per call --
    it's the same model regardless of which agent picks up, so there's nothing
    agent-specific to redo per job."""
    proc.userdata["vad"] = silero.VAD.load()


def _turn_handling_from_settings(config: AgentConfig) -> TurnHandlingOptions:
    settings = config.agent.conversation_settings

    # interruption_sensitivity is 0..1 on the dashboard (higher = more sensitive);
    # min_duration is seconds of speech needed to register as an interruption
    # (lower = more sensitive). Linearly map across the SDK's own default range.
    min_duration = 1.0 - settings.interruption_sensitivity * 0.8

    return TurnHandlingOptions(
        endpointing=EndpointingOptions(min_delay=settings.vad_threshold_ms / 1000.0),
        interruption=InterruptionOptions(min_duration=max(0.1, min_duration)),
    )


async def entrypoint(ctx: JobContext) -> None:
    started_at = time.monotonic()

    test_agent_id = _test_agent_id_from_metadata(ctx.job.metadata)
    call_sid: str | None = None
    caller_number: str | None = None

    if test_agent_id:
        config = await load_agent_config_by_id(test_agent_id)
        if config is None:
            logger.error("test dispatch for unknown agent_id=%s", test_agent_id)
            ctx.delete_room()
            return
        # No kind filter -- the dashboard's test client joins as a standard
        # (non-SIP) participant. Testing a draft/paused agent is the whole
        # point of this path, so the production active-only gate below is
        # intentionally skipped for it.
        await ctx.wait_for_participant()
    else:
        participant = await ctx.wait_for_participant(kind=rtc.ParticipantKind.PARTICIPANT_KIND_SIP)
        dialed_number = participant.attributes.get("sip.trunkPhoneNumber")
        caller_number = participant.attributes.get("sip.phoneNumber")
        call_sid = participant.attributes.get("sip.twilio.callSid")

        if not dialed_number:
            logger.error(
                "SIP participant %s has no sip.trunkPhoneNumber attribute; can't resolve an agent",
                participant.identity,
            )
            ctx.delete_room()
            return

        config = await load_agent_config_by_number(dialed_number)
        if config is None:
            logger.error("no agent is assigned to dialed number %s", dialed_number)
            ctx.delete_room()
            return

        if config.agent.status != "active":
            logger.warning(
                "agent %s for number %s is %s, not active; refusing the call",
                config.agent.agent_id,
                dialed_number,
                config.agent.status,
            )
            ctx.delete_room()
            return

    state = CallState(
        config=config,
        room_name=ctx.room.name,
        call_sid=call_sid,
        caller_number=caller_number,
        is_test=bool(test_agent_id),
    )
    state.ai_deflection_index = deflection.start_call_offset()

    provider = provider_settings()
    vad = ctx.proc.userdata.get("vad") or silero.VAD.load()

    session: AgentSession[CallState] = AgentSession(
        userdata=state,
        turn_handling=_turn_handling_from_settings(config),
        **_build_session_kwargs(config, provider, vad),
    )

    async def _on_shutdown() -> None:
        await _log_and_notify(state, session, started_at)

    ctx.add_shutdown_callback(_on_shutdown)

    await session.start(
        InboundCallAgent(config),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVCTelephony()),
    )


async def _log_and_notify(state: CallState, session: AgentSession[CallState], started_at: float) -> None:
    duration_seconds = int(time.monotonic() - started_at)
    transcript = _render_transcript(session)

    await insert_call_log(
        call_sid=state.call_sid,
        room_id=state.room_name,
        agent_id=state.config.agent.agent_id,
        caller_number=state.caller_number,
        transcript=transcript,
        duration_seconds=duration_seconds,
        outcome=state.outcome,
        matched_department=state.matched_department,
        lead_name=state.lead_name,
        lead_company=state.lead_company,
        lead_need=state.lead_need,
        is_test=state.is_test,
    )

    # Fires even for test calls -- this is the agent owner's own configured
    # integration (e.g. a CRM sync), not an internal alert channel like Slack
    # below, and a test call is a real invocation of it just like a custom
    # tool call already is during testing.
    await notify.send_end_call_webhook(
        agent=state.config.agent,
        call_sid=state.call_sid,
        room_id=state.room_name,
        caller_number=state.caller_number,
        transcript=transcript,
        recording_url=None,
        duration_seconds=duration_seconds,
        outcome=state.outcome,
        matched_department=state.matched_department,
        lead_name=state.lead_name,
        lead_company=state.lead_company,
        lead_need=state.lead_need,
        qualification_answers=state.qualification_answers,
        is_test=state.is_test,
    )

    if state.is_test:
        # Dashboard test sessions aren't real calls -- they're logged above so
        # they show up (marked as such) in the dashboard, but Slack alerts are
        # for real customer calls only.
        logger.info("test session for agent %s logged, not notifying", state.config.agent.agent_id)
        return

    if state.outcome == "transfer_failed" and state.matched_department:
        await notify.send_transfer_failure_alert(
            agent=state.config.agent,
            caller_number=state.caller_number,
            department_name=state.matched_department,
            callback_number=state.transfer_failed_callback_number,
        )
    else:
        await notify.send_call_summary(
            agent=state.config.agent,
            caller_number=state.caller_number,
            outcome=state.outcome,
            duration_seconds=duration_seconds,
            matched_department=state.matched_department,
            lead_name=state.lead_name,
            lead_company=state.lead_company,
            lead_need=state.lead_need,
        )


def _render_transcript(session: AgentSession) -> str:
    lines = [
        f"{message.role}: {message.text_content}"
        for message in session.history.messages()
        if message.text_content
    ]
    return "\n".join(lines)


def main() -> None:
    settings = livekit_settings()
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name=settings.agent_name,
            ws_url=settings.url,
            api_key=settings.api_key,
            api_secret=settings.api_secret,
        )
    )


if __name__ == "__main__":
    main()
