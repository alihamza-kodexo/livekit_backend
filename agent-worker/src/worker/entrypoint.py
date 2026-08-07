"""Worker entrypoint: one JobContext per inbound call.

Resolves which agent owns the call from the SIP-dialed number (the Supabase
lookup Project Plan v2 describes), builds the STT/LLM/TTS pipeline from that
agent's config, runs the call, then logs it -- directly to Supabase and Slack,
with no n8n hop, per the decision to skip n8n for this build.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from google.genai import types as genai_types
from livekit import rtc
from livekit.agents import (
    AgentSession,
    EndpointingOptions,
    InterruptionOptions,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    PreemptiveGenerationOptions,
    RoomInputOptions,
    RoomOutputOptions,
    TurnHandlingOptions,
    WorkerOptions,
    cli,
    metrics,
    stt,
)
from livekit.plugins import deepgram, google, groq, openai, silero

from . import deflection, notify
from .flow import InboundCallAgent, stt_keyterm_list
from .models import AgentConfig, ConversationSettings
from .settings import ProviderSettings, livekit_settings, provider_settings
from .state import CallState
from .supabase_client import insert_call_log, load_agent_config_by_id, load_agent_config_by_number

logger = logging.getLogger("worker.entrypoint")

# Text-stream topic the dashboard's test panel listens on -- see
# dashboard/app/(protected)/agents/[agentId]/test-panel.tsx. Keep the string
# identical in both places; it's the only contract between them.
DIAGNOSTIC_TOPIC = "kodexo.diagnostic"


async def _report_diagnostic(ctx: JobContext, message: str) -> None:
    """Push a failure reason into the room so the dashboard can show it.

    Without this, every way a call can fail before the session starts -- a
    missing provider key, a paused agent, a number with no agent behind it --
    looks identical from the browser: a panel that says "Connecting" until the
    tester gives up. The failure was only ever visible in the worker's own
    stdout, which is on a different machine in production.

    Best-effort by definition, since it runs on paths where something has
    already gone wrong: it must never raise (that would mask the real error) and
    never stall teardown. A SIP caller can't read text streams, so for phone
    calls the log line is the record.
    """
    logger.error("call diagnostic: %s", message)
    try:
        await asyncio.wait_for(
            ctx.room.local_participant.send_text(message, topic=DIAGNOSTIC_TOPIC),
            timeout=2.0,
        )
    except Exception:  # noqa: BLE001 -- diagnostics must never mask the real failure
        logger.debug("couldn't publish the diagnostic into the room", exc_info=True)


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
            # Not optional on a phone call -- see the docstring below for why
            # this is the only place a Gemini Live agent's turn-taking can be
            # set at all.
            "realtime_input_config": _gemini_activity_detection(settings),
        }
        # Gemini Live's own prebuilt voice set, picked per agent in the
        # dashboard. Omitted entirely when unset so the plugin's default (Puck)
        # applies rather than us hardcoding a second default here.
        if config.agent.gemini_voice:
            realtime_kwargs["voice"] = config.agent.gemini_voice
        if provider.gemini_proactive_audio:
            realtime_kwargs["proactivity"] = True
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
        "stt": _build_stt(config, provider),
        "llm": llm,
        "tts": deepgram.TTS(**tts_kwargs),
    }


def _interruption_min_duration(settings: ConversationSettings) -> float:
    """Seconds of caller speech that have to land before it counts as barge-in.

    `interruption_sensitivity` is 0..1 on the dashboard (higher = more
    sensitive); this inverts it into seconds across the SDK's own default range.
    Shared by both engines so the slider means the same thing either way -- the
    pipeline hands it to InterruptionOptions, Gemini Live to its own detector.
    """
    return max(0.1, 1.0 - settings.interruption_sensitivity * 0.8)


def _gemini_activity_detection(
    settings: ConversationSettings,
) -> genai_types.RealtimeInputConfig:
    """Gemini Live's own turn detector, tuned for a phone line.

    This is not a refinement -- it's the only place a Gemini Live agent's
    turn-taking can be set at all. The plugin reports
    `can_disable_turn_detection=False` (its `RealtimeModel.session()` notes that
    Gemini drives manual turns through activity_start/activity_end, which the
    pipeline can't gatekeep yet), so on a realtime call *every* turn-taking
    decision is made server-side and AgentSession's endpointing and interruption
    options are discarded with a warning. Anything we want has to be said here.

    Gemini Live's own defaults are START_SENSITIVITY_HIGH and
    END_SENSITIVITY_HIGH -- the most eager setting on both ends. That is
    reasonable for a headset in a quiet room and wrong for an 8kHz phone leg,
    where room noise, a television or a second voice all read as speech. Since
    start-of-activity *is* Gemini's interrupt signal, every false positive stops
    the agent mid-sentence.
    """
    return genai_types.RealtimeInputConfig(
        automatic_activity_detection=genai_types.AutomaticActivityDetection(
            # LOW at both ends: harder to trip on noise, and slower to declare
            # the caller finished just because they paused for breath.
            start_of_speech_sensitivity=genai_types.StartSensitivity.START_SENSITIVITY_LOW,
            end_of_speech_sensitivity=genai_types.EndSensitivity.END_SENSITIVITY_LOW,
            # How long speech must persist before it commits as
            # start-of-activity. This is the actual noise gate, and the closest
            # thing Gemini has to InterruptionOptions.min_duration.
            prefix_padding_ms=int(_interruption_min_duration(settings) * 1000),
            # Floored at 500ms on purpose. The 300ms default exists for Flux,
            # which decides end-of-turn from the words themselves; Gemini's
            # detector is a silence timer with no semantic component, so 300ms
            # is just a caller drawing breath mid-sentence. Raising the
            # dashboard value above the floor still works.
            silence_duration_ms=int(max(500.0, settings.vad_threshold_ms)),
        )
    )


def _build_stt(config: AgentConfig, provider: ProviderSettings) -> stt.STT:
    """Deepgram STT, with Flux in front when it's enabled.

    Flux matters for one reason: it reports end-of-turn from the speech itself,
    so the session no longer has to sit on a fixed silence timer before handing
    the transcript to the LLM. On a phone call that timer is pure additive delay
    on top of PSTN transit, SIP transcoding and two jitter buffers, none of
    which we can shrink -- so it's the largest piece actually within reach.
    `eager_eot_threshold` goes further and lets the LLM start on a
    provisionally-complete utterance.

    Wrapped in a FallbackAdapter rather than swapped outright: if Flux is
    unavailable on the account or the socket fails mid-call, nova-3 takes over
    and still emits END_OF_SPEECH, so turn-taking keeps working -- just without
    the model-based decision. Set DEEPGRAM_STT_ENGINE=nova to skip Flux
    entirely.
    """
    nova = deepgram.STT(
        api_key=provider.deepgram_api_key,
        # NOT `or None` -- deepgram.STT does `list(keyterm)` unconditionally
        # when it isn't a str, which crashes on None. An empty list (no
        # pronunciation dictionary entries) is the correct "no boosting" value.
        keyterm=stt_keyterm_list(config),
    )

    if provider.stt_engine != "flux":
        return nova

    flux = deepgram.STTv2(
        api_key=provider.deepgram_api_key,
        model=provider.deepgram_flux_model,
        eot_threshold=provider.flux_eot_threshold,
        eager_eot_threshold=provider.flux_eager_eot_threshold,
        eot_timeout_ms=provider.flux_eot_timeout_ms,
        keyterm=stt_keyterm_list(config),
    )
    return stt.FallbackAdapter([flux, nova])


def _room_input_options() -> RoomInputOptions:
    """Attaches BVC noise cancellation only where it can actually run.

    Enhanced noise cancellation is a LiveKit Cloud feature: the filter asks the
    server to authorize it, and a self-hosted server has no such endpoint, so it
    answers 404 and the plugin logs "noise cancellation unavailable" every few
    seconds for the length of the call while doing nothing at all.

    Worth knowing when chasing background noise on a self-hosted deployment:
    there is no input-side suppression there whatsoever, so the only defences are
    the model's own turn detector (see `_gemini_activity_detection`) and whatever
    the caller's handset does.
    """
    if ".livekit.cloud" not in livekit_settings().url:
        logger.info(
            "self-hosted LiveKit (%s): skipping BVC noise cancellation, which is Cloud-only",
            livekit_settings().url,
        )
        return RoomInputOptions()

    # Imported here rather than at module scope on purpose: merely importing the
    # plugin makes its native filter probe the server for authorization, which
    # on a self-hosted deployment fails and logs a warning pair every few
    # seconds for the whole call. Keeping the import inside the Cloud branch is
    # what actually silences that, not just declining to pass the filter.
    from livekit.plugins import noise_cancellation

    return RoomInputOptions(noise_cancellation=noise_cancellation.BVCTelephony())


def prewarm(proc: JobProcess) -> None:
    """Loads the VAD model once per worker process rather than once per call --
    it's the same model regardless of which agent picks up, so there's nothing
    agent-specific to redo per job."""
    proc.userdata["vad"] = silero.VAD.load()


def _turn_handling_from_settings(
    config: AgentConfig, provider: ProviderSettings
) -> TurnHandlingOptions:
    """How the session decides a turn ended, and how eagerly it gets ahead.

    Only reaches the STT/LLM/TTS pipeline -- a Gemini Live agent returns early,
    because none of this is negotiable with a realtime model. Its equivalents
    live in `_gemini_activity_detection`.

    Three deliberate choices here, all aimed at the delay a caller actually
    feels between finishing a sentence and hearing a reply:

    - `turn_detection="stt"` when Flux is in use. Without it the session
      auto-selects VAD (it prefers VAD whenever a VAD model is passed, which it
      always is here for interruption detection), and Flux's end-of-turn signal
      would be ignored entirely.
    - `mode="dynamic"` endpointing shortens the wait when the utterance already
      sounds finished, instead of always spending the full delay.
    - `preemptive_tts` puts the *audio* generation in front of turn
      confirmation too. LLM preemption is already on by default in this SDK;
      this is the remaining half. It costs the occasional discarded synthesis
      when a caller resumes talking, in exchange for first-audio arriving
      sooner on the turns where they don't.
    """
    settings = config.agent.conversation_settings

    if config.agent.llm_provider == "gemini_live":
        # State what's actually true instead of passing options that get thrown
        # away: a realtime model owns turn-taking outright (see
        # _gemini_activity_detection), so endpointing, interruption and
        # preemptive generation have nothing here to act on -- the SDK logs a
        # warning and drops them. The equivalents are set on the model itself.
        return TurnHandlingOptions(turn_detection="realtime_llm")

    options = TurnHandlingOptions(
        endpointing=EndpointingOptions(
            mode="dynamic",
            min_delay=settings.vad_threshold_ms / 1000.0,
        ),
        interruption=InterruptionOptions(min_duration=_interruption_min_duration(settings)),
        preemptive_generation=PreemptiveGenerationOptions(preemptive_tts=True),
    )

    if provider.stt_engine == "flux":
        options["turn_detection"] = "stt"

    return options


def _log_turn_metrics(state: CallState, event: MetricsCollectedEvent) -> None:
    """Per-turn timings, so "the phone feels slower than the browser" can be
    answered with numbers instead of inference.

    `transport` is the whole point of logging this: the same agent on a SIP call
    and in a browser test runs an identical STT/LLM/TTS pipeline, so any
    difference has to be end-of-turn detection or the audio path. Comparing
    these lines across the two says which.
    """
    metrics.log_metrics(event.metrics)

    def timing(name: str) -> str | None:
        """Negative means "not measured for this engine", not "instant" -- a
        realtime model reports ttft=-1 on the turn that opens the session, and
        never reports end_of_utterance_delay at all, since it never had a
        separate endpointing step to measure. Printing -1.000 as if it were a
        latency reading is worse than saying nothing."""
        value = getattr(event.metrics, name, None)
        if not isinstance(value, (int, float)) or value < 0:
            return None
        return f"{value:.3f}"

    parts = {
        "eou": timing("end_of_utterance_delay"),
        "ttft": timing("ttft"),
        "ttfb": timing("ttfb"),
    }
    if not any(parts.values()):
        return

    logger.info(
        "turn timing transport=%s %s",
        "web" if state.is_test else "sip",
        " ".join(f"{key}={value}" for key, value in parts.items() if value),
    )


async def entrypoint(ctx: JobContext) -> None:
    """Reports why a call failed before re-raising, then lets the framework do
    its normal teardown. Anything raised in here used to surface only in the
    worker's log -- see `_report_diagnostic`."""
    try:
        await _run_call(ctx)
    except Exception as exc:
        await _report_diagnostic(ctx, f"{type(exc).__name__}: {exc}")
        raise


async def _run_call(ctx: JobContext) -> None:
    started_at = time.monotonic()

    test_agent_id = _test_agent_id_from_metadata(ctx.job.metadata)
    call_sid: str | None = None
    caller_number: str | None = None

    if test_agent_id:
        # Waits for the tester *before* loading the config, not after: a
        # diagnostic sent into an empty room is dropped, and "this agent no
        # longer exists" is exactly what the panel needs to be able to show.
        # No kind filter -- the dashboard's test client joins as a standard
        # (non-SIP) participant. Testing a draft/paused agent is the whole
        # point of this path, so the production active-only gate below is
        # intentionally skipped for it.
        await ctx.wait_for_participant()
        config = await load_agent_config_by_id(test_agent_id)
        if config is None:
            await _report_diagnostic(
                ctx, f"No agent exists with id {test_agent_id}. It may have been deleted."
            )
            ctx.delete_room()
            return
    else:
        participant = await ctx.wait_for_participant(kind=rtc.ParticipantKind.PARTICIPANT_KIND_SIP)
        dialed_number = participant.attributes.get("sip.trunkPhoneNumber")
        caller_number = participant.attributes.get("sip.phoneNumber")
        call_sid = participant.attributes.get("sip.twilio.callSid")

        # A SIP caller can't receive a text stream, so on this branch
        # _report_diagnostic is really just a uniform error log -- but it keeps
        # every "why did that call drop" reason phrased the same way in one place.
        if not dialed_number:
            await _report_diagnostic(
                ctx,
                f"SIP participant {participant.identity} has no sip.trunkPhoneNumber "
                f"attribute, so no agent can be resolved. Check the LiveKit inbound "
                f"trunk configuration.",
            )
            ctx.delete_room()
            return

        config = await load_agent_config_by_number(dialed_number)
        if config is None:
            await _report_diagnostic(
                ctx, f"No agent is assigned to the dialed number {dialed_number}."
            )
            ctx.delete_room()
            return

        if config.agent.status != "active":
            await _report_diagnostic(
                ctx,
                f"Agent {config.agent.name} ({config.agent.agent_id}) is "
                f"{config.agent.status}, not active, so the call to {dialed_number} "
                f"was refused.",
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
        turn_handling=_turn_handling_from_settings(config, provider),
        **_build_session_kwargs(config, provider, vad),
    )

    @session.on("metrics_collected")
    def _on_metrics(event: MetricsCollectedEvent) -> None:
        _log_turn_metrics(state, event)

    async def _on_shutdown() -> None:
        await _log_and_notify(state, session, started_at)

    ctx.add_shutdown_callback(_on_shutdown)

    await session.start(
        InboundCallAgent(config),
        room=ctx.room,
        room_input_options=_room_input_options(),
        # DTX off, RED on. DTX stops sending during silence, which is cheaper but
        # leaves the far side's adaptive jitter buffer without a steady stream to
        # converge on -- it then keeps a larger safety margin, and that margin is
        # added delay on every reply. RED adds redundant payloads so a lost
        # packet doesn't force the buffer to grow either. Both matter far more on
        # a SIP leg than on a browser's local WebRTC connection.
        room_output_options=RoomOutputOptions(
            audio_publish_options=rtc.TrackPublishOptions(
                source=rtc.TrackSource.SOURCE_MICROPHONE,
                dtx=False,
                red=True,
            ),
        ),
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

    # Reports zero load rather than just raising the ceiling: even 0.99 was
    # crossed on a developer machine, and a refused dispatch is a lost call when
    # there's no second worker to take it. See LiveKitSettings.load_threshold.
    load_options: dict = (
        {"load_fnc": lambda: 0.0}
        if settings.load_threshold is None
        else {"load_threshold": settings.load_threshold}
    )

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name=settings.agent_name,
            ws_url=settings.url,
            api_key=settings.api_key,
            api_secret=settings.api_secret,
            **load_options,
        )
    )


if __name__ == "__main__":
    main()
