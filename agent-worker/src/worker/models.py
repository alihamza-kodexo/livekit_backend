"""Row shapes for the Supabase schema in supabase/migrations/0001_init_schema.sql.

Mirrors dashboard/lib/types.ts -- keep the two in sync if the schema changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

AgentStatus = Literal["active", "paused", "draft"]

# Switchable per-agent from the dashboard, no worker redeploy needed -- see
# VOICE_STACK_DECISION.md. Groq is fast; DeepSeek is cheaper per-token but
# slower. "gemini_live" is a different shape entirely: a speech-to-speech
# realtime model that replaces the whole STT+LLM+TTS pipeline, not just the
# LLM leg -- see entrypoint.py's session-building branch and the tradeoffs
# documented in VOICE_STACK_DECISION.md before enabling it on a live agent.
LLMProvider = Literal["groq", "deepseek", "gemini_live"]

# How the agent opens a call -- see flow.py's on_enter.
FirstMessageMode = Literal["agent_generates", "agent_says_exact", "user_starts"]

CallOutcome = Literal[
    "qualified",
    "department_transfer",
    "not_qualified",
    "transfer_failed",
    "dropped",
]


@dataclass(frozen=True)
class QualificationCriterion:
    key: str
    question: str
    required: bool

    @staticmethod
    def from_row(row: dict[str, Any]) -> "QualificationCriterion":
        return QualificationCriterion(
            key=row["key"], question=row["question"], required=bool(row.get("required"))
        )


@dataclass(frozen=True)
class PronunciationEntry:
    term: str
    say_as: str

    @staticmethod
    def from_row(row: dict[str, Any]) -> "PronunciationEntry":
        return PronunciationEntry(term=row["term"], say_as=row["say_as"])


# Mirrors CONVERSATION_SETTING_DEFAULTS in dashboard/lib/types.ts -- what an
# unset field resolves to. Keep the numbers identical in both places: the
# dashboard shows these as placeholder text, so a drift here makes that text
# a lie about what actually happens on a call.
CONVERSATION_SETTING_DEFAULTS: dict[str, float] = {
    "temperature": 0.7,
    "max_reply_sentences": 2,
    "tts_stability": 0.6,
    "speech_rate": 1.0,
    "vad_threshold_ms": 500,
    "interruption_sensitivity": 0.5,
    "backchannel_frequency": 0.2,
}


@dataclass(frozen=True)
class ConversationSettings:
    temperature: float
    max_reply_sentences: int
    tts_stability: float
    speech_rate: float
    vad_threshold_ms: float
    interruption_sensitivity: float
    backchannel_frequency: float

    @staticmethod
    def from_row(row: dict[str, Any]) -> "ConversationSettings":
        merged = {**CONVERSATION_SETTING_DEFAULTS, **(row or {})}
        return ConversationSettings(
            temperature=float(merged["temperature"]),
            max_reply_sentences=int(merged["max_reply_sentences"]),
            tts_stability=float(merged["tts_stability"]),
            speech_rate=float(merged["speech_rate"]),
            vad_threshold_ms=float(merged["vad_threshold_ms"]),
            interruption_sensitivity=float(merged["interruption_sensitivity"]),
            backchannel_frequency=float(merged["backchannel_frequency"]),
        )


@dataclass(frozen=True)
class Agent:
    agent_id: str
    name: str
    twilio_number: str | None
    status: AgentStatus
    prompt: str
    first_message_mode: FirstMessageMode
    # Only used when first_message_mode is "agent_says_exact".
    first_message_text: str | None
    qualification_criteria: list[QualificationCriterion]
    stt_provider: str
    tts_provider: str
    llm_provider: LLMProvider
    voice_id: str | None
    pronunciation_dictionary: list[PronunciationEntry]
    conversation_settings: ConversationSettings
    # Null means "use flow.py's default end-of-call guidance". Doesn't change what
    # end_call *does* (still just hangs up) -- only the conditions the model is
    # told to watch for before calling it.
    end_call_instructions: str | None

    @staticmethod
    def from_row(row: dict[str, Any]) -> "Agent":
        return Agent(
            agent_id=row["agent_id"],
            name=row["name"],
            twilio_number=row.get("twilio_number"),
            status=row["status"],
            prompt=row.get("prompt") or "",
            first_message_mode=row.get("first_message_mode") or "agent_generates",  # type: ignore[assignment]
            first_message_text=row.get("first_message_text"),
            qualification_criteria=[
                QualificationCriterion.from_row(r) for r in row.get("qualification_criteria") or []
            ],
            stt_provider=row.get("stt_provider") or "deepgram",
            tts_provider=row.get("tts_provider") or "deepgram",
            llm_provider=row.get("llm_provider") or "groq",  # type: ignore[assignment]
            voice_id=row.get("voice_id"),
            pronunciation_dictionary=[
                PronunciationEntry.from_row(r) for r in row.get("pronunciation_dictionary") or []
            ],
            conversation_settings=ConversationSettings.from_row(row.get("conversation_settings") or {}),
            end_call_instructions=row.get("end_call_instructions"),
        )


@dataclass(frozen=True)
class Department:
    department_id: str
    agent_id: str
    department_name: str
    transfer_number: str
    routing_keywords: str | None

    @staticmethod
    def from_row(row: dict[str, Any]) -> "Department":
        return Department(
            department_id=row["department_id"],
            agent_id=row["agent_id"],
            department_name=row["department_name"],
            transfer_number=row["transfer_number"],
            routing_keywords=row.get("routing_keywords"),
        )


@dataclass(frozen=True)
class KnowledgeBaseEntry:
    kb_id: str
    agent_id: str
    title: str
    content: str

    @staticmethod
    def from_row(row: dict[str, Any]) -> "KnowledgeBaseEntry":
        return KnowledgeBaseEntry(
            kb_id=row["kb_id"], agent_id=row["agent_id"], title=row["title"], content=row["content"]
        )


@dataclass(frozen=True)
class Tool:
    tool_id: str
    agent_id: str
    name: str
    description: str
    parameter_schema: dict[str, Any]
    webhook_url: str | None
    is_builtin: bool

    @staticmethod
    def from_row(row: dict[str, Any]) -> "Tool":
        return Tool(
            tool_id=row["tool_id"],
            agent_id=row["agent_id"],
            name=row["name"],
            description=row["description"],
            parameter_schema=row.get("parameter_schema") or {},
            webhook_url=row.get("webhook_url"),
            is_builtin=bool(row.get("is_builtin")),
        )


@dataclass
class AgentConfig:
    """Everything the worker needs to run one call, loaded once per job."""

    agent: Agent
    departments: list[Department]
    knowledge_base: list[KnowledgeBaseEntry]
    tools: list[Tool] = field(default_factory=list)
