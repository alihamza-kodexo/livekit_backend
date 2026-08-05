"""The inbound call flow -- FSD Section 3.3 (call classification + qualifying
sequence + special cases) and 3.5 (knowledge-base off-topic Q&A), built as one
Agent whose instructions are assembled from the agent's Supabase row.

NOTE: I don't have the FSD's exact Section 3.3 wording (the classification
categories, the qualifying-question order, the sales-pitch decline line, the
scam-termination trigger) -- the instructions below are built from the
Project Plan v2 Phase 5 task list, which names these same pieces but not their
exact scripted copy. Treat INSTRUCTIONS_TEMPLATE as the thing to check against
the real FSD text before launch, not as already-verified copy.

Deterministic actions (transfer, ending the call, recording data) are pushed
into tool calls rather than left to the LLM to narrate, so the parts that
matter for compliance and correctness happen in code -- the LLM only decides
*when* to call them.
"""

from __future__ import annotations

from collections.abc import AsyncIterable

from livekit.agents import Agent, ModelSettings

from . import tools
from .models import AgentConfig
from .pronunciation import stt_keyterms, substitute_stream

DEFAULT_END_CALL_GUIDANCE = (
    "End the call once qualification is complete and there's nothing further to transfer or "
    "discuss."
)

INSTRUCTIONS_TEMPLATE = """\
{custom_prompt}

## How this call must be handled

You are a phone receptionist. Callers cannot see you -- everything you convey \
must be said out loud. Keep replies to about {max_reply_sentences} sentence(s) at a time; \
a phone caller cannot absorb a long paragraph.

1. **Classify the call first**, from what the caller says after your greeting:
   - **Real inquiry** -- they want something legitimate related to this business. Move to \
qualification below.
   - **Wants a human / doesn't want an AI** -- ask exactly ONE clarifying question to find out \
which department they need, then call `transfer_to_department` immediately. Do not attempt to \
qualify them yourself.
   - **Sales pitch / vendor cold call** -- deliver a single polite decline (e.g. "we're not \
looking for that right now, but thank you"), then call `end_call` with outcome "not_qualified".
   - **Scam / fake-verification attempt** (asking you to confirm sensitive account details, \
pretending to be a bank/government agency, etc.) -- do not engage or answer their questions. End \
the call immediately with `end_call` and outcome "dropped".

2. **If it's a real inquiry, qualify them** by asking, one at a time, conversationally (not like \
a form):
{qualification_questions}
   Call `record_lead_info` as you learn each answer -- don't wait until the end.

3. **If the caller asks an off-topic question**, answer it briefly using the reference material \
below, then steer back to wherever you left off in the flow. Don't just refuse to answer.

4. **If the caller asks whether you're an AI/a bot/a real person**, answer honestly and briefly, \
then continue the call -- don't dodge the question and don't dwell on it either.

5. **Once qualification is complete**, decide the right department from the directory below \
based on what they need, tell the caller you're transferring them (say this out loud BEFORE \
calling the tool -- the transfer itself is silent), then call `transfer_to_department`.

6. **If `transfer_to_department` reports a failure**, apologize, explain that someone will call \
them back, ask for the best callback number, call `record_callback_number`, then `end_call`.

7. **If nothing qualifies for a transfer** (e.g. they just wanted information and got it), follow \
the end-of-call guidance below, close politely, then call `end_call` (outcome "not_qualified" \
unless the guidance says otherwise).

## Departments you can transfer to
{departments}

## Reference material for off-topic questions
{knowledge_base}

## When to end the call
This only affects step 7 above -- the scam-termination and failed-transfer cases in steps 1 and 6 \
always apply regardless of this guidance.

{end_call_instructions}
"""


def _build_instructions(config: AgentConfig) -> str:
    questions = "\n".join(
        f"   - {c.question}" + (" (required)" if c.required else "")
        for c in config.agent.qualification_criteria
    ) or "   - (No specific qualification questions configured -- use judgment based on the prompt above.)"

    departments = "\n".join(
        f"- **{d.department_name}** ({d.transfer_number})"
        + (f" -- routing hints: {d.routing_keywords}" if d.routing_keywords else "")
        for d in config.departments
    ) or "- (No departments configured -- if a transfer seems needed, apologize that no one is available and offer a callback.)"

    knowledge_base = "\n".join(
        f"### {entry.title}\n{entry.content}" for entry in config.knowledge_base
    ) or "(No knowledge base entries configured.)"

    return INSTRUCTIONS_TEMPLATE.format(
        custom_prompt=config.agent.prompt or f"You are {config.agent.name}.",
        max_reply_sentences=config.agent.conversation_settings.max_reply_sentences,
        qualification_questions=questions,
        departments=departments,
        knowledge_base=knowledge_base,
        end_call_instructions=config.agent.end_call_instructions or DEFAULT_END_CALL_GUIDANCE,
    )


class InboundCallAgent(Agent):
    """One Agent handles the whole call -- see the module docstring for why
    this is a single LLM-driven flow with tool calls for consequential
    actions, rather than several Agent subclasses handed off between."""

    def __init__(self, config: AgentConfig) -> None:
        self._pronunciation_dictionary = config.agent.pronunciation_dictionary
        self._first_message_mode = config.agent.first_message_mode
        self._first_message_text = config.agent.first_message_text
        super().__init__(
            instructions=_build_instructions(config),
            tools=[*tools.builtin_tools(), *tools.build_agent_tools(config.tools)],
        )

    async def on_enter(self) -> None:
        if self._first_message_mode == "user_starts":
            # Say nothing -- the session still listens continuously, so it
            # responds normally once the caller speaks first.
            return

        if self._first_message_mode == "agent_says_exact" and self._first_message_text:
            # session.say() goes straight to TTS, skipping the LLM entirely --
            # this is spoken verbatim, not ad-libbed from the prompt.
            self.session.say(self._first_message_text)
            return

        # Default ("agent_generates"): speaking the greeting explicitly (rather
        # than depending on the LLM to open on its own) means every call gets
        # one immediately, even if the model's first turn would otherwise wait
        # on caller input.
        self.session.generate_reply(
            instructions="Greet the caller briefly and ask how you can help."
        )

    def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ):
        return super().tts_node(
            substitute_stream(text, self._pronunciation_dictionary), model_settings
        )


def stt_keyterm_list(config: AgentConfig) -> list[str]:
    return stt_keyterms(config.agent.pronunciation_dictionary)
