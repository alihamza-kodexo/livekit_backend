# Kodexo Voice Agent -- Backend

Self-hosted, LiveKit-based inbound voice-AI platform: the Python (LiveKit
Agents SDK) worker that answers real calls via Twilio SIP trunking, the
self-hosted LiveKit + SIP infra it runs on, and the database schema both this
worker and the dashboard depend on.

The admin dashboard (Next.js) lives in a separate repo: [livekit_frontend](https://github.com/aiautomationkodexo/livekit_frontend).

- `agent-worker/` -- Python LiveKit agent worker (STT -> LLM -> TTS call flow)
- `infra/` -- self-hosted LiveKit + SIP bridge (docker-compose), backup scripts
- `supabase/` -- database schema migrations
