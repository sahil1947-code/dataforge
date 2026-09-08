# RIME — Persistent Voice Operating System

A local-first, identity-aware voice agent that maintains conversational continuity through interruptions and long-running tasks, with Rime providing the primary real-time spoken experience.

---

## What this is

RIME is not a chatbot with a voice skin. It is a voice operating system built around three hard engineering problems that existing assistants do not solve:

1. **Interruption correctness** — the user can barge in at any point (during speech, during a running tool call) and the system recovers to a correct conversational state every time. Stale results are never spoken.
2. **Persistent speaker identity** — the system recognises who is speaking from voice alone, loads their context, and remembers them across sessions without any login step.
3. **Offline-first operation** — all Tier-1 capabilities (calculation, tasks, memory, local STT/LLM) work with no network connection. External services are used only when genuinely necessary.

---

## Quick start

### 1. Clone and set up the environment

```bash
cd "Rime PS2/Codespace"
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure secrets

```bash
cp .env.example .env
```

Open `.env` and fill in the values that apply to your setup:

| Key | Required | Notes |
|-----|----------|-------|
| `RIME_API_KEY` | **Yes** | Get from [rime.ai](https://rime.ai). Pin the exact model, speaker, and endpoint from the live catalog. |
| `RIME_MODEL_ID` | **Yes** | Exact model ID from the Rime catalog at submission time. Default: `mist` |
| `RIME_SPEAKER` | **Yes** | Exact voice name. Default: `lagoon` |
| `RIME_ENDPOINT` | **Yes** | WebSocket endpoint. Default: `wss://users.rime.ai/v1/rime-tts` |
| `OLLAMA_BASE_URL` | Recommended | Local LLM via Ollama. Leave blank to use the stub agent. |
| `OLLAMA_MODEL` | Recommended | e.g. `llama3`, `mistral` |
| `OPENAI_API_KEY` | Optional | External LLM fallback only — used only when Ollama is unreachable. |
| `WEATHER_API_KEY` | Optional | OpenWeatherMap key for live weather tool. |

> **Security**: Never commit your real `.env` file. The `.env.example` ships with placeholders only.

### 3. Install a local STT model (optional but recommended)

```bash
# faster-whisper (recommended — lower latency on CPU)
pip install faster-whisper

# OR standard openai-whisper
pip install openai-whisper
```

The system falls back to a clearly-labeled stub if neither is installed. Whisper model weights are downloaded automatically on first use to `~/.cache/whisper`.

### 4. Install and start Ollama (optional but recommended)

```bash
# Install from https://ollama.ai, then:
ollama pull llama3
ollama serve
```

The agent falls back to OpenAI (if `OPENAI_API_KEY` is set) or a plain stub if Ollama is unreachable.

### 5. Run

```bash
python main.py
```

Or with uvicorn directly (enables hot reload during development):

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
```

Open your browser at **http://localhost:8080** to see the companion UI with all five surfaces.

The WebSocket session endpoint is at **ws://localhost:8080/ws/session**.

---

## Architecture overview

```
USER VOICE
  → Audio Capture (in-memory ring buffer, never persisted)
  → VAD (Silero — local, ≤30 ms/frame)
      → Speaker Identification (ECAPA-TDNN — local, ≤200 ms)
  → Session Manager (state machine + generation counter)
  → STT (Whisper — local, ≤300 ms)
  → Intent Router (pattern → LOCAL_TOOL | LIVE_DATA | AGENT_ONLY)
      → Local tools (calculator, time, tasks, memory)
      → External tools via ExternalServiceManager (weather, transport)
  → Agent / LLM (Ollama local-first → OpenAI fallback → stub)
  → Response Manager  ← THE CORRECTNESS GATE
      • Every artifact tagged with generation ID
      • Stale artifacts (from superseded requests) discarded here — never spoken
  → Rime TTS (WebSocket streaming — primary and only spoken output)
  → USER HEARS RESPONSE
```

### Key modules

| Module | Responsibility |
|--------|---------------|
| `conversation/interruption.py` | **Response Manager** — generation-ID fencing, the core correctness guarantee |
| `conversation/state_machine.py` | Session state machine (7 states, interrupt from any state in ≤50 ms) |
| `conversation/generations.py` | Per-session generation counter — O(1) check on every artifact |
| `speaker/matcher.py` | 3-tier confidence policy (confident / tentative / unknown) |
| `speech/rime.py` | Rime WebSocket streaming + mid-utterance cancel |
| `routing/external.py` | ExternalServiceManager — the only module allowed to make outbound calls |
| `memory/cleanup.py` | Atomic "forget me" cascade across all profile-scoped tables |
| `safety/emergency.py` | Distress classifier — never triggers autonomous external action |

---

## Project structure

```
Codespace/
├── main.py                     ← Entry point
├── config.py                   ← All settings (loaded from .env)
├── requirements.txt
├── .env.example                ← Copy to .env and fill in secrets
│
├── audio/                      ← Microphone capture + VAD
├── speaker/                    ← Embedding, enrollment, matching, profiles
├── speech/                     ← STT (Whisper) + Rime TTS
├── conversation/               ← State machine, generation IDs, Response Manager, Session
├── memory/                     ← 4-level memory model + cleanup
├── tasks/                      ← Task manager + async scheduler
├── tools/                      ← Tool registry + 6 tools
├── routing/                    ← Intent router, local/external dispatch, permissions
├── safety/                     ← Emergency detection, privacy mode, confirmations
├── agent/                      ← Reasoning core (Ollama → OpenAI → stub)
├── observability/              ← Structured logging, pipeline traces, dashboard data
├── api/                        ← FastAPI app, WebSocket handler, REST routes, pipeline
├── database/                   ← SQLite schema, migrations, async connection manager
├── frontend/templates/         ← Companion UI (5 surfaces in one HTML file)
└── benchmarks/interruption/    ← Stress test harness
```

---

## Rime configuration (pinned values)

The following Rime parameters must be pinned at submission time and match what is in `.env`. They are verified against the live Rime catalog, not copied from stale documentation.

| Parameter | Value (set in .env) |
|-----------|---------------------|
| Model ID | `RIME_MODEL_ID` |
| Speaker / voice | `RIME_SPEAKER` |
| Language | `RIME_LANGUAGE` |
| Transport | WebSocket (streaming) |
| Audio format | `RIME_AUDIO_FORMAT` |
| Endpoint | `RIME_ENDPOINT` |

---

## Running the interruption stress test

The stress test proves the core claim: stale tool results are never spoken after a barge-in, across 50+ consecutive trials.

```bash
python benchmarks/interruption/stress_test.py
```

Results are written to `benchmarks/interruption/results.json` and printed to the console. See `RIME_EVIDENCE.md` for the acceptance criteria and how to interpret the output.

---

## Running tests

```bash
# All tests
pytest tests/ -v

# Unit tests only (fast, no network, no models)
pytest tests/unit/ -v

# Specific test suite
pytest tests/unit/test_generations.py -v
pytest tests/unit/test_response_manager.py -v
pytest tests/memory/ -v
pytest tests/speaker/ -v
```

---

## Memory levels

| Level | Name | Lifetime | Example |
|-------|------|----------|---------|
| 0 | Temporary | Discarded after processing | Literal words just spoken |
| 1 | Session | Discarded at session end | "We were discussing trains" |
| 2 | Short-term | 30 days (configurable) | Recent conversation summaries |
| 3 | Persistent | Until explicit forget | "My preferred language is Hindi" |

---

## Privacy controls

All privacy settings are controllable by voice and mirrored in the Privacy Center UI:

| Command | Effect |
|---------|--------|
| "Privacy mode" | Disables speaker profiling, memory writes, audio retention |
| "Exit privacy mode" | Restores normal operation |
| "Don't remember this conversation" | Session data destroyed at end |
| "Clear my history" | Full cascading deletion of all profile data (requires confirmation) |

---

## Security notes

- All API keys are server-side only, loaded from `.env` via `config.py`.
- No credential is ever read directly by any module other than `ExternalServiceManager` (`routing/external.py`).
- Voice embeddings are stored locally and never transmitted to any external service.
- Audio is never written to disk — the ring buffer is in-memory only.
- The `.env.example` ships with placeholder values only.
