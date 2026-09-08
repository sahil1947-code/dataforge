# RIME_EVIDENCE.md
## Interruption Correctness — Claim, Procedure, and Measured Results

**Version:** 1.0
**Companion to:** README.md, TRD §3.9, PRD §9, benchmarks/interruption/

---

## 1. The Claim

> A user can interrupt RIME at any point — during Rime speech playback or
> during a running tool call — and the system will:
>
> 1. Stop Rime audio within **≤ 150 ms** of detected speech onset.
> 2. Discard every artifact (LLM output, tool result, audio chunk) tagged
>    with the superseded generation — those artifacts are **never spoken**
>    and **never written to the conversation as `status = 'spoken'`**.
> 3. Respond correctly to the new request, using only artifacts tagged
>    with the new generation.
>
> This must hold across **≥ 95 %** of stress-test trials, including trials
> where the tool call deliberately takes several seconds to return
> (simulating a slow network response).

This is the core engineering claim of the product. Every other feature
depends on this being true. The stress test below is the only acceptable
proof.

---

## 2. Why This Is Hard

Existing voice assistants fail this in one or more of three ways:

| Failure mode | What the user hears |
|---|---|
| Late tool result surfaces | The original (stale) answer is spoken after the user has already asked something new |
| Rime keeps playing | Audio continues for 1–3 s after the user starts speaking |
| State corruption | The session gets stuck — the assistant is still "thinking" about the old request while the new one is ignored |

RIME prevents all three with a single mechanism: **generation-ID fencing**
(TRD §3.9, `conversation/interruption.py`).

Every user turn increments a per-session `generation` counter. Every
downstream artifact — LLM token, tool result, Rime audio chunk — is tagged
with the generation it was produced for. The Response Manager checks this
tag on every artifact before anything reaches the TTS layer or the database.
If `artifact.generation != session.current_generation`, the artifact is
silently discarded. This check is O(1) and runs on every artifact without
exception.

---

## 3. How to Run the Stress Test

### Prerequisites

```bash
# From the Codespace directory, with the virtualenv active:
pip install -r requirements.txt
```

The stress test does **not** require a live microphone, a real Rime API key,
or a running LLM. It injects synthetic audio frames directly into the pipeline
and uses the stub Rime session, so it runs fully offline.

### Run

```bash
python benchmarks/interruption/stress_test.py
```

For a longer run (100 trials, the number used for submission evidence):

```bash
python benchmarks/interruption/stress_test.py --trials 100
```

To write results to a JSON file:

```bash
python benchmarks/interruption/stress_test.py --trials 50 --out benchmarks/interruption/results.json
```

### What the test does (one trial)

```
1. Create a fresh in-memory Session with generation counter = 0.
2. Submit a synthetic user request → generation increments to 1.
3. Start a "slow tool call" (TransportTool with _test_delay_seconds = 3.0).
4. After a randomised delay (0.1 – 1.5 s, before the tool returns),
   inject a synthetic barge-in signal → interrupt() is called.
5. Generation increments to 2. Rime session for gen 1 is cancelled.
6. The slow tool eventually returns a result tagged generation = 1.
7. PASS condition: the result is discarded (tool_calls.status = 'stale',
   no 'spoken' message row exists for generation 1).
8. Submit a new request under generation 2 (local calculator).
9. PASS condition: the calculator result IS spoken (persisted as 'spoken').
10. Measure: time from inject_interrupt() call to Rime cancel completion.
```

A trial is marked **CORRECT** when both PASS conditions hold. It is marked
**INCORRECT** if any stale artifact reaches `status = 'spoken'` — which
should never happen.

---

## 4. Acceptance Criteria

| Metric | Target | How measured |
|---|---|---|
| Interruption correctness rate | **≥ 95 %** | `correct_trials / total_trials` across the full run |
| Interrupt-to-audio-stop latency | **≤ 150 ms** | Mean and p95 across all trials |
| Time-to-first-audio (new request) | **≤ 800 ms** | From interrupt() call to first Rime chunk on the new generation |
| Stale artifacts surfaced | **0** | Count of artifacts that passed the fence with a superseded generation |

---

## 5. Reading the Results

The script prints a summary table and writes `results.json`. Example output:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RIME Interruption Stress Test — 50 trials
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Correct trials:          50 / 50
Correctness rate:        100.0 %   ✓  (target ≥ 95 %)

Interrupt → silence latency
  mean:                  18.3 ms
  p50:                   16.1 ms
  p95:                   38.7 ms   ✓  (target ≤ 150 ms)
  max:                   61.2 ms

Stale artifacts surfaced:  0       ✓  (must be 0)
New-request first audio:  <stub>   (requires live Rime key for real measurement)

Results written to benchmarks/interruption/results.json
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

> **Note on latency with real Rime:** The interrupt-to-silence figures above
> reflect the time to close the Rime WebSocket session (or cancel the stub).
> With a real Rime key and a live WebSocket connection, p95 will include
> actual network round-trip time. The `RIME_ENDPOINT` value in `.env` must
> point to the correct regional endpoint to minimise this.

---

## 6. Implementation Cross-Reference

The evidence chain from claim → code → test:

| Claim element | Code location | Test |
|---|---|---|
| Generation counter increments on interrupt | `conversation/generations.py · GenerationCounter.increment()` | `tests/unit/test_generations.py` |
| Artifact fence discards stale payloads | `conversation/interruption.py · ResponseManager.allow()` | `tests/unit/test_response_manager.py` |
| Tool calls marked `stale` in DB | `conversation/interruption.py · _mark_stale_tool_calls()` | `tests/unit/test_response_manager.py` |
| Rime session cancelled on interrupt | `speech/rime.py · RimeSession.cancel()` | `benchmarks/interruption/stress_test.py` |
| State machine transitions in ≤ 50 ms | `conversation/state_machine.py · interrupt()` | `tests/unit/test_state_machine.py` |
| Messages only written when generation is current | `conversation/interruption.py · persist_spoken_message()` | `tests/unit/test_response_manager.py` |
| Full pipeline: barge-in during tool call | `api/pipeline.py · handle_interrupt()` | `tests/integration/test_interrupt_pipeline.py` |

---

## 7. Rime Integration Evidence

The following parameters are pinned in `.env` and verified against the live
Rime catalog. They must be re-verified before each submission milestone.

| Parameter | `.env` key | Value |
|---|---|---|
| Model ID | `RIME_MODEL_ID` | *(set in .env)* |
| Speaker / voice | `RIME_SPEAKER` | *(set in .env)* |
| Language | `RIME_LANGUAGE` | *(set in .env)* |
| Transport | — | WebSocket streaming |
| Audio format | `RIME_AUDIO_FORMAT` | *(set in .env)* |
| Endpoint | `RIME_ENDPOINT` | *(set in .env)* |

Rime is used as the **primary and only** spoken output throughout every
interaction — including during normal turns, tool-result delivery,
confirmation prompts, memory confirmations, and the emergency modal
announcement. It is never reduced to a static chime or a single welcome
message.

---

## 8. Reproducibility Checklist

Before submitting results from a stress-test run, verify:

- [ ] Test was run with `--trials 50` or more
- [ ] `results.json` is committed alongside this file
- [ ] The run was performed without a live microphone (synthetic frames only) so results are deterministic
- [ ] Interrupt delay range covers both very-fast interrupts (100 ms) and mid-wait interrupts (1.5 s)
- [ ] `RIME_API_KEY` is **not** present in `results.json` or any committed artifact
- [ ] The `schema_version` in the database matches what `schema.sql` creates

---

*End of RIME_EVIDENCE.md*
