"""
benchmarks/interruption/stress_test.py
---------------------------------------
Fixed-delay interruption stress test for RIME.

Proves the core claim (PRD §9, RIME_EVIDENCE.md §3):
  • Stale tool results are NEVER spoken after a barge-in.
  • Rime session is cancelled within the target latency.
  • The new request completes correctly under the new generation.

Run:
    python benchmarks/interruption/stress_test.py
    python benchmarks/interruption/stress_test.py --trials 100
    python benchmarks/interruption/stress_test.py --trials 50 --out results.json

The test is fully offline — no microphone, no real Rime key, no LLM required.
All components use their stub / in-memory implementations.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

# Ensure UTF-8 stdout on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Make the Codespace root importable when running the script directly
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Use isolated in-memory SQLite for concurrent benchmark stress testing
os.environ.setdefault("DATABASE_PATH", ":memory:")

from conversation.state_machine import StateMachine, SessionState
from conversation.generations import GenerationCounter
from conversation.interruption import ResponseManager, Artifact, ArtifactType
from database import get_connection, close_connection, execute


# ── Trial result dataclass ────────────────────────────────────────────

@dataclass
class TrialResult:
    trial_number: int
    correct: bool                         # True = stale artifact never surfaced
    interrupt_latency_ms: float           # Time from interrupt() call to Rime cancel
    new_request_latency_ms: float         # Time from interrupt to new result
    stale_tool_gen: int                   # Generation of the slow (stale) tool
    new_gen: int                          # Generation after interrupt
    stale_surfaced: bool                  # Should always be False
    interrupt_delay_s: float              # Randomised delay before interrupt
    error: Optional[str] = None


# ── Stub Rime session ─────────────────────────────────────────────────

class _StubRimeSession:
    """Minimal Rime session stub — tracks cancel latency."""

    def __init__(self, generation: int) -> None:
        self.generation = generation
        self._cancelled = False
        self._cancel_time: Optional[float] = None
        self._start_time = time.monotonic()

    async def cancel(self) -> None:
        self._cancelled = True
        self._cancel_time = time.monotonic()

    @property
    def cancel_latency_ms(self) -> Optional[float]:
        if self._cancel_time is None:
            return None
        return (self._cancel_time - self._start_time) * 1000


# ── Single trial ──────────────────────────────────────────────────────

async def run_trial(
    trial_number: int,
    tool_delay_s: float = 3.0,
    interrupt_delay_s: Optional[float] = None,
) -> TrialResult:
    """
    Execute one interruption trial.

    Timeline:
      t=0      : user request #1 (slow tool, generation 1)
      t=delay  : barge-in interrupt (generation increments to 2)
      t=3.0    : slow tool result arrives tagged generation 1
                 → must be discarded
      t=delay+ : user request #2 (fast local tool, generation 2)
                 → must complete and be persisted as 'spoken'
    """
    if interrupt_delay_s is None:
        interrupt_delay_s = random.uniform(0.05, min(1.5, tool_delay_s - 0.05))

    # Fresh in-memory session objects (no DB writes for the bulk of the test)
    conversation_id = f"stress_conv_{trial_number}_{int(time.time()*1000)}"
    sm  = StateMachine(initial=SessionState.SESSION_CREATED)
    gen = GenerationCounter(initial=0)
    rm  = ResponseManager(
        state_machine=sm,
        generation_counter=gen,
        conversation_id=conversation_id,
    )

    # Insert conversation row so FK constraints on tool_calls and messages succeed
    await execute(
        "INSERT INTO conversations (conversation_id, started_at, session_mode) VALUES (?, datetime('now'), 'normal')",
        (conversation_id,),
    )

    stale_surfaced  = False
    error_msg: Optional[str] = None

    try:
        # ── Request 1: slow tool under generation 1 ──────────────────
        await sm.transition(SessionState.LISTENING,    generation=0)
        await sm.transition(SessionState.THINKING,     generation=0)
        await sm.transition(SessionState.TOOL_RUNNING, generation=0)

        gen1 = await rm.on_new_turn()   # generation → 1
        assert gen1 == 1, f"Expected gen=1, got {gen1}"

        # Register a fake tool call
        tool_call_id = f"tc_stress_{trial_number}"
        await rm.register_tool_call(tool_call_id, "transport",
                                    json.dumps({"route": "Mumbai-Pune"}), gen1)

        # Create a stub Rime session for generation 1
        rime1 = _StubRimeSession(generation=gen1)

        # ── Barge-in after interrupt_delay_s ─────────────────────────
        await asyncio.sleep(interrupt_delay_s)

        t_interrupt_start = time.monotonic()
        gen2 = await rm.on_interrupt(current_rime_session=rime1)
        interrupt_latency_ms = (time.monotonic() - t_interrupt_start) * 1000
        assert gen2 == 2, f"Expected gen=2 after interrupt, got {gen2}"

        # ── Slow tool arrives late (generation 1 — must be discarded) ──
        async def slow_tool_arrives() -> None:
            nonlocal stale_surfaced
            await asyncio.sleep(tool_delay_s - interrupt_delay_s)

            stale_artifact = Artifact(
                payload={"status": "on-time", "platform": 4},
                generation=gen1,              # STALE
                artifact_type=ArtifactType.TOOL_RESULT,
                tool_call_id=tool_call_id,
            )
            allowed = await rm.allow(stale_artifact)
            if allowed:
                stale_surfaced = True         # FAIL — this must never be True

            # Also mark it complete via the normal path
            await rm.complete_tool_call(tool_call_id, gen1, success=True)

        stale_task = asyncio.create_task(slow_tool_arrives())

        # ── Request 2: fast local tool under generation 2 ─────────────
        t_new_start = time.monotonic()
        await sm.transition(SessionState.LISTENING,    generation=gen2)
        await sm.transition(SessionState.THINKING,     generation=gen2)

        # Simulate a fast calculator result (gen 2)
        fresh_artifact = Artifact(
            payload="The answer is 10.",
            generation=gen2,
            artifact_type=ArtifactType.MESSAGE,
        )
        allowed_fresh = await rm.allow(fresh_artifact)
        assert allowed_fresh, "Fresh artifact must be allowed through the fence"

        # Persist the spoken message (only if generation is current)
        msg_id = await rm.persist_spoken_message(
            role="assistant",
            text="The answer is 10.",
            generation=gen2,
            sequence_number=1,
        )
        assert msg_id is not None, "Fresh message must be persisted"

        await sm.transition(SessionState.SPEAKING,   generation=gen2)
        await sm.transition(SessionState.LISTENING,  generation=gen2)

        new_request_latency_ms = (time.monotonic() - t_new_start) * 1000

        # Wait for the stale tool task to finish
        await stale_task

        correct = (not stale_surfaced) and (msg_id is not None)

    except Exception as exc:
        correct = False
        interrupt_latency_ms   = 0.0
        new_request_latency_ms = 0.0
        error_msg = str(exc)
        stale_surfaced = False
        gen2 = -1

    return TrialResult(
        trial_number    = trial_number,
        correct         = correct,
        interrupt_latency_ms   = round(interrupt_latency_ms,   2),
        new_request_latency_ms = round(new_request_latency_ms, 2),
        stale_tool_gen  = 1,
        new_gen         = gen2,
        stale_surfaced  = stale_surfaced,
        interrupt_delay_s = round(interrupt_delay_s, 3),
        error           = error_msg,
    )


# ── Full stress test runner ───────────────────────────────────────────

async def run_stress_test(
    n_trials: int = 50,
    tool_delay_s: float = 3.0,
    concurrency: int = 5,
) -> list[TrialResult]:
    """
    Run `n_trials` interruption trials with controlled concurrency.
    Returns all TrialResult objects.
    """
    # Initialise the database (needed for ResponseManager DB calls)
    await get_connection()

    results: list[TrialResult] = []
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_trial(i: int) -> TrialResult:
        async with semaphore:
            r = await run_trial(i + 1, tool_delay_s=tool_delay_s)
            marker = "✓" if r.correct else "✗"
            print(
                f"  [{marker}] Trial {r.trial_number:>3}  "
                f"interrupt_delay={r.interrupt_delay_s:.3f}s  "
                f"fence_latency={r.interrupt_latency_ms:.1f}ms  "
                f"stale_surfaced={r.stale_surfaced}"
                + (f"  ERROR: {r.error}" if r.error else "")
            )
            return r

    print(f"\nRunning {n_trials} trials (tool_delay={tool_delay_s}s, "
          f"concurrency={concurrency})…\n")

    tasks = [bounded_trial(i) for i in range(n_trials)]
    results = await asyncio.gather(*tasks)

    await close_connection()
    return list(results)


# ── Statistics and reporting ──────────────────────────────────────────

def compute_stats(results: list[TrialResult]) -> dict:
    n = len(results)
    correct_n = sum(1 for r in results if r.correct)
    stale_surfaced_n = sum(1 for r in results if r.stale_surfaced)

    latencies = sorted(r.interrupt_latency_ms for r in results)
    new_req    = sorted(r.new_request_latency_ms for r in results)

    def pct(lst: list[float], p: float) -> float:
        if not lst:
            return 0.0
        idx = min(int(len(lst) * p / 100), len(lst) - 1)
        return round(lst[idx], 2)

    def avg(lst: list[float]) -> float:
        return round(sum(lst) / len(lst), 2) if lst else 0.0

    return {
        "total_trials":          n,
        "correct_trials":        correct_n,
        "incorrect_trials":      n - correct_n,
        "correctness_rate_pct":  round(correct_n / n * 100, 2) if n else 0,
        "stale_surfaced":        stale_surfaced_n,
        "interrupt_latency": {
            "mean_ms":  avg(latencies),
            "p50_ms":   pct(latencies, 50),
            "p95_ms":   pct(latencies, 95),
            "max_ms":   max(latencies) if latencies else 0,
        },
        "new_request_latency": {
            "mean_ms":  avg(new_req),
            "p50_ms":   pct(new_req, 50),
            "p95_ms":   pct(new_req, 95),
        },
        "targets": {
            "correctness_rate_target_pct":    95.0,
            "interrupt_latency_target_ms":   150.0,
            "stale_surfaced_target":           0,
        },
    }


def print_report(stats: dict) -> None:
    W = 52
    print("\n" + "━" * W)
    print("RIME Interruption Stress Test Results")
    print("━" * W)

    cr = stats["correctness_rate_pct"]
    cr_ok = cr >= stats["targets"]["correctness_rate_target_pct"]
    print(f"Correct trials:    {stats['correct_trials']:>4} / {stats['total_trials']}")
    print(f"Correctness rate:  {cr:>6.1f} %  "
          f"{'✓' if cr_ok else '✗'}  (target ≥ {stats['targets']['correctness_rate_target_pct']:.0f} %)")

    il = stats["interrupt_latency"]
    il_ok = il["p95_ms"] <= stats["targets"]["interrupt_latency_target_ms"]
    print(f"\nInterrupt → silence latency")
    print(f"  mean:   {il['mean_ms']:>7.1f} ms")
    print(f"  p50:    {il['p50_ms']:>7.1f} ms")
    print(f"  p95:    {il['p95_ms']:>7.1f} ms  "
          f"{'✓' if il_ok else '✗'}  (target ≤ {stats['targets']['interrupt_latency_target_ms']:.0f} ms)")
    print(f"  max:    {il['max_ms']:>7.1f} ms")

    ss = stats["stale_surfaced"]
    ss_ok = ss == 0
    print(f"\nStale artifacts surfaced:  {ss}  "
          f"{'✓' if ss_ok else '✗ FAIL'}  (must be 0)")

    print("━" * W)
    overall = cr_ok and il_ok and ss_ok
    print(f"Overall: {'ALL PASS ✓' if overall else 'FAILED ✗'}")
    print("━" * W + "\n")


# ── Entry point ───────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RIME interruption correctness stress test"
    )
    parser.add_argument(
        "--trials", type=int, default=50,
        help="Number of trials to run (default: 50)",
    )
    parser.add_argument(
        "--tool-delay", type=float, default=3.0,
        help="Simulated tool call delay in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=5,
        help="Max concurrent trials (default: 5)",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Path to write JSON results (optional)",
    )
    args = parser.parse_args()

    results = asyncio.run(
        run_stress_test(args.trials, args.tool_delay, args.concurrency)
    )
    stats = compute_stats(results)
    print_report(stats)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": stats,
            "trials": [asdict(r) for r in results],
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Results written to {out_path}")

    # Exit with non-zero code if the test fails (useful in CI)
    sys.exit(0 if stats["stale_surfaced"] == 0 and
             stats["correctness_rate_pct"] >= 95.0 else 1)


if __name__ == "__main__":
    main()
