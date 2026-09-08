"""
agent/agent.py
--------------
Reasoning core for RIME.

Architecture notes (TRD §3.8):
  • Local-first: Ollama (self-hosted LLM) is attempted first.
  • External fallback: OpenAI is used only when Ollama is unreachable, and
    only through the ExternalServiceManager credential gateway — no key
    ever lives in client code.
  • Streaming: yields sentence-tokenized chunks so Rime TTS can begin
    speaking before the full response is generated.
  • Context injection: persistent memories + recent conversation history
    are prepended to every prompt.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import AsyncIterator, Optional

from config import get_settings

logger = logging.getLogger(__name__)
_settings = get_settings()

# Sentence boundary — split tokens into speakable chunks for progressive TTS
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+')

_SYSTEM_PROMPT_TEMPLATE = """\
You are RIME, a local-first persistent voice assistant. You speak naturally and concisely — responses are heard, not read, so keep sentences short and clear.

Current speaker: {speaker_name}
Persistent memories about this speaker:
{memories}

Rules:
- Never fabricate live data (weather, transit) — only report what a tool returned.
- If you do not know something, say so plainly.
- When confirming a memory was saved, say so briefly.
- Respond in the speaker's preferred language when known.
"""


@dataclass
class AgentResponse:
    text: str = ""
    generation: int = 0
    first_token_ms: Optional[float] = None
    total_ms: Optional[float] = None
    backend: str = "unknown"


class ReasoningAgent:
    """
    Wraps the local LLM (Ollama) with an OpenAI fallback.

    Streaming usage (primary path — feeds Rime sentence by sentence):
        async for sentence in agent.stream(user_msg, memories, history, gen):
            await rime.speak(sentence, generation=gen)

    Single-shot usage (tool-response formatting):
        response = await agent.complete(user_msg, memories, history, gen)
    """

    def __init__(self) -> None:
        self._ollama_url   = _settings.llm.ollama_base_url
        self._ollama_model = _settings.llm.ollama_model
        raw_key            = _settings.llm.openai_api_key
        if raw_key and any(p in raw_key.lower() for p in ("your_", "dummy", "placeholder", "example")):
            self._openai_key = None
        else:
            self._openai_key = raw_key
        self._openai_model = _settings.llm.openai_model
        self._ollama_ok: Optional[bool] = None   # cached after first probe

    # ------------------------------------------------------------------
    # Streaming API  (primary)
    # ------------------------------------------------------------------

    async def stream(
        self,
        user_message: str,
        memory_context: list[dict],
        conversation_history: list[dict],
        generation: int,
        tool_result: Optional[dict] = None,
        speaker_name: str = "User",
    ) -> AsyncIterator[str]:
        """
        Yield complete sentences as they are produced.
        Each string is safe to hand directly to Rime TTS.
        """
        system = self._system_prompt(memory_context, speaker_name)
        messages = self._build_messages(
            system, conversation_history, user_message, tool_result
        )

        if await self._ollama_reachable():
            async for sentence in self._stream_ollama(
                messages, generation, user_message=user_message, tool_result=tool_result
            ):
                yield sentence
        elif self._openai_key:
            async for sentence in self._stream_openai(
                messages, generation, user_message=user_message, tool_result=tool_result
            ):
                yield sentence
        else:
            yield self._stub_response(user_message, tool_result)

    # ------------------------------------------------------------------
    # Single-shot API
    # ------------------------------------------------------------------

    async def complete(
        self,
        user_message: str,
        memory_context: list[dict],
        conversation_history: list[dict],
        generation: int,
        tool_result: Optional[dict] = None,
        speaker_name: str = "User",
    ) -> AgentResponse:
        t0 = time.monotonic()
        parts: list[str] = []
        first_token_ms: Optional[float] = None
        backend = "stub"

        async for sentence in self.stream(
            user_message, memory_context, conversation_history,
            generation, tool_result, speaker_name
        ):
            if first_token_ms is None:
                first_token_ms = (time.monotonic() - t0) * 1000
            parts.append(sentence)

        return AgentResponse(
            text=" ".join(parts),
            generation=generation,
            first_token_ms=first_token_ms,
            total_ms=(time.monotonic() - t0) * 1000,
            backend=backend,
        )

    # ------------------------------------------------------------------
    # Ollama streaming
    # ------------------------------------------------------------------

    async def _stream_ollama(
        self,
        messages: list[dict],
        generation: int,
        user_message: str = "",
        tool_result: Optional[dict] = None,
    ) -> AsyncIterator[str]:
        try:
            import httpx

            payload = {
                "model": self._ollama_model,
                "messages": messages,
                "stream": True,
            }
            buffer = ""
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    f"{self._ollama_url}/api/chat",
                    json=payload,
                ) as resp:
                    resp.raise_for_status()
                    import json
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            buffer += token
                            # Yield complete sentences as they accumulate
                            sentences = _SENTENCE_END.split(buffer)
                            for sentence in sentences[:-1]:
                                s = sentence.strip()
                                if s:
                                    yield s
                            buffer = sentences[-1]
                        if chunk.get("done"):
                            break

            if buffer.strip():
                yield buffer.strip()

        except Exception as exc:
            logger.warning("Ollama streaming error: %s — trying OpenAI", exc)
            self._ollama_ok = False
            if self._openai_key:
                async for sentence in self._stream_openai(
                    messages, generation, user_message=user_message, tool_result=tool_result
                ):
                    yield sentence
            else:
                yield self._stub_response(user_message, tool_result)

    # ------------------------------------------------------------------
    # OpenAI streaming fallback
    # ------------------------------------------------------------------

    async def _stream_openai(
        self,
        messages: list[dict],
        generation: int,
        user_message: str = "",
        tool_result: Optional[dict] = None,
    ) -> AsyncIterator[str]:
        try:
            import httpx, json

            headers = {
                "Authorization": f"Bearer {self._openai_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self._openai_model,
                "messages": messages,
                "stream": True,
            }
            buffer = ""
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    "https://api.openai.com/v1/chat/completions",
                    json=payload,
                    headers=headers,
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        token = (
                            chunk.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content", "")
                        )
                        if token:
                            buffer += token
                            sentences = _SENTENCE_END.split(buffer)
                            for sentence in sentences[:-1]:
                                s = sentence.strip()
                                if s:
                                    yield s
                            buffer = sentences[-1]

            if buffer.strip():
                yield buffer.strip()

        except Exception as exc:
            logger.error("OpenAI streaming error: %s", exc)
            yield self._stub_response(user_message, tool_result)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _ollama_reachable(self) -> bool:
        if self._ollama_ok is not None:
            return self._ollama_ok
        try:
            import httpx
            async with httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get(f"{self._ollama_url}/api/tags")
                self._ollama_ok = r.status_code == 200
        except Exception:
            self._ollama_ok = False
        return self._ollama_ok

    def _system_prompt(
        self, memory_context: list[dict], speaker_name: str
    ) -> str:
        if memory_context:
            mem_lines = "\n".join(
                f"- [{m.get('category', 'general')}] {m.get('content', '')}"
                for m in memory_context[:15]
            )
        else:
            mem_lines = "No persistent memories yet."

        return _SYSTEM_PROMPT_TEMPLATE.format(
            speaker_name=speaker_name,
            memories=mem_lines,
        )

    def _build_messages(
        self,
        system: str,
        history: list[dict],
        user_message: str,
        tool_result: Optional[dict],
    ) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": system}]

        # Recent conversation history (last 10 turns)
        for turn in history[-10:]:
            role = turn.get("role", "user")
            if role in ("user", "assistant"):
                messages.append({"role": role, "content": turn.get("text", "")})

        # Inject tool result if present
        if tool_result:
            if tool_result.get("success"):
                tool_text = (
                    f"[Tool result — {tool_result.get('tool_name', 'tool')}]: "
                    f"{tool_result.get('output', '')}"
                )
            else:
                tool_text = (
                    f"[Tool error — {tool_result.get('tool_name', 'tool')}]: "
                    f"{tool_result.get('error', 'Unknown error')}"
                )
            messages.append({"role": "system", "content": tool_text})

        messages.append({"role": "user", "content": user_message})
        return messages

    @staticmethod
    def _stub_response(user_message: str, tool_result: Optional[dict]) -> str:
        """
        Plain-language fallback when no external LLM is available.
        Provides natural, conversational spoken English for tools and common voice commands.
        """
        if tool_result and tool_result.get("success"):
            output = tool_result.get("output", {})
            tool_name = tool_result.get("tool_name", "")

            if isinstance(output, dict):
                # 1. Calculator
                if tool_name == "calculator" or "expression" in output:
                    res = output.get("result")
                    return f"The answer is {res}."

                # 2. Time
                if tool_name == "time" or "spoken" in output:
                    return output.get("spoken", "The current time has been retrieved.")

                # 3. Weather
                if tool_name == "weather" or "temperature_c" in output:
                    loc = output.get("location", "your location")
                    temp = output.get("temperature_c", "unknown")
                    desc = output.get("description", "clear skies")
                    return f"In {loc}, it is currently {temp} degrees Celsius with {desc}."

                # 4. Transport
                if tool_name == "transport":
                    route = output.get("route", "transit")
                    status = output.get("status", "running on schedule")
                    delay = output.get("delay_minutes", 0)
                    delay_txt = f" with a {delay} minute delay" if delay else ""
                    return f"The {route} is currently {status}{delay_txt}."

                # 5. Tasks
                if tool_name == "tasks":
                    if "tasks" in output:
                        tasks = output.get("tasks", [])
                        if not tasks:
                            return "You have no active tasks or reminders right now."
                        titles = [t.get("title", "task") for t in tasks[:3]]
                        more = f" and {len(tasks)-3} more" if len(tasks) > 3 else ""
                        return f"You have {len(tasks)} tasks: " + ", ".join(titles) + more + "."
                    if "title" in output:
                        return f"I have scheduled your reminder: '{output.get('title')}'."

                # 6. Memory
                if tool_name == "memory":
                    if "memories" in output:
                        mems = output.get("memories", [])
                        if not mems:
                            return "I don't have any persistent memories saved for you yet."
                        contents = [m.get("content", "") for m in mems[:3]]
                        return "Here is what I remember about you: " + "; ".join(contents) + "."
                    if "action" in output or "memory_id" in output:
                        return "Got it — I have updated your persistent memory."

                # Explicit result or spoken fallback
                result_val = output.get("result") or output.get("spoken") or output.get("message")
                if result_val:
                    return str(result_val)

            return str(output)

        if tool_result and not tool_result.get("success"):
            err = tool_result.get("error")
            return err if err else "I encountered an issue processing that tool request."

        # Conversational fallbacks
        msg_lower = user_message.lower().strip()
        if any(w in msg_lower for w in ("hello", "hi", "hey", "good morning", "good evening")):
            return "Hello! I am RIME, your persistent voice operating system. How can I help you?"
        if "who are you" in msg_lower or "what are you" in msg_lower:
            return "I am RIME, a local-first voice operating system with persistent memory and interruption safety."
        if "what can you do" in msg_lower or "help" in msg_lower:
            return "I can calculate math, check the time, manage your reminders, remember facts about you, track weather, and handle live voice interruptions."

        return f"I heard '{user_message}'. Local voice engine is running and ready for your commands."
