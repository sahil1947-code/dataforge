"""
routing/intent.py
-----------------
Intent Router — maps a transcript to a routing decision.

Architecture notes (TRD §3.7):
  • Default to local resolution; escalate to LIVE_DATA only when local
    resolution is demonstrably insufficient.
  • Every routing decision is logged with its rationale for the
    "why did you do that?" transparency feature (App Flow §14).
  • Output: RoutingDecision(route, tool_name, args, reason, permission_level)

The router uses keyword/pattern matching as the primary signal, with an
optional LLM-based fallback for ambiguous intents.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from database import execute, encode_json

logger = logging.getLogger(__name__)


class RouteType(str, Enum):
    LOCAL_MEMORY  = "LOCAL_MEMORY"
    LOCAL_TOOL    = "LOCAL_TOOL"
    LIVE_DATA     = "LIVE_DATA"
    AGENT_ONLY    = "AGENT_ONLY"     # no tool — pure LLM reasoning
    CLARIFICATION = "CLARIFICATION"  # system needs more info


@dataclass
class RoutingDecision:
    route: RouteType
    tool_name: Optional[str]           # None for AGENT_ONLY / CLARIFICATION
    args: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    permission_level: int = 0
    requires_confirmation: bool = False
    generation: int = 0
    profile_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Pattern tables — ordered most-specific first
# ---------------------------------------------------------------------------

_LOCAL_TOOL_PATTERNS: list[tuple[re.Pattern, str, dict[str, Any]]] = [
    # Calculator
    (re.compile(
        r"(?:what(?:'s| is)\s+)?(\d[\d\s\+\-\*\/\(\)\.\^%]+\d|\d)\s*(?:=|equals?)?",
        re.I,
    ), "calculator", {}),
    (re.compile(r"\b(?:calculate|compute|math|multiply|divide|add|subtract|sqrt)\b", re.I),
     "calculator", {}),

    # Time / date
    (re.compile(r"\b(?:what(?:'s| is) (?:the )?(?:time|date|day)|what day|today|current time)\b", re.I),
     "time", {}),

    # Tasks
    (re.compile(r"\b(?:remind(?:er)?|reminder|set (?:a )?reminder|add (?:a )?task|schedule|my tasks?|what(?:'s| are) (?:my )?tasks?|complete task|cancel task)\b", re.I),
     "tasks", {}),

    # Memory read
    (re.compile(r"\b(?:what do you (?:know|remember) about me|my preferences?|my memories?|what have i told you)\b", re.I),
     "memory", {"action": "list"}),

    # Memory write
    (re.compile(r"\b(?:remember that|remember this|i prefer|call me|my name is|i always|i usually)\b", re.I),
     "memory", {"action": "add"}),

    # Forget
    (re.compile(r"\b(?:forget (?:that|this|everything)|clear my (?:history|memory|data)|delete my (?:data|history))\b", re.I),
     "memory", {"action": "delete"}),
]

_LIVE_DATA_PATTERNS: list[tuple[re.Pattern, str, dict[str, Any]]] = [
    (re.compile(r"\b(?:weather|temperature|forecast|rain|humidity|wind)\b", re.I),
     "weather", {}),
    (re.compile(r"\b(?:train|bus|metro|transit|transport|route|schedule|platform|arrival|departure)\b", re.I),
     "transport", {}),
]

_MEMORY_RECALL_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:last time|you mentioned|you said|i told you|remember when)\b", re.I), "memory_recall"),
]


class IntentRouter:
    """
    Routes a transcript to the correct tool or agent path.

    Every decision is persisted to the `events` table so the
    "why did you do that?" feature has a complete audit trail.
    """

    def __init__(self) -> None:
        self._decision_log: list[RoutingDecision] = []

    async def route(
        self,
        transcript: str,
        generation: int,
        profile_id: Optional[str] = None,
        session_context: Optional[dict[str, Any]] = None,
    ) -> RoutingDecision:
        """
        Determine the best route for `transcript`.
        Logs the decision and returns a RoutingDecision immediately.
        """
        text = transcript.strip()
        decision = self._classify(text, generation, profile_id)

        await self._log_decision(decision, text, profile_id)
        self._decision_log.append(decision)

        logger.info(
            "Intent: '%s' → %s (tool=%s, reason='%s')",
            text[:60],
            decision.route.value,
            decision.tool_name,
            decision.reason,
        )
        return decision

    def last_decision(self) -> Optional[RoutingDecision]:
        return self._decision_log[-1] if self._decision_log else None

    # ------------------------------------------------------------------
    # Classification logic
    # ------------------------------------------------------------------

    def _classify(
        self,
        text: str,
        generation: int,
        profile_id: Optional[str],
    ) -> RoutingDecision:
        # 1. Check local tool patterns
        for pattern, tool_name, base_args in _LOCAL_TOOL_PATTERNS:
            if pattern.search(text):
                args = {**base_args}
                args = self._extract_args(tool_name, text, args)
                from .permissions import PermissionLevel, requires_confirmation
                perm = self._tool_permission(tool_name, args)
                return RoutingDecision(
                    route=RouteType.LOCAL_TOOL,
                    tool_name=tool_name,
                    args=args,
                    reason=f"Pattern matched local tool '{tool_name}'.",
                    permission_level=perm,
                    requires_confirmation=requires_confirmation(perm),
                    generation=generation,
                    profile_id=profile_id,
                )

        # 2. Memory recall (context retrieval, not a tool call)
        for pattern, _ in _MEMORY_RECALL_PATTERNS:
            if pattern.search(text):
                return RoutingDecision(
                    route=RouteType.LOCAL_MEMORY,
                    tool_name=None,
                    args={},
                    reason="Memory recall detected — will inject context from DB.",
                    permission_level=0,
                    generation=generation,
                    profile_id=profile_id,
                )

        # 3. Live data patterns
        for pattern, tool_name, base_args in _LIVE_DATA_PATTERNS:
            if pattern.search(text):
                args = {**base_args}
                args = self._extract_args(tool_name, text, args)
                return RoutingDecision(
                    route=RouteType.LIVE_DATA,
                    tool_name=tool_name,
                    args=args,
                    reason=(
                        f"Live data required: '{tool_name}'. "
                        "Local resolution not possible for real-time information."
                    ),
                    permission_level=3,
                    requires_confirmation=False,
                    generation=generation,
                    profile_id=profile_id,
                )

        # 4. Default: pure agent reasoning
        return RoutingDecision(
            route=RouteType.AGENT_ONLY,
            tool_name=None,
            args={},
            reason="No specific tool pattern matched — routing to agent for reasoning.",
            permission_level=0,
            generation=generation,
            profile_id=profile_id,
        )

    # ------------------------------------------------------------------
    # Argument extraction helpers
    # ------------------------------------------------------------------

    def _extract_args(
        self, tool_name: str, text: str, base_args: dict
    ) -> dict[str, Any]:
        args = dict(base_args)
        if tool_name == "calculator":
            # Extract the expression — remove common preambles
            expr = re.sub(
                r"(?:what(?:'s| is)|calculate|compute|please|tell me|equals?)",
                "",
                text,
                flags=re.I,
            ).strip().rstrip("?.")
            args["expression"] = expr
        elif tool_name in ("weather", "transport"):
            # Try to extract a location / route from "in <place>" or "for <place>"
            loc_match = re.search(r"\b(?:in|for|at|to|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", text)
            if loc_match:
                key = "location" if tool_name == "weather" else "route"
                args[key] = loc_match.group(1)
        elif tool_name == "time":
            # Check for timezone hints
            tz_match = re.search(r"\b(?:in|for)\s+([A-Z][a-z]+(?:[/_][A-Za-z]+)*)\b", text)
            if tz_match:
                args["timezone"] = tz_match.group(1).replace("/", "/")
        elif tool_name == "tasks":
            if "action" not in args:
                if re.search(r"\b(?:add|create|set|remind|schedule)\b", text, re.I):
                    args["action"] = "create"
                elif re.search(r"\b(?:complete|done|finished|mark)\b", text, re.I):
                    args["action"] = "complete"
                elif re.search(r"\b(?:cancel|remove|delete)\b", text, re.I):
                    args["action"] = "cancel"
                else:
                    args["action"] = "list"
            # Extract title from "remind me to <title>" pattern
            title_match = re.search(
                r"\b(?:remind(?:er)? (?:me )?(?:to|about)|task(?::|:)?|add)\s+(.+?)(?:\s+(?:at|by|before|on)\b|$)",
                text, re.I,
            )
            if title_match and "title" not in args:
                args["title"] = title_match.group(1).strip().rstrip(".")
        elif tool_name == "memory":
            if args.get("action") == "add":
                # Extract content after memory trigger phrases
                content_match = re.search(
                    r"\b(?:remember that|remember this|i prefer|i always|i usually|call me|my name is)\s+(.+)",
                    text, re.I,
                )
                if content_match:
                    args["content"] = content_match.group(1).strip().rstrip(".")
        return args

    def _tool_permission(self, tool_name: str, args: dict) -> int:
        action = args.get("action", "")
        if tool_name in ("weather", "transport"):
            return 3
        if tool_name == "memory":
            if action in ("delete", "delete_category"):
                return 2
            if action == "add":
                return 1
            return 0
        if tool_name == "tasks":
            if action in ("cancel", "delete"):
                return 2
            if action == "create":
                return 1
            return 0
        return 0

    # ------------------------------------------------------------------
    # Audit logging
    # ------------------------------------------------------------------

    async def _log_decision(
        self,
        decision: RoutingDecision,
        transcript: str,
        profile_id: Optional[str],
    ) -> None:
        event_id = f"evt_{uuid.uuid4().hex[:12]}"
        metadata = encode_json({
            "transcript": transcript[:200],
            "route": decision.route.value,
            "tool": decision.tool_name,
            "reason": decision.reason,
            "generation": decision.generation,
            "permission_level": decision.permission_level,
        })
        try:
            await execute(
                """
                INSERT INTO events
                    (event_id, profile_id, event_type, timestamp, metadata)
                VALUES (?, ?, 'routing_decision', ?, ?)
                """,
                (
                    event_id,
                    profile_id,
                    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    metadata,
                ),
            )
        except Exception:  # noqa: BLE001
            pass  # Logging failure must never break routing
