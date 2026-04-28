"""ContextManager — multi-turn conversational memory for Nova.

Maintains:
  - Turn history (sliding window)
  - Current focus (last opened app/file/URL)
  - Reference resolution for pronouns and ordinals
  - Cross-session persistence of preferences and macros
"""
from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from nova.nlu.schemas import Entity, EntityType, IntentObject
from nova.core.logger import get_logger

logger = get_logger(__name__)

MAX_TURNS = 20
CONTEXT_PERSISTENCE_PATH = Path.home() / ".nova" / "session_context.json"


@dataclass
class Turn:
    """A single conversational turn."""
    role: str           # 'user' or 'nova'
    text: str
    intent_name: Optional[str] = None
    entities: List[Entity] = field(default_factory=list)
    timestamp: Optional[str] = None


@dataclass
class SessionContext:
    """Full session state."""
    turn_history: Deque[Turn] = field(default_factory=lambda: deque(maxlen=MAX_TURNS))
    current_focus: Optional[str] = None      # Last opened app / URL / file
    last_subject: Optional[Entity] = None   # Last concrete entity mentioned
    active_macro: Optional[str] = None
    open_windows_snapshot: List[str] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    named_macros: Dict[str, Any] = field(default_factory=dict)


class ContextManager:
    """Manages short-term session memory and coreference resolution."""

    _PRONOUN_PATTERN = re.compile(
        r"\b(it|that|this|the one|the last|the same|that one|those|them)\b",
        re.IGNORECASE
    )

    def __init__(self, config=None) -> None:
        self.config = config
        self._ctx = SessionContext()
        self._load_persistent_context()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        transcript: str,
        intent: Optional[IntentObject] = None
    ) -> None:
        """Record a user turn and update focus/last_subject."""
        import datetime
        turn = Turn(
            role="user",
            text=transcript,
            intent_name=intent.intent_name if intent else None,
            entities=list(intent.entities) if intent else [],
            timestamp=datetime.datetime.utcnow().isoformat(),
        )
        self._ctx.turn_history.append(turn)

        if intent:
            # Update last concrete entity of each type
            for entity in intent.entities:
                if entity.type in (
                    EntityType.APPLICATION_REF,
                    EntityType.URL_REF,
                    EntityType.FILE_PATH_REF,
                ):
                    self._ctx.current_focus = entity.value
                self._ctx.last_subject = entity

        logger.debug("Context updated. Focus: %s", self._ctx.current_focus)

    def record_execution(self, plan, result) -> None:
        """Record a Nova response turn after execution."""
        import datetime
        turn = Turn(
            role="nova",
            text=result.response_text or "Done.",
            timestamp=datetime.datetime.utcnow().isoformat(),
        )
        self._ctx.turn_history.append(turn)

    def resolve_reference(self, ref: str) -> Optional[Entity]:
        """Resolve a pronoun or vague reference to a concrete Entity.

        Strategy:
          1. Rule-based: scan back through turn history for last entity
             matching the type implied by the reference.
          2. Returns None if resolution fails (pipeline should ask user).
        """
        ref_lower = ref.lower()
        if ref_lower in {"it", "that", "this", "the one", "the last", "that one"}:
            return self._ctx.last_subject
        if ref_lower in {"the window", "that window"}:
            # Return the last application-type entity
            return self._last_entity_of_type(EntityType.APPLICATION_REF)
        if re.match(r"the (first|second|third|1st|2nd|3rd) (result|one|link)",
                    ref_lower):
            # Handled downstream by executor
            return None
        return None

    def get_llm_context_window(self, max_tokens: int = 1000) -> List[Dict]:
        """Serialize the last N turns as a list of message dicts for LLM context.

        Approximation: 1 token ~ 4 chars.
        """
        messages = []
        char_budget = max_tokens * 4
        used = 0
        for turn in reversed(self._ctx.turn_history):
            msg = {"role": turn.role, "content": turn.text}
            used += len(turn.text)
            if used > char_budget:
                break
            messages.insert(0, msg)
        return messages

    @property
    def current_focus(self) -> Optional[str]:
        return self._ctx.current_focus

    @property
    def last_subject(self) -> Optional[Entity]:
        return self._ctx.last_subject

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_persistent_context(self) -> None:
        """Write preferences and macros to disk (ephemeral state is NOT saved)."""
        data = {
            "user_preferences": self._ctx.user_preferences,
            "named_macros": self._ctx.named_macros,
        }
        CONTEXT_PERSISTENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONTEXT_PERSISTENCE_PATH.write_text(json.dumps(data, indent=2))
        logger.debug("Context persisted to %s", CONTEXT_PERSISTENCE_PATH)

    def _load_persistent_context(self) -> None:
        if CONTEXT_PERSISTENCE_PATH.exists():
            try:
                data = json.loads(CONTEXT_PERSISTENCE_PATH.read_text())
                self._ctx.user_preferences = data.get("user_preferences", {})
                self._ctx.named_macros = data.get("named_macros", {})
            except Exception as exc:
                logger.warning("Failed to load persisted context: %s", exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _last_entity_of_type(self, etype: EntityType) -> Optional[Entity]:
        for turn in reversed(self._ctx.turn_history):
            for entity in reversed(turn.entities):
                if entity.type == etype:
                    return entity
        return None
