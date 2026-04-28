"""NLU Pipeline with hybrid routing: rule -> ML -> LLM fallback.

Routing logic:
  1. Rule-based: regex patterns for high-frequency, unambiguous commands (< 2ms)
  2. ML classifier: spaCy text-cat for medium-confidence cases (< 50ms)
  3. LLM fallback: structured JSON output for novel/complex utterances (< 800ms)
"""
from __future__ import annotations

import re
import asyncio
import logging
from typing import Dict, List, Optional, Tuple

from nova.nlu.schemas import Entity, EntityType, IntentObject
from nova.core.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Rule-based intent patterns
# ---------------------------------------------------------------------------

_RULE_PATTERNS: List[Tuple[re.Pattern, str, callable]] = [
    # Pattern, intent_name, entity_extractor
    (
        re.compile(
            r"(?:open|launch|start)\s+(?P<app>[\w\s]+?)(?:\s+and\s+|$)",
            re.IGNORECASE
        ),
        "open_app",
        lambda m: [Entity(role="target_app", value=m.group("app").strip(),
                          type=EntityType.APPLICATION_REF)]
    ),
    (
        re.compile(r"close\s+(?P<target>.+)", re.IGNORECASE),
        "close_app",
        lambda m: [Entity(role="target", value=m.group("target").strip(),
                          type=EntityType.APPLICATION_REF)]
    ),
    (
        re.compile(
            r"(?:search|look up|google)\s+(?P<query>.+?)(?:\s+and|$)",
            re.IGNORECASE
        ),
        "search_web",
        lambda m: [Entity(role="query", value=m.group("query").strip(),
                          type=EntityType.QUERY_STRING)]
    ),
    (
        re.compile(r"(?:make|set|turn)\s+(?:the\s+)?volume\s+(?P<dir>louder|higher|up|quieter|lower|down)",
                   re.IGNORECASE),
        "set_volume",
        lambda m: [Entity(role="direction", value=m.group("dir").strip(),
                          type=EntityType.ACTION_VERB)]
    ),
    (
        re.compile(r"remind\s+me\s+(?:in\s+)?(?P<time>[\w\s]+?)\s+to\s+(?P<task>.+)",
                   re.IGNORECASE),
        "set_reminder",
        lambda m: [
            Entity(role="delay", value=m.group("time").strip(), type=EntityType.TEMPORAL_REF),
            Entity(role="task", value=m.group("task").strip(), type=EntityType.QUERY_STRING),
        ]
    ),
    (
        re.compile(r"(?:what(?:'s|\s+is)\s+)?(?:the\s+)?weather(?:\s+today)?",
                   re.IGNORECASE),
        "get_weather",
        lambda m: []
    ),
    (
        re.compile(r"(?:switch|go)\s+(?:to\s+)?(?:my\s+)?(?:last|previous)\s+(?:used\s+)?window",
                   re.IGNORECASE),
        "switch_window",
        lambda m: [Entity(role="target", value="last", type=EntityType.ORDINAL_REF)]
    ),
    (
        re.compile(r"send\s+(?:an\s+)?email\s+to\s+(?P<recipient>[\w\s]+?)\s+about\s+(?P<subject>.+)",
                   re.IGNORECASE),
        "send_email",
        lambda m: [
            Entity(role="recipient", value=m.group("recipient").strip(),
                   type=EntityType.PROPER_NOUN),
            Entity(role="subject", value=m.group("subject").strip(),
                   type=EntityType.QUERY_STRING),
        ]
    ),
]

# Pronouns / references that require context resolution
_CONTEXT_PRONOUNS = re.compile(
    r"\b(it|that|this|the one|the last|what we had|the same)\b",
    re.IGNORECASE
)


class NLUPipeline:
    """Hybrid NLU pipeline: rule -> ML -> LLM."""

    CONFIDENCE_ML_THRESHOLD = 0.72
    CONFIDENCE_LLM_THRESHOLD = 0.55

    def __init__(self, config) -> None:
        self.config = config
        self._ml_classifier = None   # Loaded lazily
        self._llm_client = None      # Injected or loaded lazily

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def parse(self, text: str) -> IntentObject:
        """Parse a raw transcript string and return a structured IntentObject.

        Routing decision:
          - Rule-based: deterministic regex match -> instant, high confidence.
          - ML path: spaCy text-cat when rule confidence is None.
          - LLM path: fallback for ambiguous/novel utterances.
        """
        normalized = self._normalize(text)
        requires_context = bool(_CONTEXT_PRONOUNS.search(text))

        # --- 1. Rule-based ---
        rule_result = self._rule_match(text, normalized)
        if rule_result is not None:
            logger.debug("NLU: rule-based match -> %s", rule_result.intent_name)
            return rule_result

        # --- 2. ML classifier ---
        if self._ml_classifier is not None:
            ml_result = await self._ml_classify(text, normalized, requires_context)
            if ml_result is not None and ml_result.confidence >= self.CONFIDENCE_ML_THRESHOLD:
                logger.debug("NLU: ML match -> %s (%.2f)", ml_result.intent_name, ml_result.confidence)
                return ml_result
        else:
            logger.debug("NLU: ML classifier not loaded; skipping to LLM.")
            ml_result = None

        # --- 3. LLM fallback ---
        llm_result = await self._llm_classify(text, normalized, requires_context)
        if llm_result is not None:
            logger.debug("NLU: LLM match -> %s (%.2f)", llm_result.intent_name, llm_result.confidence)
            return llm_result

        # --- Final fallback: unknown intent ---
        logger.warning("NLU: could not classify '%s'", text)
        return IntentObject(
            intent_name="unknown",
            confidence=0.0,
            entities=[],
            raw_text=text,
            normalized_text=normalized,
            requires_context=requires_context,
            nlu_strategy="none",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"[^\w\s]", "", text.lower()).strip()

    def _rule_match(self, raw: str, normalized: str) -> Optional[IntentObject]:
        """Attempt a deterministic regex match over *_RULE_PATTERNS*."""
        for pattern, intent_name, entity_extractor in _RULE_PATTERNS:
            m = pattern.search(raw)
            if m:
                entities = entity_extractor(m)
                requires_ctx = bool(_CONTEXT_PRONOUNS.search(raw))
                return IntentObject(
                    intent_name=intent_name,
                    confidence=1.0,
                    entities=entities,
                    raw_text=raw,
                    normalized_text=normalized,
                    requires_context=requires_ctx,
                    nlu_strategy="rule",
                )
        return None

    async def _ml_classify(self, raw: str, normalized: str,
                            requires_context: bool) -> Optional[IntentObject]:
        """Classify via spaCy text-cat pipeline."""
        try:
            loop = asyncio.get_event_loop()
            doc = await loop.run_in_executor(None, self._ml_classifier, raw)
            scores: Dict[str, float] = doc.cats
            if not scores:
                return None
            best_intent = max(scores, key=scores.get)
            confidence = scores[best_intent]
            return IntentObject(
                intent_name=best_intent,
                confidence=confidence,
                entities=[],   # Entity extraction handled by NER in separate pass
                raw_text=raw,
                normalized_text=normalized,
                requires_context=requires_context,
                nlu_strategy="ml",
            )
        except Exception as exc:
            logger.error("ML classifier error: %s", exc)
            return None

    async def _llm_classify(self, raw: str, normalized: str,
                             requires_context: bool) -> Optional[IntentObject]:
        """Classify via LLM structured output (JSON schema)."""
        if self._llm_client is None:
            return None
        prompt = self._build_llm_prompt(raw)
        try:
            response = await self._llm_client.structured_completion(
                prompt=prompt,
                schema=IntentObject.model_json_schema(),
                max_retries=2,
            )
            if response:
                obj = IntentObject.model_validate(response)
                obj = obj.model_copy(update={"nlu_strategy": "llm"})
                return obj
        except Exception as exc:
            logger.error("LLM NLU error: %s", exc)
        return None

    @staticmethod
    def _build_llm_prompt(utterance: str) -> str:
        examples = [
            ("Open Chrome and search for AI tools",
             '{"intent_name": "open_and_search", "confidence": 0.97}'),
            ("Close it",
             '{"intent_name": "close_app", "requires_context": true, "confidence": 0.95}'),
            ("Remind me in 20 minutes to take a break",
             '{"intent_name": "set_reminder", "confidence": 0.99}'),
        ]
        example_block = "\n".join(
            f"  User: {u}\n  Intent JSON: {j}" for u, j in examples
        )
        return (
            "You are an NLU classifier for a desktop voice assistant called Nova.\n"
            "Given a user utterance, return a JSON object matching the IntentObject schema.\n"
            "\nExamples:\n" + example_block +
            f"\n\nUser: {utterance}\nIntent JSON:"
        )
