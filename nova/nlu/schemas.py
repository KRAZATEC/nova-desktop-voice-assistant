"""Pydantic schemas for NLU and Planner data models.

Defines:
  - Entity: a single extracted entity from a user utterance
  - IntentObject: structured representation of user intent
  - ActionStep: a single executable step in a plan
  - ActionPlan: an ordered sequence of ActionSteps
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Entity types
# ---------------------------------------------------------------------------

class EntityType(str, Enum):
    APPLICATION_REF = "ApplicationRef"
    FILE_PATH_REF = "FilePathRef"
    URL_REF = "URLRef"
    QUERY_STRING = "QueryString"
    ACTION_VERB = "ActionVerb"
    TEMPORAL_REF = "TemporalRef"
    ORDINAL_REF = "OrdinalRef"
    PROPER_NOUN = "ProperNoun"
    UNKNOWN = "Unknown"


class Entity(BaseModel):
    """A single extracted entity from a user utterance."""

    role: str = Field(..., description="Semantic role, e.g. 'target_app', 'query'")
    value: str = Field(..., description="Normalised entity value")
    type: EntityType = Field(default=EntityType.UNKNOWN)
    span: Optional[tuple[int, int]] = Field(
        default=None,
        description="Character span (start, end) in raw_text"
    )
    raw_value: Optional[str] = Field(
        default=None,
        description="Original surface form before normalisation"
    )

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Intent object
# ---------------------------------------------------------------------------

class IntentObject(BaseModel):
    """Structured representation of a parsed user intent."""

    intent_name: str = Field(..., description="Canonical intent identifier, e.g. 'open_app'")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Classifier confidence in [0, 1]"
    )
    entities: List[Entity] = Field(default_factory=list)
    raw_text: str = Field(..., description="Original utterance as spoken")
    normalized_text: str = Field(..., description="Lowercase, punctuation-stripped utterance")
    requires_context: bool = Field(
        default=False,
        description="True when intent references prior context (pronouns, ordinals)"
    )
    nlu_strategy: str = Field(
        default="rule",
        description="Which NLU path produced this: 'rule', 'ml', 'llm'"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("normalized_text", mode="before")
    @classmethod
    def normalise(cls, v: str) -> str:
        import re
        return re.sub(r"[^\w\s]", "", v.lower()).strip()

    def get_entity(self, role: str) -> Optional[Entity]:
        """Return the first entity matching *role*, or None."""
        return next((e for e in self.entities if e.role == role), None)

    def get_entities_by_type(self, etype: EntityType) -> List[Entity]:
        """Return all entities of a given type."""
        return [e for e in self.entities if e.type == etype]


# ---------------------------------------------------------------------------
# Action plan
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    OPEN_APP = "OPEN_APP"
    CLOSE_APP = "CLOSE_APP"
    NAVIGATE_URL = "NAVIGATE_URL"
    TYPE_TEXT = "TYPE_TEXT"
    CLICK_ELEMENT = "CLICK_ELEMENT"
    PRESS_KEY = "PRESS_KEY"
    SEARCH_WEB = "SEARCH_WEB"
    READ_CLIPBOARD = "READ_CLIPBOARD"
    WRITE_CLIPBOARD = "WRITE_CLIPBOARD"
    RUN_MACRO = "RUN_MACRO"
    SPEAK_RESPONSE = "SPEAK_RESPONSE"
    WAIT_FOR_CONDITION = "WAIT_FOR_CONDITION"
    REQUEST_CONFIRMATION = "REQUEST_CONFIRMATION"
    SWITCH_WINDOW = "SWITCH_WINDOW"
    SET_VOLUME = "SET_VOLUME"
    SEND_EMAIL = "SEND_EMAIL"
    SET_REMINDER = "SET_REMINDER"
    READ_WEATHER = "READ_WEATHER"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ActionStep(BaseModel):
    """A single parameterised execution step."""

    action_id: str = Field(..., description="Unique identifier for this step")
    action_type: ActionType
    params: Dict[str, Any] = Field(default_factory=dict)
    preconditions: List[str] = Field(
        default_factory=list,
        description="List of condition strings that must be true before execution"
    )
    postconditions: List[str] = Field(
        default_factory=list,
        description="Expected state after successful execution"
    )
    on_failure: str = Field(
        default="abort",
        description="Strategy on failure: 'abort', 'skip', 'retry', 'ask_user'"
    )
    requires_confirmation: bool = Field(default=False)
    risk_level: RiskLevel = Field(default=RiskLevel.LOW)
    timeout_seconds: float = Field(default=10.0, gt=0)

    @field_validator("on_failure")
    @classmethod
    def validate_on_failure(cls, v: str) -> str:
        allowed = {"abort", "skip", "retry", "ask_user"}
        if v not in allowed:
            raise ValueError(f"on_failure must be one of {allowed}")
        return v


class ActionPlan(BaseModel):
    """An ordered, validated sequence of ActionSteps for a user command."""

    plan_id: str = Field(..., description="UUID for this plan instance")
    intent_name: str
    steps: List[ActionStep] = Field(..., min_length=1)
    context_snapshot: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None
    planner_strategy: str = Field(default="template")
    estimated_duration_seconds: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def check_unique_step_ids(self) -> ActionPlan:
        ids = [s.action_id for s in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("All action_id values in an ActionPlan must be unique.")
        return self

    def has_critical_steps(self) -> bool:
        return any(s.risk_level == RiskLevel.CRITICAL for s in self.steps)

    def steps_requiring_confirmation(self) -> List[ActionStep]:
        return [s for s in self.steps if s.requires_confirmation]
