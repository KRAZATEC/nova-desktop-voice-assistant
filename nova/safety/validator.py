"""SafetyValidator — mandatory gate between TaskPlanner and ExecutionEngine.

Risk classification:
  LOW    — open app, search, read clipboard
  MEDIUM — write file, submit form, send form data
  HIGH   — delete file, send email, run shell command
  CRITICAL — format drive, modify system files, exfiltrate data

For MEDIUM and above, Nova verbally states what it will do and
requires explicit user confirmation before proceeding.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from nova.nlu.schemas import ActionPlan, ActionStep, RiskLevel
from nova.core.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Forbidden command patterns (semantic + regex)
# ---------------------------------------------------------------------------

_FORBIDDEN_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # System paths
    (re.compile(r"/(etc|sys|proc|boot)/"), "Access to protected system paths is not allowed."),
    (re.compile(r"rm\s+-[rf]+"), "Recursive delete commands are blocked."),
    (re.compile(r"sudo|su\b"), "Privilege escalation commands are not permitted."),
    # Network exfiltration
    (re.compile(r"curl.*\|.*bash"), "Piping curl output to bash is blocked."),
    (re.compile(r"wget.*http"), "External wget downloads must be user-authorised."),
    # Destructive disk
    (re.compile(r"mkfs\.|dd\s+if="), "Disk format operations are critically dangerous and blocked."),
    # Payment APIs
    (re.compile(r"stripe|paypal|razorpay", re.IGNORECASE), "Payment API commands require manual execution."),
]

# Acceptance tokens for confirmation dialog
_CONFIRM_YES = {"yes", "confirm", "do it", "proceed", "sure", "ok", "yep", "yeah"}
_CONFIRM_NO = {"no", "cancel", "stop", "abort", "nope", "negative", "don't"}


class SafetyValidator:
    """Validate an ActionPlan before execution.

    Workflow:
      1. Check all step params against forbidden patterns.
      2. Classify overall plan risk level.
      3. For MEDIUM+ risk, request verbal confirmation from user.
      4. Return (approved: bool, reason: str).
    """

    CONFIRMATION_TIMEOUT = 10.0  # seconds to wait for user response

    def __init__(self, config=None) -> None:
        self.config = config

    async def validate(
        self,
        plan: ActionPlan,
        tts=None,     # TTSBase instance, for speaking confirmation prompts
        stt=None,     # STTBase instance, for listening to responses
    ) -> Tuple[bool, str]:
        """Validate *plan*.

        Returns:
            (True, "") if plan is approved.
            (False, reason) if plan is blocked or user declined.
        """
        # 1. Check for forbidden patterns
        block_reason = self._check_forbidden(plan)
        if block_reason:
            logger.warning("Plan blocked: %s", block_reason)
            return False, block_reason

        # 2. Determine aggregate risk
        risk = self._aggregate_risk(plan)
        logger.debug("Plan risk level: %s", risk)

        # 3. LOW risk — auto-approve
        if risk == RiskLevel.LOW:
            return True, ""

        # 4. CRITICAL — always reject
        if risk == RiskLevel.CRITICAL:
            msg = "This action is classified as CRITICAL risk and is blocked."
            return False, msg

        # 5. MEDIUM / HIGH — request confirmation
        if tts is not None:
            confirmation_text = self._build_confirmation_speech(plan, risk)
            await tts.speak(confirmation_text)

        if stt is not None:
            # Wait for user response
            import asyncio
            try:
                transcript = await asyncio.wait_for(
                    stt.transcribe_short_utterance(),
                    timeout=self.CONFIRMATION_TIMEOUT
                )
                response_lower = (transcript.text or "").lower().strip()
                if any(token in response_lower for token in _CONFIRM_YES):
                    logger.info("User confirmed MEDIUM/HIGH action.")
                    return True, ""
                else:
                    logger.info("User declined MEDIUM/HIGH action.")
                    return False, "User declined confirmation."
            except asyncio.TimeoutError:
                logger.warning("Confirmation timed out; action blocked.")
                return False, "No confirmation received within timeout."
        else:
            # Non-interactive mode: auto-approve MEDIUM, block HIGH
            if risk == RiskLevel.MEDIUM:
                return True, ""
            return False, "HIGH risk action requires interactive confirmation."

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_forbidden(plan: ActionPlan) -> Optional[str]:
        """Return a block reason string if any step param violates a forbidden pattern."""
        for step in plan.steps:
            for param_value in step.params.values():
                s = str(param_value)
                for pattern, reason in _FORBIDDEN_PATTERNS:
                    if pattern.search(s):
                        return f"Step {step.action_id}: {reason}"
        return None

    @staticmethod
    def _aggregate_risk(plan: ActionPlan) -> RiskLevel:
        """Return the highest risk level across all steps."""
        order = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        max_risk = RiskLevel.LOW
        for step in plan.steps:
            if order.index(step.risk_level) > order.index(max_risk):
                max_risk = step.risk_level
        return max_risk

    @staticmethod
    def _build_confirmation_speech(plan: ActionPlan, risk: RiskLevel) -> str:
        step_summaries = "; ".join(
            f"{s.action_type.value} {list(s.params.values())[:1]}"
            for s in plan.steps
        )
        return (
            f"I'm about to {step_summaries}. "
            f"This is a {risk.value.lower()}-risk action. "
            "Say 'confirm' to proceed or 'cancel' to stop."
        )
