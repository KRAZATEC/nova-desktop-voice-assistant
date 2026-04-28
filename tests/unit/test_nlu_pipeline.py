"""Unit tests for the NLU pipeline.

Covers: rule-based matching, entity extraction, context flag detection,
unknown intent handling, and edge cases.
"""
from __future__ import annotations

import pytest

from nova.nlu.pipeline import NLUPipeline
from nova.nlu.schemas import EntityType, IntentObject


@pytest.fixture()
def pipeline() -> NLUPipeline:
    """Return a pipeline with no ML or LLM backend (rule-only mode)."""
    return NLUPipeline(config=None)


# ---------------------------------------------------------------------------
# Happy-path rule-based tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_open_app_intent(pipeline: NLUPipeline) -> None:
    """'Open Chrome' should resolve to open_app with ApplicationRef entity."""
    result = await pipeline.parse("Open Chrome")

    assert isinstance(result, IntentObject)
    assert result.intent_name == "open_app"
    assert result.confidence == 1.0
    assert result.nlu_strategy == "rule"
    app_entity = result.get_entity("target_app")
    assert app_entity is not None
    assert "chrome" in app_entity.value.lower()
    assert app_entity.type == EntityType.APPLICATION_REF


@pytest.mark.asyncio
async def test_search_intent(pipeline: NLUPipeline) -> None:
    """'Search Python tutorials' should resolve to search_web with query entity."""
    result = await pipeline.parse("Search Python tutorials")

    assert result.intent_name == "search_web"
    query = result.get_entity("query")
    assert query is not None
    assert "python tutorials" in query.value.lower()
    assert query.type == EntityType.QUERY_STRING


@pytest.mark.asyncio
async def test_set_reminder_intent(pipeline: NLUPipeline) -> None:
    """'Remind me in 20 minutes to take a break' should extract delay + task."""
    result = await pipeline.parse("Remind me in 20 minutes to take a break")

    assert result.intent_name == "set_reminder"
    delay = result.get_entity("delay")
    task = result.get_entity("task")
    assert delay is not None
    assert task is not None
    assert "20 minutes" in delay.value
    assert "take a break" in task.value


@pytest.mark.asyncio
async def test_volume_louder_intent(pipeline: NLUPipeline) -> None:
    """'Make the volume louder' should resolve to set_volume."""
    result = await pipeline.parse("Make the volume louder")

    assert result.intent_name == "set_volume"
    direction = result.get_entity("direction")
    assert direction is not None
    assert direction.value.lower() == "louder"


@pytest.mark.asyncio
async def test_close_intent_with_pronoun(pipeline: NLUPipeline) -> None:
    """'Close it' should set requires_context=True."""
    result = await pipeline.parse("Close it")

    assert result.intent_name == "close_app"
    # 'it' is a pronoun that requires context resolution
    assert result.requires_context is True


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_string(pipeline: NLUPipeline) -> None:
    """Empty string should return 'unknown' intent with zero confidence."""
    result = await pipeline.parse("")

    assert result.intent_name == "unknown"
    assert result.confidence == 0.0


@pytest.mark.asyncio
async def test_weather_query(pipeline: NLUPipeline) -> None:
    """'What's the weather like today?' should resolve to get_weather."""
    result = await pipeline.parse("What's the weather like today?")

    assert result.intent_name == "get_weather"
    assert result.confidence == 1.0


@pytest.mark.asyncio
async def test_send_email_intent(pipeline: NLUPipeline) -> None:
    """'Send an email to Rohan about the meeting' should extract recipient + subject."""
    result = await pipeline.parse("Send an email to Rohan about the meeting")

    assert result.intent_name == "send_email"
    recipient = result.get_entity("recipient")
    subject = result.get_entity("subject")
    assert recipient is not None
    assert "rohan" in recipient.value.lower()
    assert subject is not None
    assert "meeting" in subject.value.lower()


@pytest.mark.asyncio
async def test_switch_window_intent(pipeline: NLUPipeline) -> None:
    """'Switch to my last used window' should resolve to switch_window."""
    result = await pipeline.parse("Switch to my last used window")

    assert result.intent_name == "switch_window"
    ordinal = result.get_entity("target")
    assert ordinal is not None
    assert ordinal.type == EntityType.ORDINAL_REF


@pytest.mark.asyncio
async def test_normalized_text_strips_punctuation(pipeline: NLUPipeline) -> None:
    """normalized_text should be lowercase and have no punctuation."""
    result = await pipeline.parse("Open Chrome!")

    assert result.normalized_text == result.normalized_text.lower()
    assert "!" not in result.normalized_text
