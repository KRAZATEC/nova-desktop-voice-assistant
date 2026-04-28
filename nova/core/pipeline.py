"""Nova main command pipeline.

Orchestrates the full flow:
  WakeWord -> STT -> NLU -> ContextManager -> TaskPlanner
  -> SafetyLayer -> ExecutionEngine -> TTS
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from nova.audio.engine import AudioEngine
from nova.audio.wake_word import WakeWordDetector
from nova.stt.base import STTBase
from nova.nlu.pipeline import NLUPipeline
from nova.context.manager import ContextManager
from nova.planner.task_planner import TaskPlanner
from nova.safety.validator import SafetyValidator
from nova.executor.engine import ExecutionEngine
from nova.tts.base import TTSBase
from nova.core.config import NovaConfig
from nova.core.logger import get_logger

logger = get_logger(__name__)


class NovaPipeline:
    """Top-level orchestrator for the Nova voice assistant pipeline."""

    def __init__(self, config: NovaConfig) -> None:
        self.config = config
        self.audio_engine = AudioEngine(config.audio)
        self.wake_word = WakeWordDetector(config.wake_word)
        self.stt: STTBase = config.stt_adapter(config.stt)
        self.nlu = NLUPipeline(config.nlu)
        self.context = ContextManager(config.context)
        self.planner = TaskPlanner(config.planner)
        self.safety = SafetyValidator(config.safety)
        self.executor = ExecutionEngine(config.executor)
        self.tts: TTSBase = config.tts_adapter(config.tts)
        self._running = False
        self._stop_event = asyncio.Event()

    async def run(self) -> None:
        """Start the Nova pipeline main loop."""
        self._running = True
        logger.info("Nova pipeline started. Listening for wake word...")
        await self.audio_engine.start()
        try:
            while self._running:
                await self._listen_and_process()
        finally:
            await self.audio_engine.stop()
            logger.info("Nova pipeline stopped.")

    async def stop(self) -> None:
        """Gracefully stop the pipeline."""
        self._running = False
        self._stop_event.set()

    async def _listen_and_process(self) -> None:
        """Wait for wake word, then process one full command."""
        # Step 1: Wait for wake word in background
        triggered = await asyncio.wait_for(
            self.wake_word.wait_for_activation(),
            timeout=None
        )
        if not triggered:
            return

        logger.debug("Wake word detected. Starting STT capture...")
        await self.tts.speak("Listening...")

        # Step 2: STT - capture and transcribe utterance
        try:
            transcript = await asyncio.wait_for(
                self.stt.transcribe_utterance(
                    audio_source=self.audio_engine
                ),
                timeout=self.config.stt.timeout_seconds
            )
        except asyncio.TimeoutError:
            logger.warning("STT timed out waiting for utterance.")
            await self.tts.speak("Sorry, I didn't catch that.")
            return

        if not transcript or not transcript.text.strip():
            return

        logger.info("Transcript: %s", transcript.text)

        # Step 3: NLU - extract intent
        try:
            intent = await asyncio.wait_for(
                self.nlu.parse(transcript.text),
                timeout=self.config.nlu.timeout_seconds
            )
        except asyncio.TimeoutError:
            logger.error("NLU timed out.")
            await self.tts.speak("I'm having trouble understanding. Please try again.")
            return

        # Step 4: Update context
        self.context.update(transcript=transcript.text, intent=intent)

        # Step 5: Task planning
        try:
            plan = await asyncio.wait_for(
                self.planner.plan(intent, self.context),
                timeout=self.config.planner.timeout_seconds
            )
        except asyncio.TimeoutError:
            logger.error("Task planner timed out.")
            await self.tts.speak("I couldn't figure out how to do that.")
            return

        if plan is None:
            await self.tts.speak("I don't know how to handle that yet.")
            return

        # Step 6: Safety check
        approved, message = await self.safety.validate(
            plan=plan,
            tts=self.tts
        )
        if not approved:
            logger.warning("Plan blocked by safety layer: %s", message)
            await self.tts.speak(f"I can't do that. {message}")
            return

        # Step 7: Execute plan
        try:
            result = await asyncio.wait_for(
                self.executor.execute(plan, self.context),
                timeout=self.config.executor.timeout_seconds
            )
        except asyncio.TimeoutError:
            logger.error("Executor timed out.")
            await self.tts.speak("That took too long and was cancelled.")
            return

        # Step 8: Respond
        response_text = result.response_text or "Done."
        await self.tts.speak(response_text)
        self.context.record_execution(plan=plan, result=result)
