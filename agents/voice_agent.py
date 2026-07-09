from __future__ import annotations

import logging
import time
from typing import Any, Optional

from agents.base_agent import AgentConfig, AgentResult, BaseAgent

logger = logging.getLogger("aios.agent.voice")


class VoiceAgent(BaseAgent):
    def __init__(self, config: AgentConfig | None = None) -> None:
        super().__init__(config or AgentConfig(
            name="voice",
            system_prompt=(
                "You are a speech processing agent. Transcribe audio, synthesize speech, "
                "detect speaker emotions, and process voice commands."
            ),
        ))

    async def execute(self, task: str, context: dict[str, Any] | None = None) -> AgentResult:
        start = time.perf_counter()
        try:
            ctx = context or {}
            audio_path = ctx.get("audio_path", "")
            task_type = self._classify_task(task)

            output = f"## Voice Processing\n\n"
            output += f"**Task:** {task}\n"
            output += f"**Processing Type:** {task_type}\n"
            if audio_path:
                output += f"**Audio Source:** `{audio_path}`\n\n"
            output += self._process(task_type, audio_path)

            duration = (time.perf_counter() - start) * 1000
            return AgentResult(
                success=True, output=output, agent_name=self.name,
                duration_ms=duration, metadata={"task_type": task_type},
            )
        except Exception as exc:
            duration = (time.perf_counter() - start) * 1000
            return AgentResult(
                success=False, output="", agent_name=self.name,
                duration_ms=duration, error=str(exc),
            )

    @staticmethod
    def _classify_task(task: str) -> str:
        tl = task.lower()
        if any(w in tl for w in ["transcribe", "speech to text", "stt", "recognize speech"]):
            return "transcription"
        if any(w in tl for w in ["synthesize", "text to speech", "tts", "speak"]):
            return "synthesis"
        if any(w in tl for w in ["emotion", "sentiment", "tone"]):
            return "emotion_detection"
        if any(w in tl for w in ["command", "voice command", "control"]):
            return "command_recognition"
        return "processing"

    @staticmethod
    def _process(task_type: str, audio_path: str) -> str:
        if task_type == "transcription":
            return "### Transcription\n- Speech recognition requires an audio input.\n\n"
        if task_type == "synthesis":
            return "### Synthesis\n- Text-to-speech output will be generated.\n\n"
        if task_type == "emotion_detection":
            return "### Emotion Analysis\n- Speaker emotion detection requires audio input.\n\n"
        if task_type == "command_recognition":
            return "### Voice Commands\n- Voice command processing ready.\n\n"
        return "### Processing\n- Voice processing pending.\n"
