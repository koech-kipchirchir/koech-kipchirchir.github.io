from __future__ import annotations

import logging
import time
from typing import Any, Optional

from agents.base_agent import AgentConfig, AgentResult, BaseAgent

logger = logging.getLogger("aios.agent.vision")


class VisionAgent(BaseAgent):
    def __init__(self, config: AgentConfig | None = None) -> None:
        super().__init__(config or AgentConfig(
            name="vision",
            system_prompt=(
                "You are a computer vision agent. Analyze images, detect objects, "
                "recognize patterns, extract text (OCR), and describe visual content."
            ),
        ))

    async def execute(self, task: str, context: dict[str, Any] | None = None) -> AgentResult:
        start = time.perf_counter()
        try:
            ctx = context or {}
            image_path = ctx.get("image_path") or ctx.get("image_url", "")
            task_type = self._classify_task(task)

            output = f"## Vision Analysis\n\n"
            output += f"**Task:** {task}\n"
            output += f"**Analysis Type:** {task_type}\n"
            if image_path:
                output += f"**Source:** `{image_path}`\n\n"
            output += self._analyze(task_type, task, image_path)

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
        if any(w in tl for w in ["detect", "find", "locate", "object"]):
            return "object_detection"
        if any(w in tl for w in ["ocr", "text", "read", "extract"]):
            return "ocr"
        if any(w in tl for w in ["classify", "identify", "what is", "recognize"]):
            return "classification"
        if any(w in tl for w in ["describe", "caption", "explain"]):
            return "captioning"
        return "analysis"

    @staticmethod
    def _analyze(task_type: str, task: str, image_path: str) -> str:
        if task_type == "object_detection":
            return "### Detected Objects\n- Object detection requires a model and image input.\n"
        if task_type == "ocr":
            return "### OCR Result\n- Text extraction requires an image with visible text.\n"
        if task_type == "classification":
            return "### Classification\n- Image classification requires a trained model.\n"
        if task_type == "captioning":
            return "### Caption\n- Image captioning requires a vision-language model.\n"
        return "### Analysis\n- Visual analysis pending input image.\n"
