from __future__ import annotations

import logging
import re
import time
from typing import Any, Optional

from agents.base_agent import AgentConfig, AgentResult, BaseAgent

logger = logging.getLogger("aios.agent.cybersecurity")


class CybersecurityAgent(BaseAgent):
    def __init__(self, config: AgentConfig | None = None) -> None:
        super().__init__(config or AgentConfig(
            name="cybersecurity",
            system_prompt=(
                "You are a cybersecurity expert. Analyze threats, identify vulnerabilities, "
                "recommend mitigations, and assess security postures. "
                "Always follow responsible disclosure principles."
            ),
        ))

    async def execute(self, task: str, context: dict[str, Any] | None = None) -> AgentResult:
        start = time.perf_counter()
        try:
            analysis_type = self._classify_analysis(task)
            output = self._generate_analysis(task, analysis_type, context or {})
            duration = (time.perf_counter() - start) * 1000
            return AgentResult(
                success=True, output=output, agent_name=self.name,
                duration_ms=duration, metadata={"analysis_type": analysis_type},
            )
        except Exception as exc:
            duration = (time.perf_counter() - start) * 1000
            return AgentResult(
                success=False, output="", agent_name=self.name,
                duration_ms=duration, error=str(exc),
            )

    @staticmethod
    def _classify_analysis(task: str) -> str:
        tl = task.lower()
        if any(w in tl for w in ["vulnerability", "cve", "exploit", "bug"]):
            return "vulnerability_assessment"
        if any(w in tl for w in ["phishing", "social engineering", "scam"]):
            return "phishing_analysis"
        if any(w in tl for w in ["malware", "virus", "ransomware", "trojan"]):
            return "malware_analysis"
        if any(w in tl for w in ["network", "firewall", "port", "scan"]):
            return "network_security"
        if any(w in tl for w in ["password", "authentication", "mfa", "credential"]):
            return "authentication_review"
        return "general_security"

    @staticmethod
    def _generate_analysis(task: str, analysis_type: str, context: dict[str, Any]) -> str:
        template = {
            "vulnerability_assessment": (
                "## Vulnerability Assessment\n\n"
                "### Summary\nPotential security vulnerability identified.\n\n"
                "### Risk Level\nMedium\n\n"
                "### Recommended Actions\n"
                "1. Verify the vulnerability in a controlled environment\n"
                "2. Apply security patches if available\n"
                "3. Implement input validation and sanitization\n"
                "4. Review access controls\n"
            ),
            "phishing_analysis": (
                "## Phishing Analysis\n\n"
                "### Indicators\n- Suspicious sender address\n- Urgency pressure tactics\n"
                "- Generic greetings\n- Suspicious links\n\n"
                "### Verdict\nPotentially malicious — do not interact.\n"
            ),
            "malware_analysis": (
                "## Malware Analysis\n\n"
                "### Behavior\nAnalyzing sample behavior...\n\n"
                "### IOCs\n- Network connections: Unknown\n"
                "- File modifications: Pending analysis\n\n"
                "### Recommendations\nIsolate affected systems immediately.\n"
            ),
        }
        result = template.get(analysis_type, "## Security Analysis\n\nGeneral security analysis.\n")
        result += f"\n**Task:** {task}\n"
        if context:
            result += f"\n**Context:** {context}\n"
        return result
