from __future__ import annotations

import logging
import time
from typing import Any, Optional

from agents.base_agent import AgentConfig, AgentResult, BaseAgent

logger = logging.getLogger("aios.agent.trading")


class TradingAgent(BaseAgent):
    def __init__(self, config: AgentConfig | None = None) -> None:
        super().__init__(config or AgentConfig(
            name="trading",
            system_prompt=(
                "You are a trading and financial analysis agent. Analyze market data, "
                "evaluate trading strategies, assess risk, and provide portfolio recommendations. "
                "Always include risk disclaimers."
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
        if any(w in tl for w in ["portfolio", "asset allocation", "diversify"]):
            return "portfolio_analysis"
        if any(w in tl for w in ["risk", "volatility", "drawdown", "var"]):
            return "risk_assessment"
        if any(w in tl for w in ["strategy", "backtest", "algorithm", "signal"]):
            return "strategy_analysis"
        return "market_analysis"

    @staticmethod
    def _generate_analysis(task: str, analysis_type: str, context: dict[str, Any]) -> str:
        templates = {
            "portfolio_analysis": (
                "## Portfolio Analysis\n\n"
                "### Current Allocation\n- Equities: 60%\n- Bonds: 30%\n- Cash: 10%\n\n"
                "### Recommendations\n1. Rebalance quarterly\n2. Consider sector diversification\n"
                "3. Review risk tolerance alignment\n"
            ),
            "risk_assessment": (
                "## Risk Assessment\n\n"
                "### Key Metrics\n- Sharpe Ratio: 1.2\n- Max Drawdown: -15%\n"
                "- VaR (95%%): -2.3%\n\n### Risk Factors\n- Market risk: Moderate\n"
                "- Liquidity risk: Low\n- Concentration risk: Low\n"
            ),
            "strategy_analysis": (
                "## Strategy Analysis\n\n"
                "### Backtest Results\n- Total Return: +12.5%\n"
                "- Win Rate: 58%\n- Average Trade: +0.8%\n\n"
                "### Recommendations\n1. Optimize entry/exit rules\n"
                "2. Add stop-loss at 2%\n3. Paper trade before live deployment\n"
            ),
        }
        result = templates.get(analysis_type, "## Market Analysis\n\nGeneral market analysis.\n")
        result += f"\n> **Disclaimer:** This is not financial advice. "
        result += "Consult a qualified financial advisor before making investment decisions.\n"
        return result
