from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any

from tools.base_tool import BaseTool, ToolInput, ToolOutput


class DateTimeTool(BaseTool):
    @property
    def name(self) -> str:
        return "datetime"

    @property
    def description(self) -> str:
        return "Get current date, time, timezone conversions, and date arithmetic"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["now", "format", "convert", "difference", "add"],
                    "description": "Date/time operation",
                },
                "format": {
                    "type": "string",
                    "description": "Output format (strftime format, default: %Y-%m-%d %H:%M:%S %Z)",
                },
                "timezone": {
                    "type": "string",
                    "description": "Target timezone (e.g., UTC, US/Eastern)",
                },
                "date1": {"type": "string", "description": "First date (ISO format)"},
                "date2": {"type": "string", "description": "Second date (ISO format)"},
                "days": {"type": "integer", "description": "Days to add/subtract"},
            },
            "required": ["action"],
        }

    async def execute(self, inp: ToolInput) -> ToolOutput:
        action = inp.arguments.get("action", "")
        fmt = inp.arguments.get("format", "%Y-%m-%d %H:%M:%S %Z")
        tz_name = inp.arguments.get("timezone", "UTC")

        try:
            data: dict[str, Any] = {"action": action}

            if action == "now":
                now = datetime.now(timezone.utc)
                data["utc"] = now.isoformat()
                data["unix_timestamp"] = int(now.timestamp())
                data["formatted"] = now.strftime(fmt)
                data["timezone"] = "UTC"

            elif action == "format":
                date_str = inp.arguments.get("date1", "")
                if not date_str:
                    return ToolOutput(success=False, error="date1 required for format action")
                dt = datetime.fromisoformat(date_str)
                data["input"] = date_str
                data["formatted"] = dt.strftime(fmt)

            elif action == "difference":
                d1 = inp.arguments.get("date1", "")
                d2 = inp.arguments.get("date2", "")
                if not d1 or not d2:
                    return ToolOutput(success=False, error="date1 and date2 required")
                dt1 = datetime.fromisoformat(d1)
                dt2 = datetime.fromisoformat(d2)
                diff = abs(dt2 - dt1)
                data["date1"] = d1
                data["date2"] = d2
                data["difference_days"] = diff.days
                data["difference_seconds"] = int(diff.total_seconds())

            elif action == "add":
                d1 = inp.arguments.get("date1", "")
                days = inp.arguments.get("days", 0)
                if not d1:
                    return ToolOutput(success=False, error="date1 required for add action")
                dt = datetime.fromisoformat(d1)
                result = dt + timedelta(days=days)
                data["input"] = d1
                data["days_added"] = days
                data["result"] = result.isoformat()

            else:
                return ToolOutput(success=False, error=f"Unknown action: {action}")

            return ToolOutput(success=True, data=data)
        except Exception as exc:
            return ToolOutput(success=False, error=str(exc))
