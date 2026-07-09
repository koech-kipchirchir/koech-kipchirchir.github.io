from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any

from tools.base_tool import BaseTool, ToolInput, ToolOutput


class WeatherTool(BaseTool):
    @property
    def name(self) -> str:
        return "weather"

    @property
    def description(self) -> str:
        return "Get current weather for a location using free APIs"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or coordinates (lat,lon)",
                },
            },
            "required": ["location"],
        }

    async def execute(self, inp: ToolInput) -> ToolOutput:
        location = inp.arguments.get("location", "")
        if not location:
            return ToolOutput(success=False, error="No location provided")

        try:
            return await self._fetch_weather(location)
        except Exception as exc:
            return ToolOutput(success=False, data={"location": location}, error=str(exc))

    async def _fetch_weather(self, location: str) -> ToolOutput:
        if "," in location and location.replace(",", "").replace(".", "").replace("-", "").strip().replace(" ", "").isdigit():
            lat, lon = location.split(",")
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat.strip()}&longitude={lon.strip()}&current_weather=true&timezone=auto"
        else:
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(location)}&count=1&language=en&format=json"
            req = urllib.request.Request(geo_url, headers={"User-Agent": "AIOS/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                geo_data = json.loads(resp.read().decode("utf-8"))

            if "results" not in geo_data or not geo_data["results"]:
                return ToolOutput(success=False, error=f"Location not found: {location}")

            result = geo_data["results"][0]
            lat = result["latitude"]
            lon = result["longitude"]
            location = f"{result.get('name', location)}, {result.get('country', '')}"
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&timezone=auto"

        req = urllib.request.Request(url, headers={"User-Agent": "AIOS/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            weather_data = json.loads(resp.read().decode("utf-8"))

        current = weather_data.get("current_weather", {})
        return ToolOutput(success=True, data={
            "location": location,
            "temperature_c": current.get("temperature"),
            "windspeed_kmh": current.get("windspeed"),
            "wind_direction": current.get("winddirection"),
            "weather_code": current.get("weathercode"),
            "time": current.get("time"),
        })
