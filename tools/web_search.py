from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any

from tools.base_tool import BaseTool, ToolInput, ToolOutput


class WebSearchTool(BaseTool):
    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web or fetch content from a URL"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query or URL to fetch"},
                "max_results": {"type": "integer", "description": "Maximum results (default: 5)"},
            },
            "required": ["query"],
        }

    async def execute(self, inp: ToolInput) -> ToolOutput:
        query = inp.arguments.get("query", "")
        max_results = inp.arguments.get("max_results", 5)

        if not query:
            return ToolOutput(success=False, error="No query provided")

        if query.startswith(("http://", "https://")):
            return await self._fetch_url(query)

        try:
            return await self._search_duckduckgo(query, max_results)
        except Exception as exc:
            return ToolOutput(success=False, data={"query": query}, error=str(exc))

    async def _fetch_url(self, url: str) -> ToolOutput:
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (AIOS/1.0)"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                content = resp.read().decode("utf-8", errors="replace")
                return ToolOutput(
                    success=True,
                    data={"url": url, "content": content[:10000], "status": resp.status},
                )
        except Exception as exc:
            return ToolOutput(success=False, error=f"Failed to fetch URL: {exc}")

    async def _search_duckduckgo(self, query: str, max_results: int) -> ToolOutput:
        encoded = urllib.parse.quote(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (AIOS/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        import re
        results = []
        for match in re.finditer(
            r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            html,
            re.DOTALL,
        ):
            link = match.group(1)
            title = re.sub(r"<[^>]+>", "", match.group(2)).strip()
            snippet_match = re.search(
                rf'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
                html[match.end():match.end() + 500],
                re.DOTALL,
            )
            snippet = ""
            if snippet_match:
                snippet = re.sub(r"<[^>]+>", "", snippet_match.group(1)).strip()
            results.append({"title": title, "url": link, "snippet": snippet})
            if len(results) >= max_results:
                break

        return ToolOutput(success=True, data={"query": query, "results": results})
