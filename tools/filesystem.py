from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from tools.base_tool import BaseTool, ToolInput, ToolOutput


class FileSystemTool(BaseTool):
    def __init__(self, allowed_root: str | None = None) -> None:
        super().__init__()
        self._allowed_root = Path(allowed_root).resolve() if allowed_root else None

    @property
    def name(self) -> str:
        return "filesystem"

    @property
    def description(self) -> str:
        return "Read, write, list, and manage files and directories"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["read", "write", "list", "exists", "delete", "mkdir", "info"],
                    "description": "File operation to perform",
                },
                "path": {"type": "string", "description": "File or directory path"},
                "content": {"type": "string", "description": "Content to write (for write action)"},
            },
            "required": ["action", "path"],
        }

    @property
    def requires_permission(self) -> bool:
        return True

    async def execute(self, inp: ToolInput) -> ToolOutput:
        action = inp.arguments.get("action", "")
        path_str = inp.arguments.get("path", "")
        content = inp.arguments.get("content", "")

        if not action or not path_str:
            return ToolOutput(success=False, error="action and path are required")

        path = Path(path_str).resolve()

        if self._allowed_root and not str(path).startswith(str(self._allowed_root)):
            return ToolOutput(success=False, error=f"Path not allowed: {path}")

        try:
            data: dict[str, Any] = {"path": str(path), "action": action}
            if action == "read":
                if not path.exists():
                    return ToolOutput(success=False, error=f"File not found: {path}")
                data["content"] = path.read_text(encoding="utf-8", errors="replace")
                data["size"] = path.stat().st_size
            elif action == "write":
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
                data["size"] = len(content)
            elif action == "list":
                if not path.exists():
                    return ToolOutput(success=False, error=f"Directory not found: {path}")
                entries = []
                for entry in path.iterdir():
                    entries.append({
                        "name": entry.name,
                        "type": "directory" if entry.is_dir() else "file",
                        "size": entry.stat().st_size if entry.is_file() else 0,
                    })
                data["entries"] = entries
                data["count"] = len(entries)
            elif action == "exists":
                data["exists"] = path.exists()
            elif action == "delete":
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
                data["deleted"] = True
            elif action == "mkdir":
                path.mkdir(parents=True, exist_ok=True)
                data["created"] = True
            elif action == "info":
                if not path.exists():
                    return ToolOutput(success=False, error=f"Path not found: {path}")
                stat = path.stat()
                data.update({
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                    "type": "directory" if path.is_dir() else "file",
                })
            else:
                return ToolOutput(success=False, error=f"Unknown action: {action}")

            return ToolOutput(success=True, data=data)
        except PermissionError:
            return ToolOutput(success=False, error=f"Permission denied: {path}")
        except Exception as exc:
            return ToolOutput(success=False, error=str(exc))
