from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from typing import Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, Header, Query, Request
from pydantic import BaseModel, Field

from api.dependencies import get_config, get_request_id
from api.exceptions import BadRequestError, EngineError, NotFoundError, ServiceUnavailableError
from api.config import ApiConfig
from tools.base_tool import BaseTool, ToolInput, ToolOutput

logger = logging.getLogger("aios.api.tools")

router = APIRouter(prefix="/tools", tags=["Tools"])


# ---------------------------------------------------------------------------
# Dynamic tool — wraps an API-registered tool definition as a BaseTool
# ---------------------------------------------------------------------------

class DynamicTool(BaseTool):
    def __init__(
        self,
        name: str,
        description: str = "",
        parameters: dict[str, Any] | None = None,
        requires_permission: bool = False,
        endpoint: str | None = None,
    ) -> None:
        self._name = name
        self._description = description
        self._parameters = parameters or {}
        self._requires_permission = requires_permission
        self._endpoint = endpoint
        super().__init__()

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    @property
    def requires_permission(self) -> bool:
        return self._requires_permission

    async def execute(self, inp: ToolInput) -> ToolOutput:
        if self._endpoint:
            try:
                import httpx
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.post(
                        self._endpoint,
                        json=inp.arguments,
                        headers={"Content-Type": "application/json"},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    return ToolOutput(success=True, data=data)
            except Exception as exc:
                return ToolOutput(success=False, error=f"Endpoint call failed: {exc}")
        return ToolOutput(success=True, data={"tool": self._name, "arguments": inp.arguments, "status": "invoked"})


# ---------------------------------------------------------------------------
# Execution metrics
# ---------------------------------------------------------------------------

_METRICS_LOCK: Any = None
try:
    import threading
    _METRICS_LOCK = threading.Lock()
except ImportError:
    pass


class ToolExecutionMetricsSchema(BaseModel):
    tool_name: str = Field("", description="Tool name")
    success: bool = Field(True, description="Whether execution succeeded")
    duration_ms: float = Field(0.0, description="Execution duration in milliseconds")
    timestamp: float = Field(0.0, description="Unix timestamp of execution")


class ToolMetricsSnapshot(BaseModel):
    total_executions: int = Field(0, description="Total executions since startup")
    total_success: int = Field(0, description="Successful executions")
    total_failed: int = Field(0, description="Failed executions")
    avg_duration_ms: float = Field(0.0, description="Average execution duration")
    per_tool: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-tool breakdown",
    )


class MetricsStore:
    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []
        self._per_tool: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"executions": 0, "success": 0, "failed": 0, "total_duration_ms": 0.0}
        )

    def record(self, tool_name: str, success: bool, duration_ms: float) -> None:
        rec = {
            "tool_name": tool_name,
            "success": success,
            "duration_ms": duration_ms,
            "timestamp": time.time(),
        }
        if _METRICS_LOCK:
            with _METRICS_LOCK:
                self._records.append(rec)
                pt = self._per_tool[tool_name]
                pt["executions"] += 1
                pt["total_duration_ms"] += duration_ms
                if success:
                    pt["success"] += 1
                else:
                    pt["failed"] += 1
        else:
            self._records.append(rec)
            pt = self._per_tool[tool_name]
            pt["executions"] += 1
            pt["total_duration_ms"] += duration_ms
            if success:
                pt["success"] += 1
            else:
                pt["failed"] += 1

        self._records = self._records[-1000:]

    def snapshot(self) -> ToolMetricsSnapshot:
        if _METRICS_LOCK:
            with _METRICS_LOCK:
                return self._build_snapshot()
        return self._build_snapshot()

    def _build_snapshot(self) -> ToolMetricsSnapshot:
        total = len(self._records)
        success = sum(1 for r in self._records if r["success"])
        failed = total - success
        avg = (
            sum(r["duration_ms"] for r in self._records) / total
            if total > 0 else 0.0
        )
        per_tool = {}
        for name, pt in self._per_tool.items():
            per_tool[name] = {
                "executions": pt["executions"],
                "success": pt["success"],
                "failed": pt["failed"],
                "avg_duration_ms": round(
                    pt["total_duration_ms"] / pt["executions"], 2
                ) if pt["executions"] > 0 else 0.0,
            }
        return ToolMetricsSnapshot(
            total_executions=total,
            total_success=success,
            total_failed=failed,
            avg_duration_ms=round(avg, 2),
            per_tool=per_tool,
        )


_metrics = MetricsStore()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ToolInfoSchema(BaseModel):
    name: str = Field(..., description="Tool name")
    description: str = Field("", description="Human-readable description")
    parameters: dict[str, Any] = Field(default_factory=dict, description="JSON Schema for arguments")
    requires_permission: bool = Field(False, description="Whether explicit permission is needed")
    is_dynamic: bool = Field(False, description="Whether tool was registered via API")


class RunToolRequest(BaseModel):
    tool_name: str = Field(..., min_length=1, description="Tool to execute")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
    permission_override: bool | None = Field(
        None,
        description="Set True to bypass permission check",
    )
    timeout: int | None = Field(None, ge=1, le=300, description="Execution timeout in seconds")

    model_config = {"json_schema_extra": {"example": {"tool_name": "calculator", "arguments": {"expression": "2 + 2"}}}}


class RunToolResponse(BaseModel):
    success: bool = Field(..., description="Whether execution succeeded")
    data: Any = Field(None, description="Tool output data")
    error: str = Field("", description="Error message if failed")
    tool_name: str = Field("", description="Executed tool name")
    duration_ms: float = Field(0.0, description="Execution duration")
    mime_type: str = Field("text/plain", description="Response MIME type")


class RegisterToolRequest(BaseModel):
    name: str = Field(..., min_length=1, pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$", description="Unique tool name")
    description: str = Field("", description="Human-readable description")
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}, "required": []},
        description="JSON Schema for arguments",
    )
    requires_permission: bool = Field(False, description="Whether permission is needed to execute")
    endpoint: str | None = Field(None, description="Optional webhook URL for execution logic")


class RegisterToolResponse(BaseModel):
    status: str = Field("ok", description="Registration status")
    name: str = Field("", description="Registered tool name")
    is_dynamic: bool = Field(True, description="Whether tool was registered dynamically")


class DeleteToolResponse(BaseModel):
    status: str = Field("ok", description="Operation status")
    name: str = Field("", description="Unregistered tool name")
    deleted: bool = Field(True, description="Whether tool was found and unregistered")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_manager(request: Request):
    mgr = getattr(request.app.state, "tool_manager", None)
    if mgr is None:
        raise ServiceUnavailableError(detail="Tool manager not initialized")
    return mgr


def _tool_to_schema(tool: BaseTool, is_dynamic: bool = False) -> ToolInfoSchema:
    return ToolInfoSchema(
        name=tool.name,
        description=tool.description,
        parameters=tool.parameters,
        requires_permission=tool.requires_permission,
        is_dynamic=is_dynamic,
    )


# ---------------------------------------------------------------------------
# GET /tools  — List available tools
# ---------------------------------------------------------------------------

@router.get("", summary="List all registered tools")
async def list_tools(
    request: Request,
) -> list[ToolInfoSchema]:
    mgr = _get_manager(request)
    try:
        tool_dicts = mgr.list_tools()
    except Exception as exc:
        logger.error("List tools error: %s", exc)
        raise EngineError(detail=str(exc))

    results: list[ToolInfoSchema] = []
    for td in tool_dicts:
        results.append(ToolInfoSchema(
            name=td.get("name", ""),
            description=td.get("description", ""),
            parameters=td.get("parameters", {}),
            requires_permission=td.get("requires_permission", False),
            is_dynamic=td.get("name", "") not in getattr(request.app.state, "_builtin_tools", set()),
        ))
    return results


# ---------------------------------------------------------------------------
# POST /tools/run  — Execute a tool
# ---------------------------------------------------------------------------

@router.post("/run", summary="Execute a tool")
async def run_tool(
    body: RunToolRequest,
    request: Request,
) -> RunToolResponse:
    mgr = _get_manager(request)

    tool = mgr.get_tool(body.tool_name)
    if tool is None:
        raise NotFoundError(detail=f"Tool not found: '{body.tool_name}'")

    arguments = dict(body.arguments)
    if body.timeout is not None:
        arguments.setdefault("metadata", {})
        arguments["metadata"]["timeout"] = body.timeout

    start = time.perf_counter()
    try:
        output = await mgr.execute(
            tool_name=body.tool_name,
            arguments=arguments,
            permission_override=body.permission_override,
        )
    except Exception as exc:
        duration = (time.perf_counter() - start) * 1000
        _metrics.record(body.tool_name, False, duration)
        logger.error("Tool '%s' execution error: %s", body.tool_name, exc)
        raise EngineError(detail=str(exc))

    duration = (time.perf_counter() - start) * 1000
    _metrics.record(body.tool_name, output.success, duration)
    logger.info(
        "Tool '%s' executed: success=%s duration=%.0fms",
        body.tool_name, output.success, duration,
    )

    return RunToolResponse(
        success=output.success,
        data=output.data,
        error=output.error,
        tool_name=body.tool_name,
        duration_ms=round(duration, 2),
        mime_type=output.mime_type,
    )


# ---------------------------------------------------------------------------
# GET /tools/status  — Tool execution metrics
# ---------------------------------------------------------------------------

@router.get("/status", summary="Get tool execution metrics")
async def tool_status(
    request: Request,
) -> ToolMetricsSnapshot:
    return _metrics.snapshot()


# ---------------------------------------------------------------------------
# POST /tools/register  — Dynamically register a new tool
# ---------------------------------------------------------------------------

@router.post("/register", summary="Register a new tool dynamically", status_code=201)
async def register_tool(
    body: RegisterToolRequest,
    request: Request,
) -> RegisterToolResponse:
    mgr = _get_manager(request)

    existing = mgr.get_tool(body.name)
    if existing is not None:
        raise BadRequestError(detail=f"Tool '{body.name}' is already registered")

    try:
        dt = DynamicTool(
            name=body.name,
            description=body.description,
            parameters=body.parameters,
            requires_permission=body.requires_permission,
            endpoint=body.endpoint,
        )
        mgr.register_tool(dt)
    except Exception as exc:
        logger.error("Failed to register tool '%s': %s", body.name, exc)
        raise EngineError(detail=str(exc))

    logger.info("Dynamic tool registered: %s", body.name)
    return RegisterToolResponse(status="ok", name=body.name, is_dynamic=True)


# ---------------------------------------------------------------------------
# DELETE /tools  — Unregister a tool
# ---------------------------------------------------------------------------

@router.delete("", summary="Unregister a tool")
async def delete_tool(
    request: Request,
    name: str = Query(..., min_length=1, description="Tool name to unregister"),
) -> DeleteToolResponse:
    mgr = _get_manager(request)

    existing = mgr.get_tool(name)
    if existing is None:
        raise NotFoundError(detail=f"Tool not found: '{name}'")

    if name in getattr(request.app.state, "_builtin_tools", set()):
        raise BadRequestError(detail=f"Cannot unregister built-in tool: '{name}'")

    try:
        mgr.unregister_tool(name)
    except Exception as exc:
        logger.error("Failed to unregister tool '%s': %s", name, exc)
        raise EngineError(detail=str(exc))

    logger.info("Tool unregistered: %s", name)
    return DeleteToolResponse(status="ok", name=name, deleted=True)
