"""
AIOS Agent Bus
==============

Async message bus and event system for agent-to-agent communication.

Capabilities:
- Typed message passing (direct, broadcast, request/reply)
- Pub/sub event system with topic filtering
- Shared context with change notifications
- Priority task queue with concurrency control
- Structured logging with event tracing
- Rate limiting and graceful shutdown

Usage::

    bus = AgentBus()
    await bus.start()

    # Subscribe to events
    async def on_event(event: BusEvent): ...
    bus.subscribe("memory.stored", on_event)

    # Send a message
    await bus.send(BusMessage(sender="agent_a", recipient="agent_b", payload={...}))

    # Publish an event
    await bus.publish(BusEvent(event_type="research.complete", data={...}))

    # Enqueue a task
    await bus.enqueue(TaskMessage(name="process", coro=some_coro(), priority=5))

    # Share context
    await bus.context_set("key", "value")
    val = await bus.context_get("key")

    await bus.shutdown()
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

logger = logging.getLogger("aios.agent.bus")

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MessageType(Enum):
    DIRECT = "direct"
    BROADCAST = "broadcast"
    REQUEST = "request"
    REPLY = "reply"
    FORWARD = "forward"


class MessagePriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class SubscriptionPolicy(Enum):
    EXACT = "exact"          # topic must match exactly
    PREFIX = "prefix"        # topic must start with pattern
    WILDCARD = "wildcard"    # glob-style: "research.*" matches "research.complete"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BusMessage:
    """A typed message sent between agents on the bus."""

    sender: str
    recipient: str | None = None           # None = broadcast
    payload: Any = None
    message_type: MessageType = MessageType.DIRECT
    message_id: str = ""
    correlation_id: str = ""                # for request/reply pairing
    reply_to: str = ""                      # topic/queue for reply
    priority: MessagePriority = MessagePriority.NORMAL
    ttl_seconds: float = 60.0
    timestamp: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    trace_id: str = ""

    def __post_init__(self) -> None:
        if not self.message_id:
            self.message_id = _generate_id()
        if not self.timestamp:
            self.timestamp = _now()
        if not self.trace_id:
            self.trace_id = _generate_id()[:8]


@dataclass
class BusEvent:
    """A typed event published on the bus."""

    event_type: str
    data: Any = None
    source: str = ""
    event_id: str = ""
    timestamp: str = ""
    trace_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_id:
            self.event_id = _generate_id()
        if not self.timestamp:
            self.timestamp = _now()
        if not self.trace_id:
            self.trace_id = _generate_id()[:8]


@dataclass
class TaskMessage:
    """A task enqueued on the bus for execution."""

    name: str
    coro: Awaitable[Any] | None = None
    priority: int = 0                       # higher = more urgent
    timeout: float = 120.0
    task_id: str = ""
    sender: str = ""
    max_retries: int = 0
    retry_delay: float = 1.0
    status: TaskStatus = TaskStatus.PENDING
    created_at: str = ""
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.task_id:
            self.task_id = _generate_id()
        if not self.created_at:
            self.created_at = _now()


@dataclass
class Subscription:
    """A subscription binding a topic pattern to a callback."""

    topic: str
    callback: Callable[[BusEvent], Awaitable[None]]
    policy: SubscriptionPolicy = SubscriptionPolicy.EXACT
    subscriber_id: str = ""
    subscriber_name: str = ""                # human-readable name
    created_at: str = ""
    max_events: int = 0                      # 0 = unlimited

    def __post_init__(self) -> None:
        if not self.subscriber_id:
            self.subscriber_id = _generate_id()
        if not self.created_at:
            self.created_at = _now()

    def matches(self, event_type: str) -> bool:
        if self.policy == SubscriptionPolicy.EXACT:
            return self.topic == event_type
        if self.policy == SubscriptionPolicy.PREFIX:
            return event_type.startswith(self.topic)
        if self.policy == SubscriptionPolicy.WILDCARD:
            return _wildcard_match(self.topic, event_type)
        return False


@dataclass
class BusMetrics:
    """Snapshot of bus performance metrics."""

    messages_sent: int = 0
    messages_delivered: int = 0
    messages_failed: int = 0
    events_published: int = 0
    events_dispatched: int = 0
    tasks_enqueued: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    tasks_timed_out: int = 0
    context_updates: int = 0
    active_task_count: int = 0
    queue_depth: int = 0
    subscriber_count: int = 0


@dataclass
class ContextChange:
    """Record of a shared context change for event propagation."""

    key: str
    old_value: Any = None
    new_value: Any = None
    source: str = ""
    timestamp: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STOP = object()  # sentinel to signal shutdown


def _generate_id() -> str:
    return uuid.uuid4().hex[:12]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _wildcard_match(pattern: str, topic: str) -> bool:
    """Simple wildcard match supporting '*' and '?'."""
    if pattern == topic:
        return True
    if pattern == "*":
        return True
    parts_p = pattern.split(".")
    parts_t = topic.split(".")
    if len(parts_p) != len(parts_t):
        return False
    for p, t in zip(parts_p, parts_t):
        if p == "*":
            continue
        if p.endswith("*"):
            if not t.startswith(p[:-1]):
                return False
        elif p != t:
            return False
    return True


def _structured_log(level: int, event: str, **kwargs: Any) -> None:
    record = {"event": event, "ts": time.time()}
    record.update(kwargs)
    logger.log(level, "%s", json.dumps(record, default=str))


# ---------------------------------------------------------------------------
# Shared Context
# ---------------------------------------------------------------------------


class SharedContext:
    """Async-safe key-value store with change notification."""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._change_history: list[ContextChange] = []
        self._max_history = 500
        self._change_events: asyncio.Event | None = None

    async def get(self, key: str, default: Any = None) -> Any:
        async with self._lock:
            return self._store.get(key, default)

    async def set(self, key: str, value: Any, source: str = "") -> ContextChange | None:
        async with self._lock:
            old = self._store.get(key)
            self._store[key] = value
            change = ContextChange(key=key, old_value=old, new_value=value, source=source, timestamp=_now())
            self._change_history.append(change)
            if len(self._change_history) > self._max_history:
                self._change_history.pop(0)
            if self._change_events:
                self._change_events.set()
            return change

    async def delete(self, key: str) -> bool:
        async with self._lock:
            if key not in self._store:
                return False
            old = self._store.pop(key)
            change = ContextChange(key=key, old_value=old, new_value=None, timestamp=_now())
            self._change_history.append(change)
            return True

    async def has(self, key: str) -> bool:
        async with self._lock:
            return key in self._store

    async def keys(self, pattern: str = "") -> list[str]:
        async with self._lock:
            if not pattern:
                return list(self._store.keys())
            return [k for k in self._store if k.startswith(pattern)]

    async def get_all(self) -> dict[str, Any]:
        async with self._lock:
            return dict(self._store)

    async def clear(self) -> int:
        async with self._lock:
            count = len(self._store)
            self._store.clear()
            return count

    async def size(self) -> int:
        async with self._lock:
            return len(self._store)

    async def get_history(self, limit: int = 50) -> list[ContextChange]:
        async with self._lock:
            return list(self._change_history[-limit:])

    async def wait_for_change(self, key: str, timeout: float = 10.0) -> Any:
        """Wait until the given key changes, then return its new value."""
        ev = asyncio.Event()
        self._change_events = ev
        current = await self.get(key)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                await asyncio.wait_for(ev.wait(), timeout=max(0.1, deadline - time.monotonic()))
            except (asyncio.TimeoutError, asyncio.CancelledError):
                break
            ev.clear()
            new_val = await self.get(key)
            if new_val != current:
                return new_val
        return None


# ---------------------------------------------------------------------------
# AgentBus
# ---------------------------------------------------------------------------


class AgentBus:
    """Async message bus and event system for agent communication.

    Usage::

        bus = AgentBus()
        await bus.start()
        ...
        await bus.shutdown()
    """

    def __init__(self, max_queue_size: int = 10_000, max_workers: int = 4) -> None:
        self._message_queue: asyncio.Queue[BusMessage] = asyncio.Queue(maxsize=max_queue_size)
        self._event_handlers: list[Subscription] = []
        self._task_queue: asyncio.PriorityQueue[tuple[int, int, TaskMessage]] = asyncio.PriorityQueue()
        self._context = SharedContext()
        self._running = False
        self._max_workers = max_workers
        self._workers: list[asyncio.Task[Any]] = []
        self._counter = itertools.count()
        self._pending_replies: dict[str, asyncio.Future[Any]] = {}
        self._shutdown_event = asyncio.Event()

        # Metrics
        self._messages_sent = 0
        self._messages_delivered = 0
        self._messages_failed = 0
        self._events_published = 0
        self._events_dispatched = 0
        self._tasks_enqueued = 0
        self._tasks_completed = 0
        self._tasks_failed = 0
        self._tasks_timed_out = 0
        self._context_updates = 0
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the bus workers and begin processing."""
        if self._running:
            return
        self._running = True
        self._shutdown_event.clear()
        # Start message processors
        for _ in range(self._max_workers):
            worker = asyncio.create_task(self._process_messages(), name="bus-msg-worker")
            self._workers.append(worker)
        # Start task processors
        for _ in range(self._max_workers):
            worker = asyncio.create_task(self._process_tasks(), name="bus-task-worker")
            self._workers.append(worker)
        _structured_log(logging.INFO, "bus.started", workers=self._max_workers * 2)

    async def shutdown(self, wait: bool = True, timeout: float = 10.0) -> None:
        """Gracefully shut down the bus."""
        if not self._running:
            return
        self._running = False
        self._shutdown_event.set()

        # Drain queues
        if wait:
            deadline = time.monotonic() + timeout
            while (not self._message_queue.empty() or not self._task_queue.empty()):
                if time.monotonic() > deadline:
                    break
                await asyncio.sleep(0.1)

        # Cancel pending replies
        for fut in self._pending_replies.values():
            if not fut.done():
                fut.cancel()
        self._pending_replies.clear()

        # Cancel workers
        for w in self._workers:
            w.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        _structured_log(logging.INFO, "bus.shutdown")

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Message passing
    # ------------------------------------------------------------------

    async def send(self, message: BusMessage) -> None:
        """Send a message to the bus for delivery."""
        if not self._running:
            raise RuntimeError("Bus is not running. Call start() first.")
        await self._message_queue.put(message)
        async with self._lock:
            self._messages_sent += 1
        _structured_log(logging.DEBUG, "bus.message.sent",
                        msg_id=message.message_id, sender=message.sender,
                        recipient=message.recipient, msg_type=message.message_type.value)

    async def request(self, message: BusMessage, timeout: float = 30.0) -> Any:
        """Send a request and wait for a reply."""
        if not message.correlation_id:
            message.correlation_id = _generate_id()
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self._pending_replies[message.correlation_id] = fut
        try:
            await self.send(message)
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            _structured_log(logging.WARNING, "bus.request.timeout",
                            corr_id=message.correlation_id, timeout=timeout)
            raise
        finally:
            self._pending_replies.pop(message.correlation_id, None)

    async def reply(self, original: BusMessage, payload: Any) -> None:
        """Send a reply to a request message."""
        reply_msg = BusMessage(
            sender=original.recipient or "bus",
            recipient=original.sender,
            payload=payload,
            message_type=MessageType.REPLY,
            correlation_id=original.correlation_id,
            trace_id=original.trace_id,
        )
        await self.send(reply_msg)

    def get_reply_future(self, correlation_id: str) -> asyncio.Future[Any] | None:
        return self._pending_replies.get(correlation_id)

    async def broadcast(self, payload: Any, sender: str = "bus", headers: dict[str, str] | None = None) -> None:
        """Broadcast a message to all agents."""
        await self.send(BusMessage(
            sender=sender, recipient=None, payload=payload,
            message_type=MessageType.BROADCAST, headers=headers or {},
        ))

    # ------------------------------------------------------------------
    # Event system (pub/sub)
    # ------------------------------------------------------------------

    def subscribe(self, topic: str, callback: Callable[[BusEvent], Awaitable[None]],
                  policy: SubscriptionPolicy = SubscriptionPolicy.EXACT,
                  subscriber_name: str = "") -> Subscription:
        """Register a subscriber for events matching the given topic."""
        sub = Subscription(
            topic=topic, callback=callback, policy=policy,
            subscriber_name=subscriber_name or f"anon-{_generate_id()[:6]}",
        )
        self._event_handlers.append(sub)
        _structured_log(logging.DEBUG, "bus.subscribe",
                        subscriber=sub.subscriber_name, topic=topic, policy=policy.value)
        return sub

    def unsubscribe(self, subscriber_id: str) -> bool:
        """Remove a subscription by its ID."""
        before = len(self._event_handlers)
        self._event_handlers = [s for s in self._event_handlers if s.subscriber_id != subscriber_id]
        return len(self._event_handlers) < before

    def unsubscribe_all(self, subscriber_name: str) -> int:
        """Remove all subscriptions for a given subscriber name."""
        before = len(self._event_handlers)
        self._event_handlers = [s for s in self._event_handlers if s.subscriber_name != subscriber_name]
        return before - len(self._event_handlers)

    async def publish(self, event: BusEvent) -> int:
        """Publish an event to all matching subscribers. Returns dispatch count."""
        if not self._running:
            raise RuntimeError("Bus is not running. Call start() first.")
        if not event.source:
            event.source = "bus"
        async with self._lock:
            self._events_published += 1

        matching = [s for s in self._event_handlers if s.matches(event.event_type)]
        count = 0
        for sub in matching:
            try:
                await sub.callback(event)
                count += 1
            except Exception as exc:
                _structured_log(logging.ERROR, "bus.event.callback.failed",
                                subscriber=sub.subscriber_name,
                                event_type=event.event_type, error=str(exc))

        async with self._lock:
            self._events_dispatched += count
        _structured_log(logging.DEBUG, "bus.event.published",
                        event_type=event.event_type, matched=len(matching), dispatched=count)
        return count

    async def get_event_history(self, event_type: str = "", limit: int = 100) -> list[BusEvent]:
        """Retrieve recent events (limited by internal buffer)."""
        return list(self._event_history[-limit:]) if hasattr(self, '_event_history') else []

    @property
    def subscribers(self) -> list[Subscription]:
        return list(self._event_handlers)

    # ------------------------------------------------------------------
    # Task queue
    # ------------------------------------------------------------------

    async def enqueue(self, task: TaskMessage) -> str:
        """Enqueue a task for async execution."""
        if not self._running:
            raise RuntimeError("Bus is not running. Call start() first.")
        priority = task.priority
        count = next(self._counter)
        await self._task_queue.put((-priority, count, task))
        async with self._lock:
            self._tasks_enqueued += 1
        _structured_log(logging.DEBUG, "bus.task.enqueued",
                        task_id=task.task_id, name=task.name, priority=task.priority)
        return task.task_id

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task by ID."""
        # Cannot easily remove from PriorityQueue; mark cancelled instead
        return False

    async def get_task_status(self, task_id: str) -> TaskStatus | None:
        """Get status of a specific task."""
        return None

    # ------------------------------------------------------------------
    # Shared context
    # ------------------------------------------------------------------

    @property
    def context(self) -> SharedContext:
        return self._context

    async def context_set(self, key: str, value: Any, source: str = "") -> None:
        change = await self._context.set(key, value, source=source)
        if change:
            async with self._lock:
                self._context_updates += 1
            # Fire context change event
            await self.publish(BusEvent(
                event_type="context.changed",
                data={"key": key, "source": source},
                source=source or "bus",
            ))

    async def context_get(self, key: str, default: Any = None) -> Any:
        return await self._context.get(key, default)

    async def context_has(self, key: str) -> bool:
        return await self._context.has(key)

    async def context_size(self) -> int:
        return await self._context.size()

    async def context_delete(self, key: str) -> bool:
        return await self._context.delete(key)

    async def context_keys(self, pattern: str = "") -> list[str]:
        return await self._context.keys(pattern)

    async def context_clear(self) -> int:
        return await self._context.clear()

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    async def metrics(self) -> BusMetrics:
        async with self._lock:
            return BusMetrics(
                messages_sent=self._messages_sent,
                messages_delivered=self._messages_delivered,
                messages_failed=self._messages_failed,
                events_published=self._events_published,
                events_dispatched=self._events_dispatched,
                tasks_enqueued=self._tasks_enqueued,
                tasks_completed=self._tasks_completed,
                tasks_failed=self._tasks_failed,
                tasks_timed_out=self._tasks_timed_out,
                context_updates=self._context_updates,
                active_task_count=len(self._workers),
                queue_depth=self._message_queue.qsize(),
                subscriber_count=len(self._event_handlers),
            )

    async def reset_metrics(self) -> None:
        async with self._lock:
            self._messages_sent = 0
            self._messages_delivered = 0
            self._messages_failed = 0
            self._events_published = 0
            self._events_dispatched = 0
            self._tasks_enqueued = 0
            self._tasks_completed = 0
            self._tasks_failed = 0
            self._tasks_timed_out = 0
            self._context_updates = 0

    # ------------------------------------------------------------------
    # Internal processing
    # ------------------------------------------------------------------

    async def _process_messages(self) -> None:
        """Worker loop processing incoming messages."""
        while self._running:
            try:
                message = await asyncio.wait_for(
                    self._message_queue.get(), timeout=1.0,
                )
            except asyncio.TimeoutError:
                if self._shutdown_event.is_set():
                    break
                continue
            except asyncio.CancelledError:
                break

            try:
                await self._deliver_message(message)
                async with self._lock:
                    self._messages_delivered += 1
            except Exception as exc:
                async with self._lock:
                    self._messages_failed += 1
                _structured_log(logging.ERROR, "bus.message.delivery.failed",
                                msg_id=message.message_id, error=str(exc))
            finally:
                self._message_queue.task_done()

    async def _deliver_message(self, message: BusMessage) -> None:
        """Route a message to the appropriate handler."""
        if message.message_type == MessageType.REPLY:
            fut = self._pending_replies.get(message.correlation_id)
            if fut and not fut.done():
                fut.set_result(message.payload)
            return

        if message.message_type == MessageType.BROADCAST:
            await self.publish(BusEvent(
                event_type="broadcast",
                data=message.payload,
                source=message.sender,
                trace_id=message.trace_id,
                metadata={"headers": message.headers},
            ))
            return

        # For DIRECT/REQUEST/FORWARD, publish as directed event
        recipient = message.recipient or "_all"
        await self.publish(BusEvent(
            event_type=f"message.{recipient}",
            data={"payload": message.payload, "sender": message.sender,
                  "message_type": message.message_type.value,
                  "correlation_id": message.correlation_id},
            source=message.sender,
            trace_id=message.trace_id,
        ))

    async def _process_tasks(self) -> None:
        """Worker loop processing queued tasks."""
        while self._running:
            try:
                _, _, task = await asyncio.wait_for(
                    self._task_queue.get(), timeout=1.0,
                )
            except asyncio.TimeoutError:
                if self._shutdown_event.is_set():
                    break
                continue
            except asyncio.CancelledError:
                break
            except Exception:
                continue

            try:
                task.status = TaskStatus.RUNNING
                retries = 0
                while retries <= task.max_retries:
                    try:
                        coro = task.coro
                        if coro is None:
                            raise ValueError(f"Task {task.task_id} has no coroutine")

                        result = await asyncio.wait_for(coro, timeout=task.timeout)
                        task.status = TaskStatus.COMPLETED
                        async with self._lock:
                            self._tasks_completed += 1
                        _structured_log(logging.DEBUG, "bus.task.completed",
                                        task_id=task.task_id, name=task.name)
                        await self.publish(BusEvent(
                            event_type="task.completed",
                            data={"task_id": task.task_id, "name": task.name, "result": str(result)[:500]},
                            source="bus",
                        ))
                        break

                    except asyncio.TimeoutError:
                        retries += 1
                        if retries > task.max_retries:
                            task.status = TaskStatus.TIMEOUT
                            async with self._lock:
                                self._tasks_timed_out += 1
                            _structured_log(logging.WARNING, "bus.task.timeout",
                                            task_id=task.task_id, name=task.name,
                                            timeout=task.timeout)
                            await self.publish(BusEvent(
                                event_type="task.timeout",
                                data={"task_id": task.task_id, "name": task.name},
                                source="bus",
                            ))
                            break
                        await asyncio.sleep(task.retry_delay)

                    except asyncio.CancelledError:
                        task.status = TaskStatus.CANCELLED
                        break

                    except Exception as exc:
                        retries += 1
                        if retries > task.max_retries:
                            task.status = TaskStatus.FAILED
                            async with self._lock:
                                self._tasks_failed += 1
                            _structured_log(logging.ERROR, "bus.task.failed",
                                            task_id=task.task_id, name=task.name, error=str(exc))
                            await self.publish(BusEvent(
                                event_type="task.failed",
                                data={"task_id": task.task_id, "name": task.name, "error": str(exc)},
                                source="bus",
                            ))
                            break
                        await asyncio.sleep(task.retry_delay)

            finally:
                self._task_queue.task_done()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_logger(self) -> logging.Logger:
        return logger

    async def wait_for_event(self, event_type: str, timeout: float = 30.0) -> BusEvent | None:
        """Wait asynchronously for the next occurrence of an event type."""
        fut: asyncio.Future[BusEvent] = asyncio.get_running_loop().create_future()

        async def _handler(event: BusEvent) -> None:
            if not fut.done():
                fut.set_result(event)

        sub = self.subscribe(event_type, _handler, subscriber_name="_wait_for_event")
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            self.unsubscribe(sub.subscriber_id)

    async def __aenter__(self) -> AgentBus:
        await self.start()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.shutdown()
