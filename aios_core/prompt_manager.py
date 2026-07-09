from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from string import Formatter
from typing import Any, Optional

logger = logging.getLogger("aios.core.prompt")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PromptRole(str, Enum):
    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"


class PromptType(str, Enum):
    TEMPLATE = "template"
    STATIC = "static"
    FEW_SHOT = "few_shot"
    COMPRESSED = "compressed"


class CompressionStrategy(str, Enum):
    TRUNCATE = "truncate"
    SUMMARIZE = "summarize"
    DEDUP = "dedup"
    STRIP_WHITESPACE = "strip_whitespace"
    EXTRACT_KEYWORDS = "extract_keywords"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FewShotExample:
    """A single few-shot example with input and expected output."""

    input: str
    output: str
    role: PromptRole = PromptRole.USER
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, str]:
        return {self.role.value: self.input, "assistant": self.output}

    def to_messages(self) -> list[dict[str, str]]:
        return [
            {"role": self.role.value, "content": self.input},
            {"role": PromptRole.ASSISTANT.value, "content": self.output},
        ]


@dataclass
class PromptVersion:
    """A versioned snapshot of a prompt template."""

    version: str
    content: str
    variables: list[str]
    created_at: str
    checksum: str
    author: str = ""
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "content": self.content,
            "variables": self.variables,
            "created_at": self.created_at,
            "checksum": self.checksum,
            "author": self.author,
            "description": self.description,
        }


@dataclass
class PromptStats:
    """Usage statistics for a prompt template."""

    uses: int = 0
    total_tokens: int = 0
    last_used: float = 0.0
    avg_tokens: float = 0.0

    def record(self, tokens: int) -> None:
        self.uses += 1
        self.total_tokens += tokens
        self.last_used = time.time()
        self.avg_tokens = self.total_tokens / self.uses


# ---------------------------------------------------------------------------
# Prompt Template
# ---------------------------------------------------------------------------

class PromptTemplate:
    """A versioned, renderable prompt template.

    Supports Python string formatting (``{variable}``), conditional blocks,
    and version tracking.
    """

    def __init__(
        self,
        name: str,
        content: str,
        role: PromptRole = PromptRole.SYSTEM,
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self._content = content
        self.role = role
        self.description = description
        self.metadata = metadata or {}
        self._versions: list[PromptVersion] = []
        self._stats = PromptStats()
        self._lock = threading.Lock()
        self._logger = logging.getLogger(f"aios.prompt.template.{name}")
        self._variables = self._extract_variables(content)
        self._save_version("0.1.0", content)

    @property
    def content(self) -> str:
        return self._content

    @content.setter
    def content(self, value: str) -> None:
        with self._lock:
            old = self._content
            self._content = value
            self._variables = self._extract_variables(value)
            version = self._next_version()
            self._save_version(version, value)
            self._logger.info("Template '%s' updated: %s -> %s", self.name, version, self._checksum(value))

    @property
    def variables(self) -> list[str]:
        return list(self._variables)

    @property
    def versions(self) -> list[PromptVersion]:
        return list(self._versions)

    @property
    def latest_version(self) -> str:
        return self._versions[-1].version if self._versions else "0.0.0"

    @property
    def stats(self) -> PromptStats:
        return self._stats

    def render(self, **kwargs: Any) -> str:
        """Render the template with the given variables."""
        missing = set(self._variables) - set(kwargs.keys())
        if missing:
            self._logger.warning("Missing variables in '%s': %s", self.name, missing)

        result = self._content
        for key, value in kwargs.items():
            placeholder = "{" + key + "}"
            if placeholder in result:
                result = result.replace(placeholder, str(value))

        token_count = len(result) // 2
        self._stats.record(token_count)

        return result

    def render_conditional(self, condition_vars: dict[str, bool], **kwargs: Any) -> str:
        """Render with conditional block support.

        Conditional syntax::

            {#if show_detail}
            Detailed content here...
            {#endif}
        """
        result = self._content

        for var, enabled in condition_vars.items():
            pattern = r"\{#if " + re.escape(var) + r"\}(.*?)\{#endif\}"
            if enabled:
                result = re.sub(pattern, r"\1", result, flags=re.DOTALL)
            else:
                result = re.sub(pattern, "", result, flags=re.DOTALL)

        for key, value in kwargs.items():
            placeholder = "{" + key + "}"
            if placeholder in result:
                result = result.replace(placeholder, str(value))

        token_count = len(result) // 2
        self._stats.record(token_count)

        return result

    def get_version(self, version: str) -> PromptVersion | None:
        for v in self._versions:
            if v.version == version:
                return v
        return None

    def rollback(self, version: str) -> bool:
        """Rollback to a previous version."""
        target = self.get_version(version)
        if target is None:
            self._logger.warning("Version '%s' not found for '%s'", version, self.name)
            return False
        self._content = target.content
        self._variables = self._extract_variables(target.content)
        self._logger.info("Template '%s' rolled back to version %s", self.name, version)
        return True

    def diff(self, version_a: str, version_b: str) -> str:
        """Return a simple diff between two versions."""
        va = self.get_version(version_a)
        vb = self.get_version(version_b)
        if va is None or vb is None:
            return "Version not found"

        lines_a = va.content.splitlines(keepends=True)
        lines_b = vb.content.splitlines(keepends=True)

        diff: list[str] = []
        import difflib

        for line in difflib.unified_diff(lines_a, lines_b, fromfile=version_a, tofile=version_b):
            diff.append(line.rstrip())
        return "\n".join(diff)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role.value,
            "content": self._content,
            "variables": self._variables,
            "version": self.latest_version,
            "versions": len(self._versions),
            "uses": self._stats.uses,
            "description": self.description,
        }

    def _extract_variables(self, text: str) -> list[str]:
        variables: list[str] = []
        for _, field_name, _, _ in Formatter().parse(text):
            if field_name is not None and not field_name.startswith("#"):
                variables.append(field_name)
        return variables

    def _save_version(self, version: str, content: str) -> None:
        self._versions.append(PromptVersion(
            version=version,
            content=content,
            variables=self._variables,
            created_at=datetime.now(timezone.utc).isoformat(),
            checksum=self._checksum(content),
            metadata={"author": self.metadata.get("author", "")},
        ))

    def _next_version(self) -> str:
        if not self._versions:
            return "0.1.0"
        parts = self._versions[-1].version.split(".")
        try:
            return f"{parts[0]}.{parts[1]}.{int(parts[2]) + 1}"
        except (IndexError, ValueError):
            return f"{time.time():.0f}"

    @staticmethod
    def _checksum(content: str) -> str:
        return hashlib.md5(content.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Prompt Compressor
# ---------------------------------------------------------------------------

class PromptCompressor:
    """Compress prompts using multiple strategies.

    Strategies:

    * ``truncate`` — Truncate to max_chars, preserving end if ``preserve_end`` is set.
    * ``strip_whitespace`` — Remove extra whitespace and blank lines.
    * ``dedup`` — Remove duplicate consecutive lines.
    * ``extract_keywords`` — Keep only sentences containing key terms.
    """

    def __init__(self, default_strategy: CompressionStrategy = CompressionStrategy.STRIP_WHITESPACE) -> None:
        self._default_strategy = default_strategy
        self._logger = logging.getLogger("aios.prompt.compressor")

    def compress(
        self,
        text: str,
        strategy: CompressionStrategy | None = None,
        max_chars: int = 4096,
        preserve_end: bool = True,
        keywords: list[str] | None = None,
    ) -> str:
        """Compress a prompt string and return the compressed version."""
        strategy = strategy or self._default_strategy
        original_len = len(text)

        if strategy == CompressionStrategy.STRIP_WHITESPACE:
            text = self._strip_whitespace(text)
        elif strategy == CompressionStrategy.TRUNCATE:
            text = self._truncate(text, max_chars, preserve_end)
        elif strategy == CompressionStrategy.DEDUP:
            text = self._dedup_lines(text)
        elif strategy == CompressionStrategy.EXTRACT_KEYWORDS:
            text = self._extract_keyword_sentences(text, keywords or [])

        if len(text) > max_chars and strategy != CompressionStrategy.TRUNCATE:
            text = self._truncate(text, max_chars, preserve_end)

        compressed_len = len(text)
        ratio = (1 - compressed_len / max(original_len, 1)) * 100
        self._logger.debug("Compressed %s -> %s chars (%.1f%% reduction)", original_len, compressed_len, ratio)
        return text

    def compress_messages(
        self,
        messages: list[dict[str, Any]],
        strategy: CompressionStrategy | None = None,
        max_chars: int = 4096,
    ) -> list[dict[str, Any]]:
        """Compress message contents in a list of chat messages."""
        strategy = strategy or self._default_strategy
        result: list[dict[str, Any]] = []
        total = 0

        for msg in messages:
            content = msg.get("content", "")
            compressed = self.compress(content, strategy, max_chars // max(len(messages), 1))
            total += len(compressed)
            result.append({**msg, "content": compressed})

        if total > max_chars:
            result = self._truncate_messages(result, max_chars)

        return result

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimate for a text."""
        return len(text) // 2

    @staticmethod
    def _strip_whitespace(text: str) -> str:
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r"\t", " ", text)
        return text.strip()

    @staticmethod
    def _truncate(text: str, max_chars: int, preserve_end: bool) -> str:
        if len(text) <= max_chars:
            return text
        if preserve_end:
            keep = max_chars // 2
            return text[:keep] + "\n...[truncated]...\n" + text[-(max_chars - keep - 18):]
        return text[:max_chars] + "\n...[truncated]..."

    @staticmethod
    def _dedup_lines(text: str) -> str:
        lines = text.splitlines(keepends=True)
        seen: set[str] = set()
        result: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped and stripped not in seen:
                seen.add(stripped)
                result.append(line)
        return "".join(result)

    @staticmethod
    def _extract_keyword_sentences(text: str, keywords: list[str]) -> str:
        if not keywords:
            return text
        sentences = re.split(r"(?<=[.!?])\s+", text)
        kept: list[str] = []
        for sentence in sentences:
            if any(kw.lower() in sentence.lower() for kw in keywords):
                kept.append(sentence)
        return " ".join(kept) if kept else text[:500]

    @staticmethod
    def _truncate_messages(messages: list[dict[str, Any]], max_chars: int) -> list[dict[str, Any]]:
        total = sum(len(m.get("content", "")) for m in messages)
        if total <= max_chars:
            return messages

        result = list(messages)
        while result and total > max_chars:
            removed = result.pop(0)
            total -= len(removed.get("content", ""))
        return result


# ---------------------------------------------------------------------------
# Prompt Manager
# ---------------------------------------------------------------------------

class PromptManager:
    """Central prompt management system.

    Manages prompt templates, few-shot examples, roles, compression,
    versioning, and context injection.

    Usage::

        from aios_core.prompt_manager import PromptManager, PromptTemplate, PromptRole

        pm = PromptManager()

        # Register a template
        pm.register_template(
            "qa",
            "Answer the following question concisely.\\n\\n{question}",
            role=PromptRole.USER,
        )

        # Render
        text = pm.render("qa", question="What is AI?")
        print(text)

        # Add few-shot examples
        pm.add_few_shot("qa", FewShotExample(input="What is 2+2?", output="4"))

        # Build a full chat prompt
        messages = pm.build_chat_prompt(
            system_prompt="You are a helpful assistant.",
            user_input="Explain quantum computing.",
            template_name="qa",
            context="Relevant context here...",
        )
    """

    def __init__(self, storage_path: str | Path | None = None) -> None:
        self._templates: dict[str, PromptTemplate] = {}
        self._few_shot: dict[str, list[FewShotExample]] = {}
        self._roles: dict[str, str] = {}
        self._compressor = PromptCompressor()
        self._lock = threading.Lock()
        self._logger = logging.getLogger("aios.core.prompt.manager")
        self._storage_path = Path(storage_path) if storage_path else None
        self._load_persisted()

    # -- Template Management ------------------------------------------------

    def register_template(
        self,
        name: str,
        content: str,
        role: PromptRole = PromptRole.SYSTEM,
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> PromptTemplate:
        """Register a new prompt template."""
        with self._lock:
            if name in self._templates:
                self._logger.warning("Overwriting existing template: '%s'", name)
            template = PromptTemplate(name, content, role, description, metadata)
            self._templates[name] = template
            self._logger.info("Registered template: '%s' (role=%s, vars=%s)", name, role.value, template.variables)
            self._persist()
            return template

    def get_template(self, name: str) -> PromptTemplate | None:
        return self._templates.get(name)

    def update_template(self, name: str, content: str, author: str = "") -> bool:
        """Update an existing template and create a new version."""
        template = self._templates.get(name)
        if template is None:
            self._logger.warning("Template not found: '%s'", name)
            return False
        template.metadata["author"] = author
        template.content = content
        self._persist()
        return True

    def delete_template(self, name: str) -> bool:
        with self._lock:
            if name not in self._templates:
                return False
            del self._templates[name]
            self._few_shot.pop(name, None)
            self._logger.info("Deleted template: '%s'", name)
            self._persist()
            return True

    def list_templates(self) -> list[dict[str, Any]]:
        return [t.to_dict() for t in self._templates.values()]

    def render(
        self,
        template_name: str,
        **kwargs: Any,
    ) -> str:
        """Render a template with variable substitution."""
        template = self._templates.get(template_name)
        if template is None:
            raise ValueError(f"Template not found: '{template_name}'")
        return template.render(**kwargs)

    def render_with_few_shot(
        self,
        template_name: str,
        n_examples: int = 3,
        **kwargs: Any,
    ) -> str:
        """Render a template with few-shot examples prepended."""
        template = self._templates.get(template_name)
        if template is None:
            raise ValueError(f"Template not found: '{template_name}'")

        rendered = template.render(**kwargs)
        examples = self._few_shot.get(template_name, [])
        selected = examples[:n_examples]

        if selected:
            example_parts = ["\n\nExamples:"]
            for i, ex in enumerate(selected, 1):
                example_parts.append(f"\n  {i}. Input: {ex.input}")
                example_parts.append(f"     Output: {ex.output}")
            rendered = "".join(example_parts) + "\n\n" + rendered

        return rendered

    # -- Few-Shot Management ------------------------------------------------

    def add_few_shot(
        self,
        template_name: str,
        example: FewShotExample,
    ) -> None:
        with self._lock:
            if template_name not in self._few_shot:
                self._few_shot[template_name] = []
            self._few_shot[template_name].append(example)
            self._persist()

    def add_few_shot_batch(
        self,
        template_name: str,
        examples: list[FewShotExample],
    ) -> None:
        with self._lock:
            if template_name not in self._few_shot:
                self._few_shot[template_name] = []
            self._few_shot[template_name].extend(examples)
            self._persist()

    def get_few_shot(self, template_name: str) -> list[FewShotExample]:
        return list(self._few_shot.get(template_name, []))

    def clear_few_shot(self, template_name: str) -> None:
        with self._lock:
            self._few_shot[template_name] = []
            self._persist()

    # -- Role Management ----------------------------------------------------

    def set_role_prompt(self, role: str, prompt: str) -> None:
        with self._lock:
            self._roles[role] = prompt
            self._persist()
            self._logger.info("Set prompt for role '%s'", role)

    def get_role_prompt(self, role: str) -> str:
        return self._roles.get(role, "")

    def list_roles(self) -> list[str]:
        return list(self._roles.keys())

    def delete_role(self, role: str) -> bool:
        with self._lock:
            return self._roles.pop(role, None) is not None

    # -- Context Injection --------------------------------------------------

    def inject_context(
        self,
        template: str | PromptTemplate,
        context: str,
        position: str = "before",
        marker: str = "{context}",
    ) -> str:
        """Inject context into a template at a given position.

        Args:
            template: Template content or PromptTemplate instance.
            context: Context string to inject.
            position: ``"before"``, ``"after"``, or ``"replace"`` relative to marker.
            marker: Placeholder string to inject at.

        Returns:
            The modified template string.
        """
        content = template.content if isinstance(template, PromptTemplate) else template

        if marker in content:
            if position == "replace":
                return content.replace(marker, context)
            prefix = content[:content.index(marker)]
            suffix = content[content.index(marker) + len(marker):]
            if position == "before":
                return f"{context}\n\n{content}"
            return f"{content}\n\n{context}"
        else:
            if position == "before":
                return f"{context}\n\n{content}"
            return f"{content}\n\n{context}"

    # -- Compression --------------------------------------------------------

    def compress(
        self,
        text: str,
        strategy: CompressionStrategy = CompressionStrategy.STRIP_WHITESPACE,
        max_chars: int = 4096,
        **kwargs: Any,
    ) -> str:
        return self._compressor.compress(text, strategy, max_chars, **kwargs)

    def compress_messages(
        self,
        messages: list[dict[str, Any]],
        strategy: CompressionStrategy = CompressionStrategy.STRIP_WHITESPACE,
        max_chars: int = 4096,
    ) -> list[dict[str, Any]]:
        return self._compressor.compress_messages(messages, strategy, max_chars)

    # -- Builders -----------------------------------------------------------

    def build_chat_prompt(
        self,
        system_prompt: str = "",
        user_input: str = "",
        template_name: str = "",
        context: str = "",
        few_shot: bool = True,
        n_examples: int = 3,
        developer_prompt: str = "",
    ) -> list[dict[str, Any]]:
        """Build a complete chat message list ready for an LLM API."""
        messages: list[dict[str, Any]] = []

        if developer_prompt:
            messages.append({"role": PromptRole.DEVELOPER.value, "content": developer_prompt})

        if template_name:
            template = self._templates.get(template_name)
            if template:
                render_kwargs: dict[str, Any] = {}
                if context:
                    render_kwargs["context"] = context
                if user_input:
                    render_kwargs["user_input"] = user_input

                if few_shot:
                    examples = self._few_shot.get(template_name, [])
                    selected = examples[:n_examples]
                    if selected:
                        example_parts = ["\n\nExamples:"]
                        for i, ex in enumerate(selected, 1):
                            example_parts.append(f"\n  {i}. Input: {ex.input}")
                            example_parts.append(f"     Output: {ex.output}")
                        rendered = template.render(**render_kwargs)
                        rendered = "".join(example_parts) + "\n\n" + rendered
                        messages.append({"role": template.role.value, "content": rendered})
                        return messages

                if system_prompt:
                    messages.append({"role": PromptRole.SYSTEM.value, "content": system_prompt})
                messages.append({"role": template.role.value, "content": template.render(**render_kwargs)})
                return messages

        if system_prompt:
            messages.append({"role": PromptRole.SYSTEM.value, "content": system_prompt})

        if context:
            content = f"{context}\n\n{user_input}" if user_input else context
            messages.append({"role": PromptRole.USER.value, "content": content})
        elif user_input:
            messages.append({"role": PromptRole.USER.value, "content": user_input})

        return messages

    def build_messages_with_history(
        self,
        history: list[dict[str, Any]],
        user_input: str,
        system_prompt: str = "",
        context: str = "",
        max_messages: int = 20,
    ) -> list[dict[str, Any]]:
        """Build messages from history with system prompt and context."""
        messages: list[dict[str, Any]] = []

        if system_prompt:
            messages.append({"role": PromptRole.SYSTEM.value, "content": system_prompt})

        if context:
            messages.append({"role": PromptRole.SYSTEM.value, "content": f"Context:\n{context}"})

        for msg in history[-max_messages:]:
            messages.append(msg)

        messages.append({"role": PromptRole.USER.value, "content": user_input})
        return messages

    # -- Utility ------------------------------------------------------------

    def estimate_tokens(self, text: str) -> int:
        return self._compressor.estimate_tokens(text)

    def get_stats(self, template_name: str) -> PromptStats | None:
        template = self._templates.get(template_name)
        return template.stats if template else None

    def get_all_stats(self) -> dict[str, Any]:
        return {
            name: {
                "uses": t.stats.uses,
                "total_tokens": t.stats.total_tokens,
                "avg_tokens": round(t.stats.avg_tokens, 1),
                "last_used": t.stats.last_used,
                "version": t.latest_version,
                "versions": len(t.versions),
            }
            for name, t in self._templates.items()
        }

    # -- Persistence --------------------------------------------------------

    def _persist(self) -> None:
        if self._storage_path is None:
            return
        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "templates": {n: t.to_dict() for n, t in self._templates.items()},
                "few_shot": {
                    n: [{"input": e.input, "output": e.output, "role": e.role.value} for e in examples]
                    for n, examples in self._few_shot.items()
                },
                "roles": self._roles,
            }
            self._storage_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            self._logger.warning("Failed to persist prompts: %s", exc)

    def _load_persisted(self) -> None:
        if self._storage_path is None or not self._storage_path.exists():
            return
        try:
            data = json.loads(self._storage_path.read_text(encoding="utf-8"))
            for name, tdata in data.get("templates", {}).items():
                template = PromptTemplate(
                    name=name,
                    content=tdata["content"],
                    role=PromptRole(tdata.get("role", "system")),
                    description=tdata.get("description", ""),
                )
                self._templates[name] = template
            for name, examples in data.get("few_shot", {}).items():
                self._few_shot[name] = [
                    FewShotExample(input=e["input"], output=e["output"], role=PromptRole(e.get("role", "user")))
                    for e in examples
                ]
            self._roles = data.get("roles", {})
            self._logger.info("Loaded %s templates, %s roles from %s", len(self._templates), len(self._roles), self._storage_path)
        except Exception as exc:
            self._logger.warning("Failed to load persisted prompts: %s", exc)
