"""
AIOS Coding Agent
=================

Production-grade code generation, analysis, review, refactoring, bug detection,
test suggestion, and documentation agent. Operates on local files with AST-based
static analysis for Python and regex-based heuristics for other languages.

Actions (auto-dispatched from task description):

- ``analyze`` / ``read`` — Read and parse a source file into a structured model
- ``generate`` / ``write`` / ``implement`` — Generate code from a description
- ``explain`` — Produce a line-by-line explanation of source code
- ``review`` / ``audit`` — Static analysis for bugs, security, style, performance
- ``refactor`` — Apply pattern-based code transformations
- ``test`` / ``tests`` — Generate unit-test suggestions for a code unit
- ``document`` / ``docs`` — Generate docstrings, README, or API docs
"""

from __future__ import annotations

import ast
import logging
import os
import re
import textwrap
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from agents.base_agent import AgentConfig, AgentResult, BaseAgent

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("aios.agent.coding")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FunctionInfo:
    """Metadata about a function or method."""

    name: str = ""
    params: list[str] = field(default_factory=list)
    return_type: str = ""
    line: int = 0
    docstring: str = ""
    complexity: int = 1
    is_async: bool = False
    decorators: list[str] = field(default_factory=list)
    body_lines: int = 0


@dataclass
class ClassInfo:
    """Metadata about a class."""

    name: str = ""
    bases: list[str] = field(default_factory=list)
    methods: list[FunctionInfo] = field(default_factory=list)
    line: int = 0
    docstring: str = ""
    decorators: list[str] = field(default_factory=list)


@dataclass
class ImportInfo:
    """Metadata about an import statement."""

    module: str = ""
    names: list[str] = field(default_factory=list)
    alias: str = ""
    line: int = 0


@dataclass
class CodeIssue:
    """
    A single issue found during code review.

    Attributes:
        issue_type: Category — ``bug``, ``security``, ``style``, ``performance``,
            ``correctness``, ``maintainability``.
        severity: ``critical``, ``high``, ``medium``, ``low``, ``info``.
        line: Source line number (0 = file-level).
        message: Human-readable description.
        suggestion: Actionable fix suggestion.
        code: Relevant code snippet.
    """

    issue_type: str = ""
    severity: str = "info"
    line: int = 0
    message: str = ""
    suggestion: str = ""
    code: str = ""


@dataclass
class CodeBlock:
    """A contiguous code block with metadata."""

    start_line: int = 0
    end_line: int = 0
    content: str = ""
    block_type: str = ""  # function, class, loop, conditional, etc.
    name: str = ""


@dataclass
class RefactoringSuggestion:
    """A suggested code refactoring."""

    ref_type: str = ""  # extract_function, rename, inline, simplify, etc.
    description: str = ""
    target_line: int = 0
    target_code: str = ""
    suggested_code: str = ""
    motivation: str = ""


@dataclass
class TestSuggestion:
    """A suggested test case."""

    name: str = ""
    target: str = ""  # function or class name
    test_type: str = "unit"  # unit, integration, property
    description: str = ""
    code: str = ""
    coverage_note: str = ""


@dataclass
class DocumentationBlock:
    """Auto-generated documentation for a code unit."""

    target: str = ""  # module, class, function
    target_name: str = ""
    docstring: str = ""
    signature: str = ""
    params: list[tuple[str, str]] = field(default_factory=list)
    returns: str = ""
    raises: list[str] = field(default_factory=list)
    example: str = ""


@dataclass
class CodeAnalysis:
    """Complete structured analysis of a source file."""

    file_path: str = ""
    language: str = ""
    lines: int = 0
    size_bytes: int = 0
    functions: list[FunctionInfo] = field(default_factory=list)
    classes: list[ClassInfo] = field(default_factory=list)
    imports: list[ImportInfo] = field(default_factory=list)
    issues: list[CodeIssue] = field(default_factory=list)
    complexity: int = 0
    todo_count: int = 0
    dependencies: list[str] = field(default_factory=list)
    raw_content: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "language": self.language,
            "lines": self.lines,
            "size_bytes": self.size_bytes,
            "functions": [
                {"name": f.name, "params": f.params, "return_type": f.return_type,
                 "line": f.line, "complexity": f.complexity, "body_lines": f.body_lines}
                for f in self.functions
            ],
            "classes": [
                {"name": c.name, "bases": c.bases, "methods": [m.name for m in c.methods],
                 "line": c.line}
                for c in self.classes
            ],
            "imports": [{"module": i.module, "names": i.names} for i in self.imports],
            "complexity": self.complexity,
            "todo_count": self.todo_count,
            "dependencies": self.dependencies,
        }


# ---------------------------------------------------------------------------
# Language detection & helpers
# ---------------------------------------------------------------------------

_LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".cc": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".r": "r",
}


def _detect_language_from_path(path: str) -> str:
    return _LANGUAGE_MAP.get(Path(path).suffix.lower(), "unknown")


def _detect_language_from_task(task: str) -> str:
    tl = task.lower()
    if "python" in tl:
        return "python"
    if "javascript" in tl or "typescript" in tl or "js" in tl:
        return "typescript"
    if "rust" in tl:
        return "rust"
    if "go" in tl or "golang" in tl:
        return "go"
    if "java" in tl:
        return "java"
    if "c++" in tl or "cpp" in tl:
        return "cpp"
    if "c#" in tl or "csharp" in tl:
        return "csharp"
    return "python"


def _safe_read(path: str, root: str | None = None) -> str:
    p = Path(path).resolve()
    if root:
        root_p = Path(root).resolve()
        if not str(p).startswith(str(root_p)):
            raise PermissionError(f"Access denied: {path} is outside allowed root {root}")
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return p.read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Python AST analysis
# ---------------------------------------------------------------------------


def _run_text_security_checks(source: str) -> list[CodeIssue]:
    """Run security checks on source text that don't require a parseable AST."""
    issues: list[CodeIssue] = []
    source_lower = source.lower()
    if "password" in source_lower or "secret" in source_lower or "api_key" in source_lower:
        issues.append(CodeIssue(
            issue_type="security", severity="high", line=0,
            message="Potential secret or credential in source code.",
            suggestion="Move secrets to environment variables or a vault.",
        ))
    if "eval(" in source or "exec(" in source:
        issues.append(CodeIssue(
            issue_type="security", severity="critical", line=0,
            message="Use of eval() or exec() detected.",
            suggestion="Avoid dynamic code execution. Use ast.literal_eval() if parsing literals.",
        ))
    if "import *" in source:
        issues.append(CodeIssue(
            issue_type="style", severity="medium", line=0,
            message="Wildcard import 'import *' detected.",
            suggestion="Import only the names you need for clarity and to avoid namespace pollution.",
        ))
    return issues


def _analyze_python_ast(source: str, file_path: str = "") -> CodeAnalysis:
    """Full Python AST analysis returning a structured CodeAnalysis."""
    analysis = CodeAnalysis(
        file_path=file_path,
        language="python",
        lines=len(source.splitlines()),
        size_bytes=len(source.encode("utf-8")),
        raw_content=source,
    )

    analysis.issues.extend(_run_text_security_checks(source))

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        analysis.issues.append(CodeIssue(
            issue_type="bug", severity="critical", line=exc.lineno or 0,
            message=f"Syntax error: {exc.msg}",
            suggestion="Fix the syntax error before proceeding.",
        ))
        return analysis

    # Walk the tree
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            info = _extract_function_info(node, source)
            analysis.functions.append(info)
            analysis.complexity += info.complexity
        elif isinstance(node, ast.AsyncFunctionDef):
            info = _extract_function_info(node, source, is_async=True)
            analysis.functions.append(info)
            analysis.complexity += info.complexity
        elif isinstance(node, ast.ClassDef):
            cls_info = _extract_class_info(node, source)
            analysis.classes.append(cls_info)
            for m in cls_info.methods:
                analysis.complexity += m.complexity
        elif isinstance(node, ast.Import):
            for alias in node.names:
                analysis.imports.append(ImportInfo(
                    module=alias.name or "",
                    names=[alias.asname or alias.name] if alias.asname else [alias.name],
                    line=node.lineno or 0,
                ))
        elif isinstance(node, ast.ImportFrom):
            names = [alias.name for alias in node.names]
            analysis.imports.append(ImportInfo(
                module=node.module or "",
                names=names,
                line=node.lineno or 0,
            ))

    analysis.todo_count = source.lower().count("todo") + source.lower().count("fixme")
    analysis.dependencies = _extract_dependencies_python(analysis.imports)
    analysis.issues.extend(_run_python_lint_rules(tree, source))
    return analysis


def _extract_function_info(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    source: str,
    is_async: bool = False,
) -> FunctionInfo:
    params: list[str] = []
    for arg in node.args.args:
        params.append(arg.arg)
    if node.args.vararg:
        params.append(f"*{node.args.vararg.arg}")
    if node.args.kwonlyargs:
        for arg in node.args.kwonlyargs:
            params.append(arg.arg)
    if node.args.kwarg:
        params.append(f"**{node.args.kwarg.arg}")

    return_type = ""
    if node.returns:
        try:
            return_type = ast.unparse(node.returns)
        except Exception:
            return_type = "?"

    docstring = ast.get_docstring(node) or ""
    decorators = []
    for dec in node.decorator_list:
        try:
            decorators.append(ast.unparse(dec))
        except Exception:
            decorators.append("<expr>")

    complexity = _compute_cyclomatic_complexity(node)
    body_lines = 0
    if node.body:
        last = node.body[-1]
        first = node.body[0]
        body_lines = (last.end_lineno or last.lineno or 0) - (first.lineno or 0) + 1

    return FunctionInfo(
        name=node.name,
        params=params,
        return_type=return_type,
        line=node.lineno or 0,
        docstring=docstring,
        complexity=complexity,
        is_async=is_async,
        decorators=decorators,
        body_lines=body_lines,
    )


def _extract_class_info(node: ast.ClassDef, source: str) -> ClassInfo:
    bases = []
    for base in node.bases:
        try:
            bases.append(ast.unparse(base))
        except Exception:
            bases.append("<expr>")

    docstring = ast.get_docstring(node) or ""
    decorators = []
    for dec in node.decorator_list:
        try:
            decorators.append(ast.unparse(dec))
        except Exception:
            decorators.append("<expr>")

    methods: list[FunctionInfo] = []
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods.append(_extract_function_info(
                item, source, is_async=isinstance(item, ast.AsyncFunctionDef),
            ))

    return ClassInfo(
        name=node.name,
        bases=bases,
        methods=methods,
        line=node.lineno or 0,
        docstring=docstring,
        decorators=decorators,
    )


def _compute_cyclomatic_complexity(node: ast.AST) -> int:
    """McCabe cyclomatic complexity for a function/class body."""
    complexity = 1
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
            complexity += 1
        elif isinstance(child, ast.Try):
            complexity += len(child.handlers)
        elif isinstance(child, (ast.ExceptHandler, ast.With, ast.AsyncWith)):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            complexity += len(child.values) - 1
    return complexity


def _extract_dependencies_python(imports: list[ImportInfo]) -> list[str]:
    """Extract third-party dependency names from imports."""
    stdlib = {
        "os", "sys", "json", "re", "math", "datetime", "collections",
        "typing", "pathlib", "itertools", "functools", "hashlib",
        "base64", "uuid", "abc", "enum", "dataclasses", "logging",
        "argparse", "configparser", "csv", "io", "copy", "random",
        "statistics", "string", "textwrap", "threading", "asyncio",
        "inspect", "pprint", "tempfile", "time", "warnings", "weakref",
        "contextlib", "importlib", "ast", "dis", "gc", "pickle",
        "shelve", "sqlite3", "xml", "html", "http", "urllib",
        "email", "json", "unittest", "subprocess", "signal",
    }
    deps: set[str] = set()
    for imp in imports:
        parts = imp.module.split(".") if imp.module else []
        top = parts[0] if parts else ""
        if top and top not in stdlib and not top.startswith("_"):
            deps.add(top)
        for name in imp.names:
            if name and name not in stdlib and not name.startswith("_"):
                deps.add(name)
    return sorted(deps, key=str.lower)


# ---------------------------------------------------------------------------
# Python lint rules
# ---------------------------------------------------------------------------


def _run_python_lint_rules(tree: ast.AST, source: str) -> list[CodeIssue]:
    """Run a suite of static-analysis lint rules on a Python AST."""
    issues: list[CodeIssue] = []
    lines = source.splitlines()

    for node in ast.walk(tree):
        # Bare except
        if isinstance(node, ast.ExceptHandler):
            if node.type is None:
                issues.append(CodeIssue(
                    issue_type="bug", severity="high",
                    line=node.lineno or 0,
                    message="Bare except clause catches all exceptions.",
                    suggestion="Catch a specific exception type instead of using bare 'except:'.",
                    code=_get_line_snippet(lines, node.lineno or 0),
                ))

        # Empty except / pass
        if isinstance(node, ast.ExceptHandler):
            if (len(node.body) == 1 and isinstance(node.body[0], ast.Pass)):
                issues.append(CodeIssue(
                    issue_type="bug", severity="medium",
                    line=node.lineno or 0,
                    message="Empty except block silently swallows exceptions.",
                    suggestion="Either log the exception, re-raise it, or handle it explicitly.",
                    code=_get_line_snippet(lines, node.lineno or 0),
                ))

        # Function too long
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lines_count = 0
            if node.body:
                last = node.body[-1]
                first = node.body[0]
                lines_count = (last.end_lineno or last.lineno or 0) - (first.lineno or 0) + 1
            if lines_count > 50:
                issues.append(CodeIssue(
                    issue_type="maintainability", severity="medium",
                    line=node.lineno or 0,
                    message=f"Function '{node.name}' has {lines_count} lines.",
                    suggestion="Consider breaking it into smaller helper functions.",
                ))

        # Too many arguments
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            arg_count = len(node.args.args) + len(node.args.kwonlyargs)
            if arg_count > 6:
                issues.append(CodeIssue(
                    issue_type="maintainability", severity="low",
                    line=node.lineno or 0,
                    message=f"Function '{node.name}' has {arg_count} parameters.",
                    suggestion="Consider using a dataclass or **kwargs to reduce argument count.",
                ))

        # Deep nesting
        if isinstance(node, ast.If):
            nesting = _count_nesting_depth(node, 0)
            if nesting > 4:
                issues.append(CodeIssue(
                    issue_type="style", severity="low",
                    line=node.lineno or 0,
                    message=f"Deep nesting detected (depth {nesting}).",
                    suggestion="Extract inner blocks into separate functions or use early returns.",
                ))

    # File-level checks
    if "# noqa" not in source:
        no_docstring_count = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not ast.get_docstring(node):
                    no_docstring_count += 1
        if no_docstring_count > len(list(ast.walk(tree))) * 0.3:
            issues.append(CodeIssue(
                issue_type="style", severity="low", line=0,
                message=f"{no_docstring_count} functions/classes missing docstrings.",
                suggestion="Add docstrings following PEP 257 conventions.",
            ))

    return issues


def _count_nesting_depth(node: ast.AST, depth: int) -> int:
    max_depth = depth
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
            child_depth = _count_nesting_depth(child, depth + 1)
            if child_depth > max_depth:
                max_depth = child_depth
        else:
            child_depth = _count_nesting_depth(child, depth)
            if child_depth > max_depth:
                max_depth = child_depth
    return max_depth


def _get_line_snippet(lines: list[str], line_no: int, context: int = 2) -> str:
    start = max(0, line_no - 1 - context)
    end = min(len(lines), line_no + context)
    snippet = "\n".join(
        f"{i + 1:4d} | {lines[i]}"
        for i in range(start, end)
    )
    return snippet


# ---------------------------------------------------------------------------
# Generic (non-Python) analysis
# ---------------------------------------------------------------------------


def _analyze_generic(source: str, language: str, file_path: str = "") -> CodeAnalysis:
    """Regex-based analysis for non-Python languages."""
    analysis = CodeAnalysis(
        file_path=file_path,
        language=language,
        lines=len(source.splitlines()),
        size_bytes=len(source.encode("utf-8")),
        raw_content=source,
    )

    # Basic function detection
    func_patterns = {
        "javascript": r"(?:async\s+)?function\s+\*?\s*(\w+)\s*\(",
        "typescript": r"(?:async\s+)?function\s+\*?\s*(\w+)\s*\(|(\w+)\s*\([^)]*\)\s*:\s*\w+",
        "rust": r"fn\s+(\w+)\s*\(",
        "go": r"func\s+(?:\([^)]*\)\s+)?(\w+)\s*\(",
        "java": r"(?:public|private|protected|static)?\s*\w+\s+(\w+)\s*\(",
        "cpp": r"\w+\s+(\w+)\s*\([^)]*\)\s*(?:const|override|final|\{|;)",
    }
    pattern = func_patterns.get(language)
    if pattern:
        for m in re.finditer(pattern, source):
            line_no = source[:m.start()].count("\n") + 1
            analysis.functions.append(FunctionInfo(
                name=m.group(1) or m.group(2) or "unknown",
                line=line_no,
            ))

    # Class detection
    class_patterns = {
        "typescript": r"(?:export\s+)?(?:abstract\s+)?class\s+(\w+)",
        "java": r"(?:public|abstract|final)?\s*(?:class|interface)\s+(\w+)",
        "cpp": r"class\s+(\w+)",
        "rust": r"(?:struct|trait|impl)\s+(\w+)",
        "go": r"type\s+(\w+)\s+(?:struct|interface)",
    }
    cpattern = class_patterns.get(language)
    if cpattern:
        for m in re.finditer(cpattern, source):
            line_no = source[:m.start()].count("\n") + 1
            analysis.classes.append(ClassInfo(
                name=m.group(1),
                line=line_no,
            ))

    analysis.todo_count = source.lower().count("todo") + source.lower().count("fixme")
    return analysis


# ---------------------------------------------------------------------------
# Code generation templates
# ---------------------------------------------------------------------------


def _generate_python_code(description: str, name_hint: str = "main") -> str:
    """Generate a Python code skeleton with type hints, error handling, and a docstring."""
    name = _to_snake_case(name_hint)
    lines = [
        '"""',
        f"{description}",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "import logging",
        "from typing import Any",
        "",
        "logger = logging.getLogger(__name__)",
        "",
        "",
        f"def {name}(*args: Any, **kwargs: Any) -> Any:",
        f'    """{description}"""',
        "    try:",
        "        # TODO: implement",
        "        result = None",
        "        logger.debug(\"Executed %s with args=%s\", _name := {}, args)",
        "        return result",
        "    except Exception as exc:",
        '        logger.error("Error in %s: %s", "___name__", exc)',
        "        raise",
        "",
    ]
    return "\n".join(lines)


def _generate_typescript_code(description: str, name_hint: str = "main") -> str:
    name = _to_camel_case(name_hint)
    lines = [
        "/**",
        f" * {description}",
        " */",
        "",
        f"export async function {name}<T>(input: T): Promise<T> {{",
        "    try {",
        "        // TODO: implement",
        '        console.debug(`Executed ${_name} with input:`, input);',
        "        return input;",
        "    } catch (error) {",
        '        console.error(`Error in ${_name}:`, error);',
        "        throw error;",
        "    }",
        "}",
        "",
    ]
    return "\n".join(lines)


def _generate_rust_code(description: str, name_hint: str = "main") -> str:
    name = _to_snake_case(name_hint)
    lines = [
        "///",
        f"/// {description}",
        "///",
        "",
        f"pub fn {name}(input: &str) -> Result<String, Box<dyn std::error::Error>> {{",
        "    // TODO: implement",
        '    tracing::debug!("Executed {} with input: {}", stringify!(' + name + "), input);",
        "    Ok(input.to_string())",
        "}",
        "",
    ]
    return "\n".join(lines)


_GENERATORS: dict[str, Any] = {
    "python": _generate_python_code,
    "typescript": _generate_typescript_code,
    "javascript": _generate_typescript_code,
    "rust": _generate_rust_code,
}


def _to_snake_case(name: str) -> str:
    name = re.sub(r"([A-Z])", r"_\1", name).lower().strip("_")
    name = re.sub(r"[^a-z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name)
    return name or "main"


def _to_camel_case(name: str) -> str:
    parts = re.split(r"[^a-zA-Z0-9]", name)
    return parts[0].lower() + "".join(p.capitalize() for p in parts[1:]) if parts else "main"


# ---------------------------------------------------------------------------
# Documentation generation
# ---------------------------------------------------------------------------


def _generate_docstring_python(analysis: CodeAnalysis) -> list[DocumentationBlock]:
    """Generate missing docstrings for Python code based on AST analysis."""
    blocks: list[DocumentationBlock] = []

    # Module-level doc
    if not analysis.raw_content.strip().startswith('"""'):
        blocks.append(DocumentationBlock(
            target="module",
            target_name=Path(analysis.file_path).name if analysis.file_path else "module",
            docstring=f"Module for {Path(analysis.file_path).stem.replace('_', ' ')}.",
            params=[],
            returns="",
        ))

    for func in analysis.functions:
        if func.docstring:
            continue
        params_doc = [(p, "Description of ``{p}``.") for p in func.params if p != "self"]
        blocks.append(DocumentationBlock(
            target="function",
            target_name=func.name,
            docstring=f"{func.name} — TODO: add description.",
            signature=f"def {func.name}({', '.join(func.params)}) -> {func.return_type or 'Any'}",
            params=params_doc,
            returns=func.return_type or "Any",
            raises=["Exception"] if func.body_lines > 0 else [],
        ))

    for cls in analysis.classes:
        if cls.docstring:
            continue
        blocks.append(DocumentationBlock(
            target="class",
            target_name=cls.name,
            docstring=f"{cls.name} — TODO: add description.",
            params=[],
            returns="",
        ))

    return blocks


# ---------------------------------------------------------------------------
# Test generation
# ---------------------------------------------------------------------------


def _generate_python_tests(function: FunctionInfo, module_path: str) -> list[TestSuggestion]:
    """Generate pytest-style test suggestions for a function."""
    tests: list[TestSuggestion] = []
    module_name = Path(module_path).stem if module_path else "module"
    test_name = f"test_{function.name}"

    params_code = ", ".join(function.params[:3])

    # Basic unit test
    tests.append(TestSuggestion(
        name=test_name,
        target=function.name,
        test_type="unit",
        description=f"Basic unit test for {function.name}",
        code=(
            f"from {module_name} import {function.name}\n"
            f"import pytest\n\n\n"
            f"class Test{function.name.capitalize()}:\n"
            f"    def {test_name}_basic(self):\n"
            f'        """Test {function.name} with typical inputs."""\n'
            f"        result = {function.name}({params_code})\n"
            f"        assert result is not None\n\n"
            f"    def {test_name}_edge(self):\n"
            f'        """Test {function.name} with edge cases."""\n'
            f"        result = {function.name}({params_code})\n"
            f"        assert result is not None\n\n"
            f"    def {test_name}_error(self):\n"
            f'        """Test {function.name} error handling."""\n'
            f"        with pytest.raises(Exception):\n"
            f"            {function.name}({params_code})\n"
        ),
        coverage_note=f"Covers basic, edge, and error cases for {function.name}",
    ))

    return tests


# ---------------------------------------------------------------------------
# Refactoring suggestions
# ---------------------------------------------------------------------------


def _suggest_python_refactorings(analysis: CodeAnalysis) -> list[RefactoringSuggestion]:
    """Generate refactoring suggestions based on analysis."""
    suggestions: list[RefactoringSuggestion] = []

    for func in analysis.functions:
        if func.complexity > 10:
            suggestions.append(RefactoringSuggestion(
                ref_type="simplify",
                description=f"Function '{func.name}' has high cyclomatic complexity ({func.complexity}).",
                target_line=func.line,
                motivation="High complexity makes code hard to test and maintain.",
                suggested_code=f"# Consider splitting {func.name} into smaller functions",
            ))

        if func.body_lines > 40:
            suggestions.append(RefactoringSuggestion(
                ref_type="extract_function",
                description=f"Function '{func.name}' has {func.body_lines} lines.",
                target_line=func.line,
                motivation="Long functions violate the Single Responsibility Principle.",
                suggested_code=f"# Extract helper functions from {func.name}",
            ))

    param_counts = Counter(len(f.params) for f in analysis.functions)
    if param_counts:
        max_params = max(param_counts.keys())
        if max_params > 6:
            suggestions.append(RefactoringSuggestion(
                ref_type="introduce_parameter_object",
                description=f"Functions have up to {max_params} parameters.",
                target_line=0,
                motivation="Too many parameters reduce readability and increase bug risk.",
                suggested_code="# Consider grouping parameters into a dataclass or TypedDict",
            ))

    return suggestions


# ---------------------------------------------------------------------------
# Action dispatch
# ---------------------------------------------------------------------------


def _detect_action(task: str) -> str:
    """Map task description to a CodingAgent action."""
    tl = task.lower()
    if any(w in tl for w in ["analyze", "analyse", "read file", "parse", "inspect"]):
        return "analyze"
    if any(w in tl for w in ["document", "docs", "docstring", "readme", "write documentation"]):
        return "document"
    if any(w in tl for w in ["generate", "write", "implement", "create", "produce", "build"]):
        return "generate"
    if any(w in tl for w in ["explain", "describe", "summarize", "break down"]):
        return "explain"
    if any(w in tl for w in ["review", "audit", "inspect code", "check"]):
        return "review"
    if any(w in tl for w in ["refactor", "improve", "optimize", "restructure", "clean up"]):
        return "refactor"
    if any(w in tl for w in ["bug", "bugs", "defect", "error detection", "find bugs"]):
        return "detect_bugs"
    if any(w in tl for w in ["test", "tests", "unit test", "pytest"]):
        return "suggest_tests"
    return "generate"


# ---------------------------------------------------------------------------
# CodingAgent
# ---------------------------------------------------------------------------


class CodingAgent(BaseAgent):
    """
    Production-grade code agent supporting generation, analysis, review,
    refactoring, bug detection, test suggestion, and documentation.

    The action is auto-dispatched from the task description:

    +------------------+----------------------------------------------------+
    | Action           | Keywords                                           |
    +------------------+----------------------------------------------------+
    | ``analyze``      | analyze, read file, parse, inspect                  |
    | ``generate``     | generate, write, implement, create, build           |
    | ``explain``      | explain, describe, summarize, break down             |
    | ``review``       | review, audit, inspect code, check                   |
    | ``refactor``     | refactor, improve, optimize, restructure             |
    | ``detect_bugs``  | bug, bugs, defect, error detection, find bugs        |
    | ``suggest_tests``| test, tests, unit test, pytest                      |
    | ``document``     | document, docs, docstring, readme                    |
    +------------------+----------------------------------------------------+

    Provide ``context["file_path"]`` to specify the file to operate on.
    Provide ``context["root"]`` to restrict file access to a project root.
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        allowed_root: str | None = None,
    ) -> None:
        if config is None:
            config = AgentConfig(
                name="coding",
                system_prompt=(
                    "You are a senior software engineer. Generate clean, "
                    "production-ready code with type hints, error handling, "
                    "and documentation. Analyze and refactor code to improve "
                    "quality, performance, and maintainability."
                ),
            )
        super().__init__(config)
        self._allowed_root = allowed_root

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(
        self,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> AgentResult:
        """
        Execute a coding operation determined by the task description.

        Args:
            task: Description of what to do.
            context: Optional context. Supports keys:
                ``file_path`` — target file path,
                ``root`` — allowed project root,
                ``language`` — override language detection.

        Returns:
            AgentResult with ``output`` containing the result text and
            ``metadata`` with structured analysis data.
        """
        start = time.perf_counter()
        ctx = context or {}
        file_path = ctx.get("file_path", "")
        root = ctx.get("root", self._allowed_root)
        language = ctx.get("language", "")

        try:
            action = _detect_action(task)

            if action == "analyze":
                result = await self._action_analyze(task, file_path, root)
            elif action == "generate":
                lang = language or _detect_language_from_task(task)
                result = await self._action_generate(task, lang)
            elif action == "explain":
                result = await self._action_explain(task, file_path, root)
            elif action == "review":
                result = await self._action_review(task, file_path, root)
            elif action == "refactor":
                result = await self._action_refactor(task, file_path, root)
            elif action == "detect_bugs":
                result = await self._action_detect_bugs(task, file_path, root)
            elif action == "suggest_tests":
                result = await self._action_suggest_tests(task, file_path, root)
            elif action == "document":
                result = await self._action_document(task, file_path, root)
            else:
                lang = language or _detect_language_from_task(task)
                result = await self._action_generate(task, lang)

            duration = (time.perf_counter() - start) * 1000
            result.duration_ms = duration
            return result

        except FileNotFoundError as exc:
            duration = (time.perf_counter() - start) * 1000
            return AgentResult(
                success=False, output="", agent_name=self.name,
                duration_ms=duration, error=str(exc),
            )
        except PermissionError as exc:
            duration = (time.perf_counter() - start) * 1000
            return AgentResult(
                success=False, output="", agent_name=self.name,
                duration_ms=duration, error=str(exc),
            )
        except Exception as exc:
            duration = (time.perf_counter() - start) * 1000
            logger.error("CodingAgent error: %s", exc)
            return AgentResult(
                success=False, output="", agent_name=self.name,
                duration_ms=duration, error=str(exc),
            )

    # ------------------------------------------------------------------
    # Analyze
    # ------------------------------------------------------------------

    async def _action_analyze(
        self,
        task: str,
        file_path: str,
        root: str | None,
    ) -> AgentResult:
        if not file_path:
            return AgentResult(
                success=False, output="", agent_name=self.name,
                error="No file_path provided in context. Set context['file_path'] to a source file.",
            )

        source = _safe_read(file_path, root)
        lang = _detect_language_from_path(file_path)

        if lang == "python":
            analysis = _analyze_python_ast(source, file_path)
        else:
            analysis = _analyze_generic(source, lang, file_path)

        output_lines = [
            f"## Analysis: {file_path}",
            "",
            f"- Language: {analysis.language}",
            f"- Lines: {analysis.lines}",
            f"- Size: {analysis.size_bytes} bytes",
            f"- Functions: {len(analysis.functions)}",
            f"- Classes: {len(analysis.classes)}",
            f"- Imports: {len(analysis.imports)}",
            f"- Cyclomatic complexity: {analysis.complexity}",
            f"- TODOs/FIXMEs: {analysis.todo_count}",
            f"- Dependencies: {', '.join(analysis.dependencies) if analysis.dependencies else 'none'}",
            f"- Issues: {len(analysis.issues)}",
            "",
        ]

        if analysis.issues:
            output_lines.append("### Issues")
            for iss in analysis.issues:
                sev = iss.severity.upper()
                output_lines.append(f"- [{sev}] L{iss.line}: {iss.message}")
                if iss.suggestion:
                    output_lines.append(f"  Suggestion: {iss.suggestion}")

        if analysis.functions:
            output_lines.append("")
            output_lines.append("### Functions")
            for f in analysis.functions:
                output_lines.append(
                    f"- `{f.name}({', '.join(f.params)})` → {f.return_type or '?'}  "
                    f"L{f.line}  complexity={f.complexity}"
                )

        if analysis.classes:
            output_lines.append("")
            output_lines.append("### Classes")
            for c in analysis.classes:
                bases = f"({', '.join(c.bases)})" if c.bases else ""
                output_lines.append(f"- `{c.name}{bases}` L{c.line}")
                for m in c.methods:
                    output_lines.append(f"  - `{m.name}()` L{m.line}")

        output = "\n".join(output_lines)
        return AgentResult(
            success=True, output=output, agent_name=self.name,
            metadata={
                "action": "analyze",
                "analysis": analysis.to_dict(),
                "file_path": file_path,
                "language": analysis.language,
            },
        )

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    async def _action_generate(
        self,
        task: str,
        language: str,
    ) -> AgentResult:
        generator = _GENERATORS.get(language)
        if generator is None:
            code = (
                f"// Auto-generated code for: {task}\n"
                f"// Language: {language}\n\n"
                f"// TODO: implement\n"
            )
        else:
            name_hint = self._extract_name_hint(task)
            code = generator(task, name_hint)

        output = f"```{language}\n{code}\n```"
        return AgentResult(
            success=True, output=output, agent_name=self.name,
            metadata={
                "action": "generate",
                "language": language,
                "code": code,
            },
        )

    # ------------------------------------------------------------------
    # Explain
    # ------------------------------------------------------------------

    async def _action_explain(
        self,
        task: str,
        file_path: str,
        root: str | None,
    ) -> AgentResult:
        if not file_path:
            return AgentResult(
                success=False, output="", agent_name=self.name,
                error="No file_path provided. Set context['file_path'].",
            )

        source = _safe_read(file_path, root)
        lang = _detect_language_from_path(file_path)
        lines = source.splitlines()

        if lang == "python":
            analysis = _analyze_python_ast(source, file_path)
        else:
            analysis = _analyze_generic(source, lang, file_path)

        output_parts = [
            f"## Explanation: {file_path}",
            "",
            f"**Language:** {analysis.language}  |  "
            f"**Lines:** {analysis.lines}  |  "
            f"**Functions:** {len(analysis.functions)}  |  "
            f"**Classes:** {len(analysis.classes)}",
            "",
        ]

        if analysis.classes:
            output_parts.append("### Class Overview")
            for cls in analysis.classes:
                bases = f"({', '.join(cls.bases)})" if cls.bases else ""
                output_parts.append(f"- **`{cls.name}`{bases}** — L{cls.line}")
                if cls.docstring:
                    output_parts.append(f"  > {cls.docstring.split(chr(10))[0]}")
                for method in cls.methods:
                    params_str = ", ".join(method.params)
                    output_parts.append(f"  - `{method.name}({params_str})` → L{method.line}")

        if analysis.functions:
            output_parts.append("")
            output_parts.append("### Function Overview")
            for func in analysis.functions:
                params_str = ", ".join(func.params)
                output_parts.append(f"- **`{func.name}({params_str})`** → L{func.line}")
                if func.docstring:
                    output_parts.append(f"  > {func.docstring.split(chr(10))[0]}")
                output_parts.append(f"  - Complexity: {func.complexity}, Body: {func.body_lines} lines")

        if analysis.imports:
            output_parts.append("")
            output_parts.append("### Dependencies")
            for imp in analysis.imports:
                if imp.module:
                    names = ", ".join(imp.names)
                    output_parts.append(f"- `from {imp.module} import {names}` L{imp.line}")

        # Line-by-line summary for files under 100 lines
        if len(lines) <= 100:
            output_parts.append("")
            output_parts.append("### Line-by-Line Summary")
            for i, line_text in enumerate(lines, 1):
                stripped = line_text.strip()
                if not stripped:
                    continue
                label = self._classify_line(stripped)
                output_parts.append(f"  **L{i}** [{label}] `{stripped[:80]}`")

        output = "\n".join(output_parts)
        return AgentResult(
            success=True, output=output, agent_name=self.name,
            metadata={
                "action": "explain",
                "file_path": file_path,
                "language": lang,
                "functions": len(analysis.functions),
                "classes": len(analysis.classes),
            },
        )

    # ------------------------------------------------------------------
    # Review
    # ------------------------------------------------------------------

    async def _action_review(
        self,
        task: str,
        file_path: str,
        root: str | None,
    ) -> AgentResult:
        if not file_path:
            return AgentResult(
                success=False, output="", agent_name=self.name,
                error="No file_path provided. Set context['file_path'].",
            )

        source = _safe_read(file_path, root)
        lang = _detect_language_from_path(file_path)

        if lang == "python":
            analysis = _analyze_python_ast(source, file_path)
        else:
            analysis = _analyze_generic(source, lang, file_path)

        if not analysis.issues:
            output = (
                f"## Review: {file_path}\n\n"
                f"No issues found. The code looks clean.\n"
            )
        else:
            output_parts = [f"## Review: {file_path}", ""]
            by_severity: dict[str, list[CodeIssue]] = defaultdict(list)
            for iss in analysis.issues:
                by_severity[iss.severity].append(iss)

            for sev in ["critical", "high", "medium", "low", "info"]:
                items = by_severity.get(sev, [])
                if not items:
                    continue
                label = sev.upper()
                output_parts.append(f"### {label} ({len(items)})")
                for iss in items:
                    loc = f"L{iss.line}" if iss.line else "file"
                    output_parts.append(f"- **[{iss.issue_type}]** {loc}: {iss.message}")
                    if iss.suggestion:
                        output_parts.append(f"  → {iss.suggestion}")
                    if iss.code:
                        output_parts.append(f"  ```\n{iss.code}\n  ```")
                output_parts.append("")

            output = "\n".join(output_parts)

        return AgentResult(
            success=True, output=output, agent_name=self.name,
            metadata={
                "action": "review",
                "file_path": file_path,
                "language": lang,
                "total_issues": len(analysis.issues),
                "issues": [
                    {"type": i.issue_type, "severity": i.severity,
                     "line": i.line, "message": i.message}
                    for i in analysis.issues
                ],
            },
        )

    # ------------------------------------------------------------------
    # Refactor
    # ------------------------------------------------------------------

    async def _action_refactor(
        self,
        task: str,
        file_path: str,
        root: str | None,
    ) -> AgentResult:
        if not file_path:
            return AgentResult(
                success=False, output="", agent_name=self.name,
                error="No file_path provided. Set context['file_path'].",
            )

        source = _safe_read(file_path, root)
        lang = _detect_language_from_path(file_path)

        suggestions: list[RefactoringSuggestion] = []
        if lang == "python":
            analysis = _analyze_python_ast(source, file_path)
            suggestions = _suggest_python_refactorings(analysis)
        else:
            analysis = _analyze_generic(source, lang, file_path)
            suggestions = []

        if not suggestions:
            output = (
                f"## Refactoring Suggestions: {file_path}\n\n"
                f"No refactoring opportunities detected.\n"
            )
        else:
            output_parts = [f"## Refactoring Suggestions: {file_path}", ""]
            for i, s in enumerate(suggestions, 1):
                output_parts.append(f"### {i}. {s.ref_type.replace('_', ' ').title()}")
                output_parts.append(f"**Line:** L{s.target_line}" if s.target_line else "")
                output_parts.append(f"**Description:** {s.description}")
                output_parts.append(f"**Motivation:** {s.motivation}")
                if s.suggested_code:
                    output_parts.append(f"**Suggestion:**")
                    output_parts.append(f"```\n{s.suggested_code}\n```")
                output_parts.append("")
            output = "\n".join(output_parts)

        return AgentResult(
            success=True, output=output, agent_name=self.name,
            metadata={
                "action": "refactor",
                "file_path": file_path,
                "language": lang,
                "suggestions": [
                    {"type": s.ref_type, "description": s.description, "line": s.target_line}
                    for s in suggestions
                ],
            },
        )

    # ------------------------------------------------------------------
    # Detect bugs
    # ------------------------------------------------------------------

    async def _action_detect_bugs(
        self,
        task: str,
        file_path: str,
        root: str | None,
    ) -> AgentResult:
        if not file_path:
            return AgentResult(
                success=False, output="", agent_name=self.name,
                error="No file_path provided. Set context['file_path'].",
            )

        source = _safe_read(file_path, root)
        lang = _detect_language_from_path(file_path)

        if lang == "python":
            analysis = _analyze_python_ast(source, file_path)
        else:
            analysis = _analyze_generic(source, lang, file_path)

        bug_issues = [i for i in analysis.issues if i.issue_type in ("bug", "security")]

        if not bug_issues:
            output = (
                f"## Bug Detection: {file_path}\n\n"
                f"No bugs or security issues detected.\n"
            )
        else:
            output_parts = [f"## Bug Detection: {file_path}", ""]
            for iss in bug_issues:
                loc = f"L{iss.line}" if iss.line else "file"
                output_parts.append(f"### [{iss.severity.upper()}] {loc}")
                output_parts.append(f"**Type:** {iss.issue_type}")
                output_parts.append(f"**Message:** {iss.message}")
                output_parts.append(f"**Suggestion:** {iss.suggestion}")
                if iss.code:
                    output_parts.append(f"```\n{iss.code}\n```")
                output_parts.append("")
            output = "\n".join(output_parts)

        return AgentResult(
            success=True, output=output, agent_name=self.name,
            metadata={
                "action": "detect_bugs",
                "file_path": file_path,
                "language": lang,
                "bugs_found": len(bug_issues),
                "issues": [
                    {"type": i.issue_type, "severity": i.severity,
                     "line": i.line, "message": i.message}
                    for i in bug_issues
                ],
            },
        )

    # ------------------------------------------------------------------
    # Suggest tests
    # ------------------------------------------------------------------

    async def _action_suggest_tests(
        self,
        task: str,
        file_path: str,
        root: str | None,
    ) -> AgentResult:
        if not file_path:
            return AgentResult(
                success=False, output="", agent_name=self.name,
                error="No file_path provided. Set context['file_path'].",
            )

        source = _safe_read(file_path, root)
        lang = _detect_language_from_path(file_path)

        test_suggestions: list[TestSuggestion] = []
        if lang == "python":
            analysis = _analyze_python_ast(source, file_path)
            for func in analysis.functions:
                test_suggestions.extend(_generate_python_tests(func, file_path))
        else:
            analysis = _analyze_generic(source, lang, file_path)
            for func in analysis.functions[:5]:
                test_suggestions.append(TestSuggestion(
                    name=f"test_{func.name}",
                    target=func.name,
                    test_type="unit",
                    description=f"Unit test for {func.name}",
                    code=f"// TODO: write tests for {func.name}",
                ))

        if not test_suggestions:
            output = (
                f"## Test Suggestions: {file_path}\n\n"
                f"No testable functions found in this file.\n"
            )
        else:
            output_parts = [f"## Test Suggestions: {file_path}", ""]
            output_parts.append(f"Found {len(test_suggestions)} test suggestion(s).\n")
            for ts in test_suggestions:
                output_parts.append(f"### {ts.name}")
                output_parts.append(f"**Target:** `{ts.target}` | **Type:** {ts.test_type}")
                output_parts.append(f"**Description:** {ts.description}")
                output_parts.append(f"**Coverage note:** {ts.coverage_note}")
                output_parts.append(f"```python\n{ts.code}\n```")
                output_parts.append("")
            output = "\n".join(output_parts)

        return AgentResult(
            success=True, output=output, agent_name=self.name,
            metadata={
                "action": "suggest_tests",
                "file_path": file_path,
                "language": lang,
                "test_count": len(test_suggestions),
                "tests": [
                    {"name": t.name, "target": t.target, "type": t.test_type}
                    for t in test_suggestions
                ],
            },
        )

    # ------------------------------------------------------------------
    # Document
    # ------------------------------------------------------------------

    async def _action_document(
        self,
        task: str,
        file_path: str,
        root: str | None,
    ) -> AgentResult:
        if not file_path:
            return AgentResult(
                success=False, output="", agent_name=self.name,
                error="No file_path provided. Set context['file_path'].",
            )

        source = _safe_read(file_path, root)
        lang = _detect_language_from_path(file_path)

        blocks: list[DocumentationBlock] = []
        if lang == "python":
            analysis = _analyze_python_ast(source, file_path)
            blocks = _generate_docstring_python(analysis)
        else:
            analysis = _analyze_generic(source, lang, file_path)
            blocks = []

        if not blocks:
            output = (
                f"## Documentation: {file_path}\n\n"
                f"All code units already have documentation.\n"
            )
        else:
            output_parts = [f"## Documentation: {file_path}", ""]
            output_parts.append(f"Found {len(blocks)} undocumented item(s).\n")
            for block in blocks:
                output_parts.append(f"### {block.target}: `{block.target_name}`")
                output_parts.append("")
                output_parts.append(f"```python")
                output_parts.append(f'"""{block.docstring}"""')
                if block.params:
                    output_parts.append("")
                    for p_name, p_desc in block.params:
                        output_parts.append(f":param {p_name}: {p_desc}")
                if block.returns:
                    output_parts.append(f":return: {block.returns}")
                output_parts.append("```")
                output_parts.append("")
            output = "\n".join(output_parts)

        return AgentResult(
            success=True, output=output, agent_name=self.name,
            metadata={
                "action": "document",
                "file_path": file_path,
                "language": lang,
                "documented_items": len(blocks),
                "blocks": [
                    {"target": b.target, "target_name": b.target_name, "docstring": b.docstring}
                    for b in blocks
                ],
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_name_hint(task: str) -> str:
        words = task.split()
        for keyword in ["called", "named", "function", "class"]:
            if keyword in words:
                idx = words.index(keyword)
                if idx + 1 < len(words):
                    return words[idx + 1].strip("`'\".,;:!?")
        return "main"

    @staticmethod
    def _classify_line(line: str) -> str:
        """Classify a line of source code by its syntactic role."""
        stripped = line.strip()
        if not stripped:
            return "blank"
        if stripped.startswith(("def ", "async def ")):
            return "function_def"
        if stripped.startswith("class "):
            return "class_def"
        if stripped.startswith(("import ", "from ")):
            return "import"
        if stripped.startswith(("#", "//", "/*", "*", "///")):
            return "comment"
        if stripped.startswith(('"""', "'''", '"""')):
            return "docstring"
        if stripped.startswith(("@",)):
            return "decorator"
        if stripped.startswith(("return", "yield", "raise")):
            return "control_flow"
        if stripped.startswith(("if ", "elif ", "else:", "for ", "while ", "try:", "except", "finally", "with ")):
            return "control_flow"
        if stripped.startswith(("pass", "break", "continue")):
            return "control_flow"
        return "code"
