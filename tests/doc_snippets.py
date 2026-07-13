"""Extract and execute Python snippets from Relucent documentation sources."""

from __future__ import annotations

import re
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"
README_PATH = REPO_ROOT / "README.md"


def _dedent_rst_block(block: str) -> str:
    lines = block.splitlines()
    if not lines:
        return ""
    return "\n".join(line[3:] if line.startswith("   ") else line for line in lines).rstrip() + "\n"


def _collect_indented_block(lines: list[str], start: int) -> tuple[str, int]:
    """Collect RST literal/code content until a non-indented non-blank line."""
    block_lines: list[str] = []
    index = start
    while index < len(lines):
        line = lines[index]
        if line.strip() == "" or line.startswith("   "):
            block_lines.append(line)
            index += 1
            continue
        break
    return _dedent_rst_block("\n".join(block_lines)), index


def extract_rst_python_blocks(path: Path) -> list[str]:
    """Return Python code blocks from ``.. code-block:: python`` directives."""
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks: list[str] = []
    index = 0
    while index < len(lines):
        if lines[index].strip() == ".. code-block:: python":
            index += 1
            while index < len(lines) and lines[index].strip() == "":
                index += 1
            block, index = _collect_indented_block(lines, index)
            if block.strip():
                blocks.append(block)
            continue
        index += 1
    return blocks


def extract_rst_literal_python_blocks(path: Path) -> list[str]:
    """Return Python literal blocks introduced by ``::`` (skip shell ``export`` lines)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks: list[str] = []
    index = 0
    while index < len(lines):
        if lines[index].rstrip().endswith("::") and not lines[index].startswith(".."):
            index += 1
            while index < len(lines) and lines[index].strip() == "":
                index += 1
            if index >= len(lines) or not lines[index].startswith("   "):
                continue
            block, index = _collect_indented_block(lines, index)
            stripped = block.strip()
            if not stripped or stripped.startswith("export "):
                continue
            blocks.append(block)
            continue
        index += 1
    return blocks


def extract_markdown_python_blocks(path: Path) -> list[str]:
    """Return fenced Python code blocks from Markdown."""
    text = path.read_text(encoding="utf-8")
    blocks: list[str] = []
    for match in re.finditer(r"```(?:python)?\n(.*?)```", text, flags=re.DOTALL):
        code = match.group(1).rstrip() + "\n"
        stripped = code.strip()
        if not stripped:
            continue
        first_line = stripped.splitlines()[0]
        if first_line.startswith("export ") or first_line.startswith("pip "):
            continue
        if first_line.startswith("@") or first_line.startswith("%"):
            continue
        blocks.append(code)
    return blocks


_STUB_FIG = SimpleNamespace(show=lambda *args, **kwargs: None)


@contextmanager
def _stub_plotting():
    with (
        patch("relucent.core.complex.Complex.plot", return_value=_STUB_FIG),
        patch("relucent.topology.persistence.PersistenceDiagram.plot", return_value=_STUB_FIG),
    ):
        yield


def exec_snippet(
    code: str,
    *,
    namespace: dict[str, Any] | None = None,
    as_main: bool = False,
) -> dict[str, Any]:
    """Execute documentation Python exactly as written."""
    ns: dict[str, Any] = {} if namespace is None else namespace
    if as_main:
        ns["__name__"] = "__main__"
    with _stub_plotting():
        exec(code, ns)  # noqa: S102 — intentional execution of trusted doc snippets
    return ns


def run_snippet_sequence(
    snippets: list[str],
    *,
    as_main_indices: set[int] | None = None,
    initial_namespace: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute snippets in order, sharing one namespace."""
    ns: dict[str, Any] = {} if initial_namespace is None else dict(initial_namespace)
    as_main_indices = as_main_indices or set()
    with _stub_plotting():
        for index, snippet in enumerate(snippets):
            if index in as_main_indices:
                ns["__name__"] = "__main__"
            exec(snippet, ns)  # noqa: S102
    return ns
