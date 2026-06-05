"""Run exact copies of Python snippets from Relucent documentation."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.doc_snippets import (
    DOCS_DIR,
    README_PATH,
    exec_snippet,
    extract_markdown_python_blocks,
    extract_rst_literal_python_blocks,
    extract_rst_python_blocks,
    run_snippet_sequence,
)

os.environ.setdefault("DISABLE_RESEARCH_WARNING", "1")


def _rst(name: str) -> Path:
    return DOCS_DIR / name


@pytest.mark.parametrize(
    "path,extractor",
    [
        (README_PATH, extract_markdown_python_blocks),
        (_rst("quickstart.rst"), extract_rst_python_blocks),
        (_rst("topology.rst"), extract_rst_python_blocks),
        (_rst("search_geometry.rst"), extract_rst_python_blocks),
        (_rst("network_definitions.rst"), extract_rst_python_blocks),
        (_rst("configuration.rst"), extract_rst_literal_python_blocks),
    ],
    ids=[
        "README.md",
        "quickstart.rst",
        "topology.rst",
        "search_geometry.rst",
        "network_definitions.rst",
        "configuration.rst",
    ],
)
def test_doc_source_has_python_snippets(path: Path, extractor) -> None:
    assert extractor(path), f"No Python snippets found in {path.name}"


def test_readme_snippets() -> None:
    snippets = extract_markdown_python_blocks(README_PATH)
    assert len(snippets) == 4
    ns = run_snippet_sequence(snippets, as_main_indices={0})
    assert "cplx" in ns
    assert len(ns["cplx"]) > 0


def test_quickstart_snippets() -> None:
    snippets = extract_rst_python_blocks(_rst("quickstart.rst"))
    assert len(snippets) == 3
    ns = run_snippet_sequence(snippets, as_main_indices={0})
    assert "cplx" in ns
    assert len(ns["cplx"]) > 0
    assert "p" in ns


def test_topology_snippets() -> None:
    snippets = extract_rst_python_blocks(_rst("topology.rst"))
    assert len(snippets) == 2
    exec_snippet(snippets[0])
    ns = exec_snippet(snippets[1])
    assert "diagram" in ns
    assert "fig" in ns


def test_search_geometry_snippets() -> None:
    snippets = extract_rst_python_blocks(_rst("search_geometry.rst"))
    assert len(snippets) == 5
    ns = run_snippet_sequence(snippets)
    assert "cplx" in ns
    assert len(ns["cplx"]) > 0


def test_network_definitions_snippets() -> None:
    snippets = extract_rst_python_blocks(_rst("network_definitions.rst"))
    assert len(snippets) == 4
    for snippet in snippets:
        ns = exec_snippet(snippet)
        assert "cplx" in ns
        assert len(ns["cplx"]) >= 0


def test_configuration_snippets() -> None:
    snippets = extract_rst_literal_python_blocks(_rst("configuration.rst"))
    assert len(snippets) == 3
    exec_snippet(snippets[0])
    exec_snippet(snippets[1])
    ns = exec_snippet(snippets[2])
    assert "cx" in ns
    assert len(ns["cx"]) > 0  # populated by cx.searcher(...) calls in the snippet
