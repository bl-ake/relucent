# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
import re
from importlib.metadata import version as _package_version

project = "relucent"
copyright = "2026, Blake B. Gaines"
author = "Blake B. Gaines"
release = _package_version("relucent")
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
]

autosectionlabel_prefix_document = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "networkx": ("https://networkx.org/documentation/stable", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
    "plotly": ("https://plotly.com/python-api-reference", None),
    "gurobi": ("https://docs.gurobi.com/projects/optimizer/en/current", None),
}

# Annotations name third-party types by import alias (``np.ndarray``) or by private module
# path (``plotly.graph_objs._figure.Figure``); point those at the names intersphinx knows.
_REFERENCE_ALIASES = [
    (re.compile(r"^np\.(.+)$"), r"numpy.\1"),
    (re.compile(r"^numpy\._typing(?:\.\w+)*\.(\w+)$"), r"numpy.typing.\1"),
    (re.compile(r"^nx\.(.+)$"), r"networkx.\1"),
    (re.compile(r"^nn\.(.+)$"), r"torch.nn.\1"),
    (re.compile(r"^(ConvexHull|HalfspaceIntersection)$"), r"scipy.spatial.\1"),
    (re.compile(r"^go\.(.+)$"), r"plotly.graph_objects.\1"),
    (re.compile(r"^plotly\.graph_objs\._\w+\.(\w+)$"), r"plotly.graph_objects.\1"),
    (re.compile(r"^gurobipy\._core\.(\w+)$"), r"\1"),
]

# ``Literal`` aliases from undocumented modules; the accepted values are listed where they are used.
nitpick_ignore = [
    ("py:class", "CubeMode"),
    # Private node type in a signature of the experimental boundary-search trie.
    ("py:class", "relucent.search.boundary_exclusion_trie._TrieNode"),
    # Type-checking-only alias in convert()'s signature (defining it at runtime would import torch);
    # the docstring lists the accepted (weight, bias) pair forms.
    ("py:class", "AffineLayerPair"),
]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    "show-inheritance": True,
}

autodoc_member_order = "groupwise"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_title = "Relucent Documentation"
html_theme_options = {"navigation_depth": 3, "collapse_navigation": False}
html_static_path = ["_static"]
html_extra_path = ["icon.svg"]
html_js_files = ["custom.js"]


def process_docstring(app, what_, name, obj, options, lines):
    """Strip xdoctest directives from docstrings before Sphinx renders them."""
    remove_directives = [
        re.compile(r"\s*>>>\s*#\s*x?doctest:\s*.*"),
        re.compile(r"\s*>>>\s*#\s*x?doc:\s*.*"),
    ]
    filtered_lines = [line for line in lines if not any(pat.match(line) for pat in remove_directives)]
    lines[:] = filtered_lines
    if lines and lines[-1].strip():
        lines.append("")


def resolve_aliased_reference(app, env, node, contnode):
    """Retry an unresolved reference to a third-party type under its documented name."""
    from sphinx.ext.intersphinx import missing_reference

    target = node.get("reftarget", "")
    for pattern, replacement in _REFERENCE_ALIASES:
        if pattern.match(target):
            node["reftarget"] = pattern.sub(replacement, target)
            resolved = missing_reference(app, env, node, contnode)
            if resolved is None:  # e.g. numpy documents ArrayLike as data, not a class
                node["reftype"] = "obj"
                resolved = missing_reference(app, env, node, contnode)
            return resolved
    return None


def setup(app):
    """Connect the docstring and reference hooks."""
    app.connect("autodoc-process-docstring", process_docstring)
    app.connect("missing-reference", resolve_aliased_reference)
