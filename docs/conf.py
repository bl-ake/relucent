# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
project = "relucent"
copyright = "2026, Blake B. Gaines"
author = "Blake B. Gaines"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    "show-inheritance": True,
}

autodoc_member_order = "groupwise"

autosummary_generate = True
autosummary_filename_map = {
    "relucent.Complex": "relucent.complex.Complex",
    "relucent.Polyhedron": "relucent.poly.Polyhedron",
}

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
    import re

    remove_directives = [
        re.compile(r"\s*>>>\s*#\s*x?doctest:\s*.*"),
        re.compile(r"\s*>>>\s*#\s*x?doc:\s*.*"),
    ]
    filtered_lines = [line for line in lines if not any(pat.match(line) for pat in remove_directives)]
    lines[:] = filtered_lines
    if lines and lines[-1].strip():
        lines.append("")


def setup(app):
    """Connect the process_docstring hook to Sphinx's autodoc event."""
    app.connect("autodoc-process-docstring", process_docstring)
