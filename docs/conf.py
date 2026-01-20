# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# import sys
# import os

autodoc_default_options = {
    "members": True,
    "undoc-members": False,  # Set to True to include members without docstrings
    "private-members": False,  # Set to True to include private members
    "show-inheritance": True,  # Show base classes
}


# sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
# sys.path.insert(0, "../src/relucent/")
# sys.path.insert(0, os.path.abspath("../src/relucent/"))
# sys.path.insert(0, os.path.abspath(".."))

project = "relucent"
copyright = "2026, Blake B. Gaines"
author = "Blake B. Gaines"

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

autosummary_filename_map = {
    "relucent.Complex": "relucent.complex.Complex",
    "relucent.Polyhedron": "relucent.poly.Polyhedron",
}

autosummary_generate = True

autodoc_member_order = "groupwise"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]

# Configure sidebar to show local table of contents
html_sidebars = {
    "**": [
        "localtoc.html",
        "relations.html",
        "sourcelink.html",
        "searchbox.html",
    ]
}


def process_docstring(app, what_, name, obj, options, lines):
    """
    Custom process to transform docstring lines Remove "Ignore" blocks

    Args:
        app (sphinx.application.Sphinx): the Sphinx application object

        what (str):
            the type of the object which the docstring belongs to (one of
            "module", "class", "exception", "function", "method", "attribute")

        name (str): the fully qualified name of the object

        obj: the object itself

        options: the options given to the directive: an object with
            attributes inherited_members, undoc_members, show_inheritance
            and noindex that are true if the flag option of same name was
            given to the auto directive

        lines (List[str]): the lines of the docstring, see above

    References:
        https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
    """
    import re

    remove_directives = [
        # Remove all xdoctest directives
        re.compile(r"\s*>>>\s*#\s*x?doctest:\s*.*"),
        re.compile(r"\s*>>>\s*#\s*x?doc:\s*.*"),
    ]
    filtered_lines = [line for line in lines if not any(pat.match(line) for pat in remove_directives)]
    # Modify the lines inplace
    lines[:] = filtered_lines

    # make sure there is a blank line at the end
    if lines and lines[-1].strip():
        lines.append("")


def setup(app):
    """Connect the process_docstring function to Sphinx's autodoc event"""
    app.connect("autodoc-process-docstring", process_docstring)
