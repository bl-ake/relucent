"""Test that relucent public API is importable."""

import importlib


def test_top_level_imports():
    from relucent import (
        CertifyLevel,
        Complex,
        ComplexNotCompleteError,
        IncompleteDualGraphError,
        Polyhedron,
        SearchResult,
        add_output_relu,
        convert,
        mlp,
        set_seeds,
        split_sequential,
    )

    assert Complex is not None
    assert Polyhedron is not None
    assert SearchResult is not None
    assert issubclass(ComplexNotCompleteError, Exception)
    assert issubclass(IncompleteDualGraphError, Exception)
    assert CertifyLevel.COMPLETE is not None
    assert callable(add_output_relu)
    assert callable(convert)
    assert callable(mlp)
    assert callable(set_seeds)
    assert callable(split_sequential)


def test_every_name_in_all_resolves():
    for module_name in ("relucent", "relucent.core", "relucent.search", "relucent.topology", "relucent.config"):
        module = importlib.import_module(module_name)
        for name in module.__all__:
            assert hasattr(module, name), f"{module_name}.{name}"
