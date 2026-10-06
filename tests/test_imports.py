"""Test that relucent public API is importable."""

import importlib
import pickle
import subprocess
import sys

import relucent

# Modules that pool workers import to run their tasks. A worker started with ``spawn`` (Windows, macOS)
# imports them fresh, so any of them importing torch at load time costs every pool seconds.
_WORKER_MODULES = (
    "relucent.core.complex",
    "relucent.geometry.calculations",
    "relucent.graph.incidence",
    "relucent.graph.vertex_star",
    "relucent.model.model",
    "relucent.search.boundary_mip",
    "relucent.search.boundary_search",
    "relucent.search.engine",
    "relucent.search.worker_context",
    "relucent.topology.morse",
    "relucent.verify.certify",
)


def _torch_loaded_after(code: str) -> bool:
    """Run ``code`` in a fresh interpreter and report whether torch ended up imported."""
    script = f"{code}\nimport sys\nprint('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    return out.stdout.strip().splitlines()[-1] == "True"


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
    for module_name in (
        "relucent",
        "relucent.core",
        "relucent.model",
        "relucent.search",
        "relucent.topology",
        "relucent.config",
    ):
        module = importlib.import_module(module_name)
        for name in module.__all__:
            assert hasattr(module, name), f"{module_name}.{name}"


def test_worker_modules_do_not_import_torch():
    assert not _torch_loaded_after("\n".join(f"import {name}" for name in _WORKER_MODULES))


def test_unpickling_a_network_does_not_import_torch():
    payload = pickle.dumps(relucent.mlp(widths=[2, 4, 1]))
    assert not _torch_loaded_after(f"import pickle\npickle.loads({payload!r})")


def test_no_grad_does_not_import_torch():
    code = "from relucent._internal.torch_compat import no_grad\n\n@no_grad\ndef f(x):\n    return x + 1\n\nassert f(1) == 2"
    assert not _torch_loaded_after(code)
