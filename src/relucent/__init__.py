def _require_torch() -> None:
    """Import torch or raise a friendly error.

    This is intentionally wrapped in a function so the exception binding
    doesn’t leak a module-level name into the public interface.
    """

    try:
        from torch import __version__ as _torch_version  # noqa: F401  # pyright: ignore[reportUnusedImport]
    except ImportError as exc:
        raise ImportError(
            "Relucent requires PyTorch to be installed manually. "
            + "Please install the version compatible with your system from: "
            + "https://pytorch.org/get-started/previous-versions/#:~:text=org/whl/cpu-,v2.3.0"
        ) from exc


_require_torch()
del _require_torch

from . import config  # noqa: E402
from .complex import Complex  # noqa: E402
from .config import update_settings  # noqa: E402
from .convert_model import convert  # noqa: E402
from .model import NN, get_mlp_model  # noqa: E402
from .poly import Polyhedron  # noqa: E402
from .ss import SSManager  # noqa: E402
from .utils import get_env, set_seeds, split_sequential  # noqa: E402
from .vis import data_graph, get_colors, plot_complex, plot_polyhedron  # noqa: E402

__all__ = [
    "Complex",
    "Polyhedron",
    "NN",
    "SSManager",
    "config",
    "update_settings",
    "convert",
    "data_graph",
    "get_colors",
    "get_env",
    "get_mlp_model",
    "plot_complex",
    "plot_polyhedron",
    "set_seeds",
    "split_sequential",
]
