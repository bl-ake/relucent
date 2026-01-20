try:
    from torch import __version__
    from torchvision import __version__  # noqa
except ImportError:
    raise ImportError(
        "Relucent requires PyTorch to be installed manually. "
        "Please install the version compatible with your system from: "
        "https://pytorch.org/get-started/previous-versions/#:~:text=org/whl/cpu-,v2.3.0"
    )

from .bvs import BVManager
from .complex import Complex
from .poly import Polyhedron
from .model import NN, get_mlp_model
from .convert_model import convert
from .utils import get_env, split_sequential, set_seeds

__all__ = [Complex, Polyhedron, NN, get_mlp_model, BVManager, convert, get_env, split_sequential, set_seeds]
