"""JIT-compiled C backend for GF(2) rank.

On first use, ``_gf2_rank.c`` (next to this file) is compiled with ``gcc`` to a shared
library and loaded with :mod:`ctypes`. If that fails, :func:`available` returns ``False``
and callers fall back to pure Python.

The library is cached in ``$RELUCENT_CACHE_DIR`` if set, else the user cache directory
(``~/.cache/relucent`` on Linux), else the system temp directory. It is built with
``-march=native``, so the file name is keyed by the source, the compiler flags, and the
CPU's feature flags: a shared install used from different CPUs builds one copy per CPU type.

Public API
----------
available() -> bool
gf2_rank_packed_c(packed, ncols, *, progress, progress_desc) -> int
gf2_rank_boundary_c(packed, ncols, *, progress, progress_desc) -> int
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from relucent._internal.logging import logger

if TYPE_CHECKING:
    import numpy as np

_HERE = Path(__file__).parent
_C_SRC = _HERE / "_gf2_rank.c"

# Lazily initialised: None = not yet attempted, False = failed, CDLL = success.
_lib: ctypes.CDLL | None | bool = None
_lib_lock = threading.Lock()
_ProgressFn: Any = None


_CFLAGS = ("-O3", "-march=native", "-shared", "-fPIC")
_SUFFIX = ".dll" if sys.platform == "win32" else ".so"


def _cpu_fingerprint() -> str:
    """What ``-march=native`` targets: the CPU's feature flags where the OS reports them."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.split(":", 1)[0].strip() in ("flags", "Features"):
                    return line.strip()
    except OSError:
        pass
    return platform.processor()


def _so_name() -> str:
    key = hashlib.sha1(_C_SRC.read_bytes())
    for part in (*_CFLAGS, platform.machine(), _cpu_fingerprint()):
        key.update(b"\0" + part.encode())
    return f"_gf2_rank_{key.hexdigest()[:16]}{_SUFFIX}"


def _cache_dirs() -> list[Path]:
    """Candidate cache directories, most preferred first."""
    dirs: list[Path] = []
    if override := os.getenv("RELUCENT_CACHE_DIR"):
        dirs.append(Path(override))
    if sys.platform == "win32":
        dirs.append(Path(os.getenv("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "relucent" / "Cache")
    elif sys.platform == "darwin":
        dirs.append(Path.home() / "Library" / "Caches" / "relucent")
    else:
        dirs.append(Path(os.getenv("XDG_CACHE_HOME") or Path.home() / ".cache") / "relucent")
    uid = os.getuid() if hasattr(os, "getuid") else ""
    dirs.append(Path(tempfile.gettempdir()) / f"relucent-{uid}")
    return dirs


def _compile(so: Path) -> bool:
    """Build ``so`` atomically, so concurrent processes never load a half-written library."""
    try:
        so.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=so.parent, prefix=f"{so.stem}.", suffix=".tmp")
        os.close(fd)
    except OSError:
        return False
    try:
        # Try with OpenMP first (uses all available cores in the sweep step),
        # fall back to a single-threaded build if -fopenmp is unavailable.
        for extra in (["-fopenmp"], []):
            cmd = ["gcc", *_CFLAGS, *extra, str(_C_SRC), "-o", tmp]
            try:
                result = subprocess.run(cmd, capture_output=True, timeout=60)
            except (OSError, subprocess.SubprocessError):
                return False
            if result.returncode == 0:
                os.replace(tmp, so)
                return True
        return False
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _find_or_build() -> Path | None:
    name = _so_name()
    dirs = _cache_dirs()
    for d in dirs:
        if (d / name).exists():
            return d / name
    for d in dirs:
        if _compile(d / name):
            return d / name
    return None


def _load_lib() -> ctypes.CDLL | None:
    """Compile (if needed) and load the shared library; return None on failure."""
    global _lib, _ProgressFn
    with _lib_lock:
        if _lib is not None:
            return _lib if isinstance(_lib, ctypes.CDLL) else None
        try:
            so = _find_or_build()
            if so is None:
                _lib = False
                logger.warning("relucent: could not build the GF(2) C backend with gcc; using the slower pure-Python rank.")
                return None
            lib = ctypes.CDLL(str(so))

            # int gf2_rank_packed(uint64_t*, int nrows, int ncols,
            #                     progress_fn, void*, int cb_interval)
            progress_fn_type = ctypes.CFUNCTYPE(
                ctypes.c_int,
                ctypes.c_int,  # col
                ctypes.c_int,  # ncols
                ctypes.c_void_p,  # userdata
            )
            lib.gf2_rank_packed.restype = ctypes.c_int
            lib.gf2_rank_packed.argtypes = [
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.c_int,
                ctypes.c_int,
                progress_fn_type,
                ctypes.c_void_p,
                ctypes.c_int,
            ]
            _ProgressFn = progress_fn_type

            # void gf2_transpose_packed(const uint64_t*, int, int, uint64_t*)
            lib.gf2_transpose_packed.restype = None
            lib.gf2_transpose_packed.argtypes = [
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_uint64),
            ]

            _lib = lib
        except Exception as exc:
            _lib = False
            logger.warning("relucent: could not load the GF(2) C backend (%s); using the slower pure-Python rank.", exc)
            return None
    return _lib  # type: ignore[return-value]


def available() -> bool:
    """Return True if the C backend compiled and loaded successfully."""
    return _load_lib() is not None


def gf2_rank_packed_c(
    packed: np.ndarray,
    ncols: int,
    *,
    progress: bool = False,
    progress_desc: str | None = None,
) -> int:
    """GF(2) rank via the C backend.  Raises RuntimeError if C is unavailable."""
    import numpy as np

    lib = _load_lib()
    if lib is None:
        raise RuntimeError("C GF(2) backend not available")

    arr = np.ascontiguousarray(packed, dtype=np.uint64)
    nrows = int(arr.shape[0])
    ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))

    if _ProgressFn is None:
        raise RuntimeError("C GF(2) backend not available")
    if not progress:
        return int(lib.gf2_rank_packed(ptr, nrows, ncols, _ProgressFn(), None, 0))

    # Progress bar via callback.  ctypes releases the GIL so the callback
    # thread runs concurrently with C.
    from tqdm.auto import tqdm

    desc = progress_desc or "GF(2) rank [C]"
    # Use ~200 updates regardless of matrix size.
    cb_interval = max(1, ncols // 200)

    pbar = tqdm(total=ncols, desc=desc, leave=False, unit="col")

    @_ProgressFn
    def _cb(col: int, _total: int, _ud: ctypes.c_void_p) -> int:
        pbar.n = col
        pbar.refresh()
        return 0

    try:
        rank = int(lib.gf2_rank_packed(ptr, nrows, ncols, _cb, None, cb_interval))
    finally:
        pbar.n = ncols
        pbar.refresh()
        pbar.close()

    return rank


def gf2_transpose_packed_c(packed: np.ndarray, src_ncols: int) -> tuple[np.ndarray, int]:
    """Transpose a bit-packed GF(2) matrix using the C backend.

    Returns ``(transposed, dst_ncols)`` where ``dst_ncols == src_nrows``.
    """
    import numpy as np

    lib = _load_lib()
    if lib is None:
        raise RuntimeError("C GF(2) backend not available")

    src = np.ascontiguousarray(packed, dtype=np.uint64)
    src_nrows = int(src.shape[0])
    dst_ncols = src_nrows
    dst_nwords = (dst_ncols + 63) // 64
    dst = np.zeros((src_ncols, dst_nwords), dtype=np.uint64)

    src_ptr = src.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
    dst_ptr = dst.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
    lib.gf2_transpose_packed(src_ptr, src_nrows, src_ncols, dst_ptr)

    return dst, dst_ncols


def gf2_rank_boundary_c(
    packed: np.ndarray,
    ncols: int,
    *,
    progress: bool = False,
    progress_desc: str | None = None,
) -> int:
    """Rank of a boundary matrix, transposing first when that reduces column count."""
    nrows = int(packed.shape[0])
    if nrows > ncols and ncols > 0:
        desc = f"{progress_desc} (A^T)" if progress_desc else "GF(2) rank [C] (A^T)"
        transposed, ncols_t = gf2_transpose_packed_c(packed, ncols)
        return gf2_rank_packed_c(transposed, ncols_t, progress=progress, progress_desc=desc)
    return gf2_rank_packed_c(packed, ncols, progress=progress, progress_desc=progress_desc)
